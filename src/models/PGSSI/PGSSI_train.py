"""PGSSI 训练入口。

整体流程：
1. 读取 CSV 数据集并构建带 3D 几何的分子对图。
2. 用 PGSSIModel 预测 log-gamma。
3. 在验证集上做早停，并在训练结束后输出公开/私有测试集指标。
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import platform
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from torch.optim.lr_scheduler import ReduceLROnPlateau as reduce_lr
from torch_geometric.loader import DataLoader
from tqdm import tqdm

try:
    from PGSSI_3D_architecture import PGSSIModel
    from PGSSI_data import GEOM_CACHE_VERSION, build_pair_dataset
    from physics_loss import compute_physics_regularization
except ImportError:
    from src.models.PGSSI.PGSSI_3D_architecture import PGSSIModel
    from src.models.PGSSI.PGSSI_data import GEOM_CACHE_VERSION, build_pair_dataset
    from src.models.PGSSI.physics_loss import compute_physics_regularization


PROJECT_ROOT = Path(__file__).resolve().parents[3]


def count_parameters(model):
    """统计可训练参数量，便于记录模型规模。"""
    return sum(param.numel() for param in model.parameters() if param.requires_grad)


def set_seed(seed=42):
    """固定随机种子，尽量保证训练可复现。"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def configure_torch_runtime():
    """配置 CUDA 运行时，优先提升吞吐。"""
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass


def make_grad_scaler(device):
    """兼容新旧 PyTorch AMP 接口。"""
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        return torch.amp.GradScaler(device.type, enabled=device.type == "cuda")
    return torch.cuda.amp.GradScaler(enabled=device.type == "cuda")


def amp_autocast(device):
    """兼容新旧 PyTorch autocast 接口。"""
    if hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
        return torch.amp.autocast(device_type=device.type, enabled=device.type == "cuda")
    return torch.cuda.amp.autocast(enabled=device.type == "cuda")


def default_num_workers():
    """给 DataLoader 一个稳妥的默认 worker 数。

    Windows 下 PyG + 自定义 Data 对象在多进程 spawn 时启动成本很高，
    worker 开太多反而容易在首个 batch 前卡很久。
    """
    cpu_count = os.cpu_count() or 1
    if platform.system() == "Windows":
        return 0
    return max(2, min(8, cpu_count))


def build_dataset(df, cache_path):
    """把 dataframe 转成 PGSSI 使用的 PyG 图数据列表。"""
    return build_pair_dataset(
        df,
        cache_path,
        include_k_targets=False,
        cache_desc="build PGSSI 3D dataset",
    )


def check_required_columns(df, df_name):
    """训练/验证数据至少要包含分子、温度和监督目标。"""
    required = {"Solvent_SMILES", "Solute_SMILES", "T", "log-gamma"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"{df_name} is missing required columns: {missing}")


def make_run_dir(run_dir=None, prefix="pgssi_train"):
    """解析并创建本次实验的输出目录。"""
    if run_dir is not None:
        final_run_dir = Path(run_dir)
        if not final_run_dir.is_absolute():
            final_run_dir = PROJECT_ROOT / final_run_dir
    else:
        final_run_dir = PROJECT_ROOT / "runs" / prefix
    final_run_dir.mkdir(parents=True, exist_ok=True)
    return final_run_dir


def make_cache_dir(cache_dir=None, prefix="pgssi_shared"):
    """解析并创建几何缓存目录。"""
    if cache_dir is not None:
        final_cache_dir = Path(cache_dir)
        if not final_cache_dir.is_absolute():
            final_cache_dir = PROJECT_ROOT / final_cache_dir
    else:
        final_cache_dir = PROJECT_ROOT / "cache" / prefix
    final_cache_dir.mkdir(parents=True, exist_ok=True)
    return final_cache_dir


def _cache_file(model_name, suffix, cache_dir):
    """统一缓存文件命名，避免不同阶段互相覆盖。"""
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    return str(cache_dir / f"{model_name}_{suffix}.pt")


def dataset_cache_prefix(path_str):
    return Path(path_str).stem


def _resolved_cache_path(cache_path):
    """给缓存文件补上当前几何缓存版本后缀。"""
    cache_path_obj = Path(cache_path)
    if cache_path_obj.suffix == ".pt" and not cache_path_obj.stem.endswith(GEOM_CACHE_VERSION):
        return cache_path_obj.with_name(f"{cache_path_obj.stem}_{GEOM_CACHE_VERSION}.pt")
    return cache_path_obj


def resolve_input_path(path_str):
    """把相对路径解析为项目根目录下的绝对路径，并做存在性检查。"""
    path = Path(path_str)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found: {path}")
    return path


def evaluate_loader(model, loader, device):
    """在一个 DataLoader 上计算 MAE 和 R2。"""
    model.eval()
    all_true, all_pred = [], []
    use_amp = device.type == "cuda"

    with torch.no_grad():
        for batch in loader:
            if batch is None:
                continue
            batch = batch.to(device, non_blocking=use_amp)
            with amp_autocast(device):
                pred = model(batch).view(-1)
            target = batch.y.view(-1).float()
            all_true.append(target.cpu())
            all_pred.append(pred.cpu())

    if not all_true:
        return float("inf"), float("-inf")

    y_true = torch.cat(all_true).numpy()
    y_pred = torch.cat(all_pred).numpy()
    return float(mean_absolute_error(y_true, y_pred)), float(r2_score(y_true, y_pred))


def predict_loader_with_indices(model, loader, device):
    """预测并返回原始样本索引，用于把结果回填到原始 dataframe。"""
    model.eval()
    all_pred = []
    all_indices = []
    use_amp = device.type == "cuda"

    with torch.no_grad():
        for batch in loader:
            if batch is None:
                continue
            batch = batch.to(device, non_blocking=use_amp)
            with amp_autocast(device):
                pred = model(batch).view(-1)
            all_pred.append(pred.cpu())
            if hasattr(batch, "sample_index"):
                all_indices.append(batch.sample_index.view(-1).cpu())

    preds = torch.cat(all_pred).numpy() if all_pred else np.array([], dtype=np.float32)
    indices = torch.cat(all_indices).numpy() if all_indices else np.array([], dtype=np.int64)
    return preds, indices


def evaluate_physics_metrics(model, loader, device):
    model.eval()
    use_amp = device.type == "cuda"
    metric_terms = {
        "physics_reg": [],
        "physics_smoothness": [],
        "physics_short_range_repulsion": [],
        "physics_long_range_decay": [],
        "physics_charge_sign_consistency": [],
        "physics_thermo_derivative": [],
    }

    with torch.no_grad():
        for batch in loader:
            if batch is None:
                continue
            batch = batch.to(device, non_blocking=use_amp)
            with amp_autocast(device):
                outputs = model(batch, return_dict=True)
            physics_reg, terms = compute_physics_regularization(outputs)
            metric_terms["physics_reg"].append(float(physics_reg.detach().cpu()))
            metric_terms["physics_smoothness"].append(float(terms.get("smoothness", 0.0)))
            metric_terms["physics_short_range_repulsion"].append(float(terms.get("short_range_repulsion", 0.0)))
            metric_terms["physics_long_range_decay"].append(float(terms.get("long_range_decay", 0.0)))
            metric_terms["physics_charge_sign_consistency"].append(float(terms.get("charge_sign_consistency", 0.0)))
            metric_terms["physics_thermo_derivative"].append(float(terms.get("thermo_derivative", 0.0)))

    summary = {}
    for key, values in metric_terms.items():
        summary[key] = float(np.mean(values)) if values else float("nan")
    return summary


def evaluate_test_file(
    model_path,
    test_path,
    test_name,
    model_name,
    artifact_prefix,
    hidden_dim,
    enable_cross_interaction,
    num_intra_layers,
    use_interaction_types,
    use_moe,
    use_physics_prior,
    disable_cross_refine,
    topology_only,
    direct_loggamma_head,
    disable_formula_layer,
    output_dir,
    cache_dir,
):
    """在测试集 CSV 上做完整评估并写出预测文件/指标文件。"""
    test_df = pd.read_csv(test_path)
    cache_path = _cache_file(model_name, f"{dataset_cache_prefix(test_path)}_{test_name}", cache_dir)
    dataset = build_dataset(test_df, cache_path)
    # 旧缓存可能没有 sample_index，发现后直接删掉并重建。
    if dataset and not hasattr(dataset[0], "sample_index"):
        stale_cache_path = _resolved_cache_path(cache_path)
        if stale_cache_path.exists():
            try:
                stale_cache_path.unlink()
            except OSError:
                pass
        dataset = build_dataset(test_df, cache_path)

    loader = DataLoader(
        dataset,
        batch_size=32,
        shuffle=False,
        drop_last=False,
        num_workers=0,
        pin_memory=True,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PGSSIModel(
        hidden_dim=hidden_dim,
        enable_cross_interaction=enable_cross_interaction,
        num_intra_layers=num_intra_layers,
        use_interaction_types=use_interaction_types,
        use_moe=use_moe,
        use_physics_prior=use_physics_prior,
        disable_cross_refine=disable_cross_refine,
        topology_only=topology_only,
        direct_loggamma_head=direct_loggamma_head,
        disable_formula_layer=disable_formula_layer,
    ).to(device)
    checkpoint = torch.load(model_path, map_location=device)
    state_dict = checkpoint["model_state_dict"] if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint else checkpoint
    model.load_state_dict(state_dict)

    preds, sample_indices = predict_loader_with_indices(model, loader, device)
    result_df = test_df.copy()
    result_df["pred_log-gamma"] = np.nan
    if sample_indices.size > 0:
        result_df.loc[sample_indices, "pred_log-gamma"] = preds

    metrics = {
        "dataset_name": test_name,
        "dataset_path": str(Path(test_path).resolve()),
        "num_samples_total": int(len(result_df)),
        "num_samples_predicted": int(sample_indices.size),
        "num_samples_skipped": int(len(result_df) - sample_indices.size),
        "prediction_file": str(Path(output_dir) / f"{artifact_prefix}_{test_name}_predictions.csv"),
    }

    if "log-gamma" in result_df.columns:
        valid_mask = result_df["pred_log-gamma"].notna()
        y_true = result_df.loc[valid_mask, "log-gamma"].to_numpy(dtype=float)
        y_pred = result_df.loc[valid_mask, "pred_log-gamma"].to_numpy(dtype=float)
        abs_error = np.abs(y_true - y_pred)
        result_df["abs_error"] = np.nan
        result_df["squared_error"] = np.nan
        result_df.loc[valid_mask, "abs_error"] = abs_error
        result_df.loc[valid_mask, "squared_error"] = np.square(y_true - y_pred)
        if y_true.size > 0:
            metrics["mae"] = float(mean_absolute_error(y_true, y_pred))
            metrics["rmse"] = float(np.sqrt(mean_squared_error(y_true, y_pred)))
            metrics["r2"] = float(r2_score(y_true, y_pred))
            metrics["AE<=0.1"] = float(np.mean(abs_error <= 0.1) * 100.0)
            metrics["AE<=0.2"] = float(np.mean(abs_error <= 0.2) * 100.0)
            metrics["AE<=0.3"] = float(np.mean(abs_error <= 0.3) * 100.0)
            metrics["AE<=0.1_count"] = int(np.sum(abs_error <= 0.1))
            metrics["AE<=0.2_count"] = int(np.sum(abs_error <= 0.2))
            metrics["AE<=0.3_count"] = int(np.sum(abs_error <= 0.3))
            metrics.update(evaluate_physics_metrics(model, loader, device))

    prediction_path = Path(output_dir) / f"{artifact_prefix}_{test_name}_predictions.csv"
    result_df.to_csv(prediction_path, index=False)

    metrics_path = Path(output_dir) / f"{artifact_prefix}_{test_name}_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=2, ensure_ascii=False)
    metrics["metrics_file"] = str(metrics_path)
    return metrics


def print_test_summary(test_results):
    """把测试指标以简洁格式打印到终端。"""
    for dataset_name, metrics in test_results.items():
        tqdm.write(
            f"[{dataset_name}] "
            f"MAE: {metrics.get('mae', float('nan')):.6f} | "
            f"RMSE: {metrics.get('rmse', float('nan')):.6f} | "
            f"R2: {metrics.get('r2', float('nan')):.6f} | "
            f"AE<=0.1: {metrics.get('AE<=0.1', float('nan')):.2f}% | "
            f"AE<=0.2: {metrics.get('AE<=0.2', float('nan')):.2f}% | "
            f"AE<=0.3: {metrics.get('AE<=0.3', float('nan')):.2f}%"
        )


def train_PGSSI(train_df, model_name, hyperparameters, output_dir, cache_dir, resume=False, valid_df=None):
    """训练主函数。

    这里的监督信号只有 log-gamma。
    模型内部仍然会预测 K1/K2，并通过公式层得到最终输出。
    """
    model_dir = Path(output_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    artifact_prefix = hyperparameters.get("artifact_prefix", model_name)

    report_path = model_dir / f"Report_training_{artifact_prefix}.txt"
    report = open(report_path, "w", encoding="utf-8")

    def print_report(message, file=report):
        """同时写终端和训练报告文件。"""
        tqdm.write(str(message))
        file.write("\n" + message)
        file.flush()

    print_report(f" Report for {model_name}")
    print_report("-" * 50)

    hidden_dim = hyperparameters["hidden_dim"]
    lr = hyperparameters["lr"]
    n_epochs = hyperparameters["n_epochs"]
    batch_size = hyperparameters["batch_size"]
    weight_decay = hyperparameters.get("weight_decay", 0.0)
    patience = hyperparameters.get("early_stopping_patience", 20)
    enable_cross_interaction = hyperparameters.get("enable_cross_interaction", True)
    num_intra_layers = int(hyperparameters.get("num_intra_layers", 2))
    use_interaction_types = bool(hyperparameters.get("use_interaction_types", True))
    use_moe = bool(hyperparameters.get("use_moe", True))
    train_num_workers = hyperparameters.get("train_num_workers", default_num_workers())
    valid_num_workers = hyperparameters.get("valid_num_workers", train_num_workers)
    train_cache_prefix = hyperparameters.get("train_cache_prefix", "train")
    valid_cache_prefix = hyperparameters.get("valid_cache_prefix", "valid")
    checkpoint_interval = max(1, int(hyperparameters.get("checkpoint_interval", 10)))
    physics_loss_weight = float(hyperparameters.get("physics_loss_weight", 0.0))
    use_physics_loss = physics_loss_weight > 0.0
    init_model_path = hyperparameters.get("init_model_path")
    quiet_progress = bool(hyperparameters.get("quiet_progress", False))
    use_physics_prior = bool(hyperparameters.get("use_physics_prior", True))
    disable_cross_refine = bool(hyperparameters.get("disable_cross_refine", False))
    topology_only = bool(hyperparameters.get("topology_only", False))
    direct_loggamma_head = bool(hyperparameters.get("direct_loggamma_head", False))
    disable_formula_layer = bool(hyperparameters.get("disable_formula_layer", False))
    start = time.time()
    use_cuda = torch.cuda.is_available()
    loader_kwargs = {
        "num_workers": train_num_workers,
        "pin_memory": use_cuda,
    }
    if train_num_workers > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = 4

    # 训练/验证集都缓存成 PyG 数据，避免每次都重新做 RDKit 3D 构图。
    train_dataset = build_dataset(train_df, _cache_file(model_name, f"{train_cache_prefix}_train", cache_dir))
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
        **loader_kwargs,
    )

    valid_loader = None
    if valid_df is not None:
        valid_loader_kwargs = {
            "num_workers": valid_num_workers,
            "pin_memory": use_cuda,
        }
        if valid_num_workers > 0:
            valid_loader_kwargs["persistent_workers"] = True
            valid_loader_kwargs["prefetch_factor"] = 4
        valid_dataset = build_dataset(valid_df, _cache_file(model_name, f"{valid_cache_prefix}_valid", cache_dir))
        valid_loader = DataLoader(
            valid_dataset,
            batch_size=batch_size,
            shuffle=False,
            drop_last=False,
            **valid_loader_kwargs,
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PGSSIModel(
        hidden_dim=hidden_dim,
        enable_cross_interaction=enable_cross_interaction,
        num_intra_layers=num_intra_layers,
        use_interaction_types=use_interaction_types,
        use_moe=use_moe,
        use_physics_prior=use_physics_prior,
        disable_cross_refine=disable_cross_refine,
        topology_only=topology_only,
        direct_loggamma_head=direct_loggamma_head,
        disable_formula_layer=disable_formula_layer,
    ).to(device)
    print_report(f"Using device: {device}")
    print_report(f"Number of model parameters: {count_parameters(model)}")
    print_report(f"Train num_workers: {train_num_workers}")
    print_report(f"Valid num_workers: {valid_num_workers if valid_df is not None else 0}")
    print_report(f"AMP enabled: {device.type == 'cuda'}")
    print_report(f"Checkpoint save interval: every {checkpoint_interval} epoch(s)")
    print_report(f"Physics loss enabled: {use_physics_loss} | weight={physics_loss_weight}")
    print_report(
        f"Ablation switches | num_intra_layers={num_intra_layers} | "
        f"use_interaction_types={use_interaction_types} | use_moe={use_moe} | "
        f"use_physics_prior={use_physics_prior} | disable_cross_refine={disable_cross_refine} | "
        f"topology_only={topology_only} | direct_loggamma_head={direct_loggamma_head} | "
        f"disable_formula_layer={disable_formula_layer}"
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = reduce_lr(optimizer, mode="min", factor=0.8, patience=3, min_lr=1e-7)
    scaler = make_grad_scaler(device)

    checkpoint_path = model_dir / f"{artifact_prefix}_resume.pth"
    best_model_path = model_dir / f"{artifact_prefix}_best.pth"

    mae_history = []
    r2_history = []
    train_loss_history = []
    valid_mae_history = []
    valid_r2_history = []
    best_mae = np.inf
    best_model = None
    epochs_without_improve = 0
    start_epoch = 0

    if init_model_path and not resume:
        init_model_file = Path(init_model_path)
        if not init_model_file.is_absolute():
            init_model_file = PROJECT_ROOT / init_model_file
        if init_model_file.exists():
            state_obj = torch.load(init_model_file, map_location=device)
            state_dict = state_obj["model_state_dict"] if isinstance(state_obj, dict) and "model_state_dict" in state_obj else state_obj
            model.load_state_dict(state_dict, strict=False)
            print_report(f"Loaded initialization weights from: {init_model_file}")
        else:
            print_report(f"Initialization weights not found: {init_model_file}")

    # resume 只恢复训练状态，不改变本次实验配置。
    if resume and checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        if "scaler_state_dict" in checkpoint:
            scaler.load_state_dict(checkpoint["scaler_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        best_mae = checkpoint["best_mae"]
        mae_history = checkpoint["mae_history"]
        r2_history = checkpoint["r2_history"]
        train_loss_history = checkpoint["train_loss_history"]
        valid_mae_history = checkpoint.get("valid_mae_history", [])
        valid_r2_history = checkpoint.get("valid_r2_history", [])
        best_model = checkpoint.get("best_model_state_dict")
        print_report(f"Resuming training from checkpoint: {checkpoint_path}")
        print_report(f"Resuming training from epoch {start_epoch}")
    elif resume:
        print_report(f"Resume requested but checkpoint not found: {checkpoint_path}")

    for epoch in range(start_epoch, n_epochs):
        model.train()
        loss_sum = 0.0
        n_graphs = 0
        train_true_batches, train_pred_batches = [], []

        batch_bar = tqdm(
            train_loader,
            desc=f"Train {epoch + 1}/{n_epochs}",
            dynamic_ncols=True,
            leave=False,
            disable=quiet_progress,
        )
        for batch in batch_bar:
            if batch is None:
                continue
            batch = batch.to(device, non_blocking=device.type == "cuda")
            with amp_autocast(device):
                outputs = model(batch, return_dict=use_physics_loss)
                pred = outputs["log_gamma"].view(-1) if use_physics_loss else outputs.view(-1)
                target = batch.y.view(-1).float()
                supervised_loss = F.mse_loss(pred, target)
                if use_physics_loss:
                    physics_loss, _ = compute_physics_regularization(outputs)
                    loss = supervised_loss + (physics_loss_weight * physics_loss)
                else:
                    loss = supervised_loss
            target = batch.y.view(-1).float()

            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            batch_graphs = int(target.numel())
            loss_sum += loss.item() * batch_graphs
            n_graphs += batch_graphs
            train_true_batches.append(target.detach().cpu())
            train_pred_batches.append(pred.detach().cpu())

        train_loss = loss_sum / max(n_graphs, 1)
        y_true = torch.cat(train_true_batches).numpy()
        y_pred = torch.cat(train_pred_batches).numpy()
        train_mae = float(mean_absolute_error(y_true, y_pred))
        train_r2 = float(r2_score(y_true, y_pred))

        # 有验证集时用验证 MAE 做调度和早停；否则退化为看训练集。
        if valid_loader is not None:
            valid_mae, valid_r2 = evaluate_loader(model, valid_loader, device)
        else:
            valid_mae, valid_r2 = train_mae, train_r2

        scheduler.step(valid_mae)

        mae_history.append(train_mae)
        r2_history.append(train_r2)
        train_loss_history.append(train_loss)
        valid_mae_history.append(valid_mae)
        valid_r2_history.append(valid_r2)

        print_report(
            f"Epoch {epoch + 1}/{n_epochs} | "
            f"Train MAE: {train_mae:.6f} | Train R2: {train_r2:.6f} | "
            f"Valid MAE: {valid_mae:.6f} | Valid R2: {valid_r2:.6f}"
        )

        if valid_mae < best_mae:
            best_mae = valid_mae
            best_model = copy.deepcopy(model.state_dict())
            epochs_without_improve = 0
            torch.save(best_model, best_model_path)
            print_report(
                f"Saved best model at epoch {epoch + 1} | "
                f"Best valid MAE: {best_mae:.6f} | "
                f"Path: {best_model_path}"
            )
        else:
            epochs_without_improve += 1
            if epochs_without_improve >= patience:
                print_report(f"Early stopping triggered after {epoch + 1} epochs")
                break

        # 每个 epoch 都保存可恢复的 checkpoint。
        should_save_checkpoint = (
            (epoch + 1) % checkpoint_interval == 0
            or (epoch + 1) == n_epochs
            or epochs_without_improve >= patience
        )
        if should_save_checkpoint:
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "best_model_state_dict": best_model if best_model is not None else model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "scaler_state_dict": scaler.state_dict(),
                    "best_mae": best_mae,
                    "mae_history": mae_history,
                    "r2_history": r2_history,
                    "train_loss_history": train_loss_history,
                    "valid_mae_history": valid_mae_history,
                    "valid_r2_history": valid_r2_history,
                },
                checkpoint_path,
            )
            print_report(f"Saved resume checkpoint at epoch {epoch + 1} | Path: {checkpoint_path}")

    if best_model is None:
        best_model = model.state_dict()
        torch.save(best_model, best_model_path)

    trajectory = pd.DataFrame(
        {
            "Train_loss": train_loss_history,
            "MAE_Train": mae_history,
            "R2_Train": r2_history,
            "MAE_Valid": valid_mae_history,
            "R2_Valid": valid_r2_history,
        }
    )
    trajectory.to_csv(model_dir / f"{artifact_prefix}_training.csv", index=False)

    summary = {
        "timestamp": pd.Timestamp.now().isoformat(),
        "model_name": model_name,
        "best_model_path": str(best_model_path),
        "checkpoint_path": str(checkpoint_path),
        "output_dir": str(model_dir),
        "cache_dir": str(cache_dir),
        "enable_cross_interaction": enable_cross_interaction,
        "num_intra_layers": num_intra_layers,
        "use_interaction_types": use_interaction_types,
        "use_moe": use_moe,
    }
    with open(model_dir / f"{artifact_prefix}_summary.json", "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, ensure_ascii=False)

    end = time.time()
    print_report("-" * 30)
    print_report(f"Best valid MAE: {best_mae}")
    print_report(f"Training time (min): {(end - start) / 60}")
    report.close()

    return best_model_path
def parse_args():
    """命令行参数定义。"""
    parser = argparse.ArgumentParser(description="PGSSI parametric training")
    parser.add_argument("--run-dir", type=str, default=None)
    parser.add_argument("--cache-dir", type=str, default=None)

    parser.add_argument(
        "--train-path",
        type=str,
        default=str(PROJECT_ROOT / "dataset" / "wu_et_al" / "molecule_train.csv"),
    )
    parser.add_argument(
        "--valid-path",
        type=str,
        default=str(PROJECT_ROOT / "dataset" / "wu_et_al" / "molecule_valid.csv"),
    )
    parser.add_argument(
        "--test-path",
        type=str,
        default=str(PROJECT_ROOT / "dataset" / "all" / "all_merged_test.csv"),
    )

    parser.add_argument("--model-name", type=str, default="PGSSI")
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--lr", type=float, default=2.5e-4)
    parser.add_argument("--n-epochs", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--train-num-workers", type=int, default=default_num_workers())
    parser.add_argument("--valid-num-workers", type=int, default=default_num_workers())
    parser.add_argument("--weight-decay", type=float, default=2e-5)
    parser.add_argument("--early-stopping-patience", type=int, default=20)
    parser.add_argument("--disable-cross-interaction", action="store_true")
    parser.add_argument("--num-intra-layers", type=int, default=2)
    parser.add_argument("--disable-interaction-types", action="store_true")
    parser.add_argument("--disable-moe", action="store_true")
    parser.add_argument("--disable-physics-prior", action="store_true")
    parser.add_argument("--disable-cross-refine", action="store_true")
    parser.add_argument("--topology-only", action="store_true")
    parser.add_argument("--direct-loggamma-head", action="store_true")
    parser.add_argument("--disable-formula-layer", action="store_true")
    # 启用后会从 runs/<run-dir>/<model-name>_resume.pth 恢复模型和优化器状态。
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--checkpoint-interval", type=int, default=10)
    parser.add_argument("--physics-loss-weight", type=float, default=0.0)
    parser.add_argument("--init-model-path", type=str, default=None)
    parser.add_argument("--quiet-progress", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    """???????????????????"""
    args = parse_args()
    set_seed(args.seed)
    configure_torch_runtime()

    enable_cross = not args.disable_cross_interaction
    use_interaction_types = not args.disable_interaction_types
    use_moe = not args.disable_moe
    use_physics_prior = not args.disable_physics_prior
    output_dir = make_run_dir(args.run_dir, prefix="pgssi_train")
    cache_dir = make_cache_dir(args.cache_dir, prefix="pgssi_shared")
    tqdm.write(f"Artifacts directory: {output_dir}")
    tqdm.write(f"Cache directory: {cache_dir}")

    train_path = resolve_input_path(args.train_path)
    valid_path = resolve_input_path(args.valid_path) if args.valid_path else None
    test_path = resolve_input_path(args.test_path)
    test_name = dataset_cache_prefix(test_path)
    artifact_prefix = f"{dataset_cache_prefix(train_path)}_{args.model_name}"

    train_df = pd.read_csv(train_path)
    valid_df = pd.read_csv(valid_path) if valid_path else None
    best_model_path = train_PGSSI(
        train_df=train_df,
        valid_df=valid_df,
        model_name=args.model_name,
        hyperparameters={
            "hidden_dim": args.hidden_dim,
            "lr": args.lr,
            "n_epochs": args.n_epochs,
            "batch_size": args.batch_size,
            "train_num_workers": args.train_num_workers,
            "valid_num_workers": args.valid_num_workers,
            "weight_decay": args.weight_decay,
            "early_stopping_patience": args.early_stopping_patience,
            "enable_cross_interaction": enable_cross,
            "num_intra_layers": args.num_intra_layers,
            "use_interaction_types": use_interaction_types,
            "use_moe": use_moe,
            "use_physics_prior": use_physics_prior,
            "disable_cross_refine": args.disable_cross_refine,
            "topology_only": args.topology_only,
            "direct_loggamma_head": args.direct_loggamma_head,
            "disable_formula_layer": args.disable_formula_layer,
            "train_cache_prefix": dataset_cache_prefix(train_path),
            "valid_cache_prefix": dataset_cache_prefix(valid_path) if valid_path else "valid",
            "artifact_prefix": artifact_prefix,
            "checkpoint_interval": args.checkpoint_interval,
            "physics_loss_weight": args.physics_loss_weight,
            "init_model_path": args.init_model_path,
            "quiet_progress": args.quiet_progress,
        },
        output_dir=output_dir,
        cache_dir=cache_dir,
        resume=args.resume,
    )

    test_results = {
        test_name: evaluate_test_file(
            model_path=best_model_path,
            test_path=str(test_path),
            test_name=test_name,
            model_name=args.model_name,
            artifact_prefix=artifact_prefix,
            hidden_dim=args.hidden_dim,
            enable_cross_interaction=enable_cross,
            num_intra_layers=args.num_intra_layers,
            use_interaction_types=use_interaction_types,
            use_moe=use_moe,
            use_physics_prior=use_physics_prior,
            disable_cross_refine=args.disable_cross_refine,
            topology_only=args.topology_only,
            direct_loggamma_head=args.direct_loggamma_head,
            disable_formula_layer=args.disable_formula_layer,
            output_dir=output_dir,
            cache_dir=cache_dir,
        ),
    }
    with open(Path(output_dir) / f"{artifact_prefix}_test_summary.json", "w", encoding="utf-8") as fh:
        json.dump(test_results, fh, indent=2, ensure_ascii=False)
    print_test_summary(test_results)


if __name__ == "__main__":
    main()
