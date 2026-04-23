"""
训练脚本
用于训练张量补全模型
"""
import numpy as np
import pandas as pd
from data_loader import DataLoader, load_example_data
from tensor_completion import TensorCompletionModel, calculate_metrics
from config import MODEL_CONFIG, DATA_CONFIG
import os
import pickle
from sklearn.model_selection import KFold
import json
import re


def _find_latest_checkpoint(checkpoint_dir: str) -> str:
    """
    在 checkpoint_dir 下寻找最新的 .ckpt 文件。
    优先解析形如 tcm_epoch_{epoch}.ckpt 的 epoch 最大者；若无法解析则按修改时间最新。
    找不到则返回 None。
    """
    if not checkpoint_dir or not os.path.isdir(checkpoint_dir):
        return None

    try:
        entries = os.listdir(checkpoint_dir)
    except Exception:
        return None

    ckpt_paths = []
    for name in entries:
        if not name.lower().endswith(".ckpt"):
            continue
        full = os.path.join(checkpoint_dir, name)
        if os.path.isfile(full):
            ckpt_paths.append(full)

    if not ckpt_paths:
        return None

    epoch_re = re.compile(r"tcm_epoch_(\d+)\.ckpt$", re.IGNORECASE)
    best_epoch = None
    best_path = None
    for p in ckpt_paths:
        m = epoch_re.search(os.path.basename(p))
        if not m:
            continue
        try:
            ep = int(m.group(1))
        except Exception:
            continue
        if best_epoch is None or ep > best_epoch:
            best_epoch = ep
            best_path = p

    if best_path is not None:
        return best_path

    # fallback: pick most recently modified
    try:
        ckpt_paths.sort(key=lambda x: os.path.getmtime(x), reverse=True)
    except Exception:
        ckpt_paths.sort()
    return ckpt_paths[0]


def _standardize_df_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    兼容不同数据集字段命名，统一为:
    solute, solvent, temperature, ln_gamma_inf
    """
    out = df.copy()
    out = out.rename(
        columns={
            "Solute_SMILES": "solute",
            "Solvent_SMILES": "solvent",
            "T": "temperature",
            "log-gamma": "ln_gamma_inf",
            # 兼容可能出现的其它命名
            "solute_smiles": "solute",
            "solvent_smiles": "solvent",
            "temp": "temperature",
            "temperature_bin": "temperature",
        }
    )
    return out


def _bin_temperature(values: np.ndarray, bin_width: float = 1.0) -> np.ndarray:
    """与 DataLoader.create_temperature_bins 保持一致的温度 bin 规则。"""
    binned = np.round(values.astype(float) / float(bin_width)) * float(bin_width)
    return np.round(binned, 6)


def _build_subset_mask(
    loader: DataLoader,
    df_subset: pd.DataFrame,
    bin_width: float = 1.0,
) -> np.ndarray:
    """
    将一个子集 DataFrame 映射为与 loader 张量同 shape 的 mask。
    mask 仅在该子集出现过的 (solute, solvent, temperature_bin) 位置为 True。
    """
    tensor, _ = loader.get_tensor()
    m1, m2, m3 = tensor.shape
    mask = np.zeros((m1, m2, m3), dtype=bool)

    if df_subset is None or len(df_subset) == 0:
        return mask

    df_subset = _standardize_df_columns(df_subset)
    df_subset = df_subset.dropna(subset=["solute", "solvent", "temperature", "ln_gamma_inf"]).copy()
    df_subset["solute"] = df_subset["solute"].astype(str)
    df_subset["solvent"] = df_subset["solvent"].astype(str)
    df_subset["temperature"] = df_subset["temperature"].astype(float)
    df_subset["ln_gamma_inf"] = df_subset["ln_gamma_inf"].astype(float)

    df_subset["temperature_bin"] = _bin_temperature(df_subset["temperature"].to_numpy(), bin_width=bin_width)

    solute_to_idx = {s: i for i, s in enumerate(loader.solutes)}
    solvent_to_idx = {s: i for i, s in enumerate(loader.solvents)}
    temp_to_idx = {float(t): i for i, t in enumerate(loader.temperatures)}

    grouped = (
        df_subset.groupby(["solute", "solvent", "temperature_bin"], as_index=False)["ln_gamma_inf"]
        .mean()
    )

    skipped = 0
    for _, row in grouped.iterrows():
        s = str(row["solute"])
        v = str(row["solvent"])
        t = float(row["temperature_bin"])
        if s not in solute_to_idx or v not in solvent_to_idx or t not in temp_to_idx:
            skipped += 1
            continue
        i = solute_to_idx[s]
        j = solvent_to_idx[v]
        k = temp_to_idx[t]
        mask[i, j, k] = True

    if skipped > 0:
        print(f"[Warn] subset 中有 {skipped} 个格点无法映射到主张量索引（已跳过）")

    return mask


def cross_validation(data_loader: DataLoader, n_folds: int = 10, 
                     ranks: tuple = (5, 5, 2), max_iter: int = 100):
    """
    执行10折交叉验证
    
    参数:
        data_loader: 数据加载器
        n_folds: 折数
        ranks: Tucker分解的秩
        max_iter: 最大迭代次数
        
    返回:
        所有折的评估结果
    """
    tensor, mask = data_loader.get_tensor()
    all_results = []
    
    print(f"\n开始 {n_folds} 折交叉验证...")
    print(f"数据张量形状: {tensor.shape}")
    print(f"总数据点数: {np.sum(mask)}")
    
    for fold in range(n_folds):
        print(f"\n{'='*60}")
        print(f"折 {fold + 1}/{n_folds}")
        print(f"{'='*60}")
        
        # 创建系统级分割
        train_mask, test_mask = data_loader.create_system_wise_split(
            n_folds=n_folds, fold=fold, random_state=DATA_CONFIG['random_state']
        )
        
        train_size = np.sum(train_mask)
        test_size = np.sum(test_mask)
        print(f"训练集大小: {train_size}")
        print(f"测试集大小: {test_size}")
        
        # 创建训练张量（缺失值用NaN）
        train_tensor = tensor.copy()
        train_tensor[~train_mask] = np.nan
        
        # 训练模型
        print("\n训练模型...")
        import time
        start_time = time.time()
        model = TensorCompletionModel(ranks=ranks)
        # 使用MAE为监控指标，迭代次数按要求为 max_iter
        from config import MODEL_CONFIG
        patience = MODEL_CONFIG.get('early_stopping_patience', 10)
        model.fit(train_tensor, train_mask, max_iter=max_iter, verbose=True, early_stopping_patience=patience)
        total_time = time.time() - start_time
        print(f"\n本折训练总用时: {total_time:.2f} 秒")
        
        # 预测（使用完整重构张量，在测试掩码位置评估）
        print("\n进行预测...")
        reconstructed = model.predict(tensor, test_mask)
        
        # 评估：仅在测试掩码位置取预测
        y_true = tensor[test_mask]
        y_pred = reconstructed[test_mask]
        
        metrics = calculate_metrics(y_true, y_pred)
        
        print(f"\n折 {fold + 1} 评估结果:")
        for key, value in metrics.items():
            print(f"  {key}: {value:.6f}")
        
        all_results.append({
            'fold': fold + 1,
            'metrics': metrics,
            'train_size': train_size,
            'test_size': test_size
        })
    
    # 计算平均指标
    print(f"\n{'='*60}")
    print("交叉验证总结")
    print(f"{'='*60}")
    
    avg_metrics = {}
    for key in all_results[0]['metrics'].keys():
        avg_metrics[key] = np.mean([r['metrics'][key] for r in all_results])
        std_metrics = np.std([r['metrics'][key] for r in all_results])
        print(f"{key}: {avg_metrics[key]:.6f} ± {std_metrics:.6f}")
    
    return all_results, avg_metrics


def train_full_model(data_loader: DataLoader, ranks: tuple = (5, 5, 2), 
                     max_iter: int = 100, save_path: str = None,
                     use_full_data: bool = False,
                     train_ratio: float = 0.7, val_ratio: float = 0.15, test_ratio: float = 0.15,
                     resume_from: str = None,
                     checkpoint_dir: str = None,
                     checkpoint_every: int = None,
                     fixed_masks: tuple = None):
    """
    使用全部数据训练模型
    
    参数:
        data_loader: 数据加载器
        ranks: Tucker分解的秩
        max_iter: 最大迭代次数
        save_path: 模型保存路径
        
    返回:
        训练好的模型
    """
    tensor, mask = data_loader.get_tensor()
    report_lines = []
    best_model_state = None
    best_metric = float("inf")
    best_epoch = 0
    best_metrics_snapshot = None

    if use_full_data:
        print(f"\n使用全部数据训练模型...")
        print(f"数据张量形状: {tensor.shape}")
        print(f"数据点数量: {int(np.sum(mask))}")
        train_mask = mask
        val_mask = None
        test_mask = None
    else:
        # 保留兼容：若仍按比例划分则走原逻辑
        if fixed_masks is not None:
            train_mask, val_mask, test_mask = fixed_masks
            print(f"\n使用固定 mask 训练/验证/测试（来自 CSV 文件切分）...")
        else:
            print(f"\n按比例划分 训练/验证/测试 (系统级)...")
            print(f"比例: train={train_ratio:.2f}, val={val_ratio:.2f}, test={test_ratio:.2f}")
            train_mask, val_mask, test_mask = data_loader.create_system_wise_split_with_validation(
                train_ratio=train_ratio, val_ratio=val_ratio, test_ratio=test_ratio
            )
        print(f"训练集点数: {int(np.sum(train_mask))}")
        print(f"验证集点数: {int(np.sum(val_mask)) if val_mask is not None else 0}")
        print(f"测试集点数: {int(np.sum(test_mask)) if test_mask is not None else 0}")
    
    # 训练张量（未知点置 NaN，仅在 train_mask 的位置贡献损失）
    train_tensor = tensor.copy()
    if not use_full_data:
        # 在训练时屏蔽掉非训练集的观测（防止泄漏）
        train_tensor[~train_mask] = np.nan
    
    # 训练模型（基于训练集早停）
    import time
    start_time = time.time()
    model = TensorCompletionModel(ranks=ranks)
    from config import MODEL_CONFIG
    patience = MODEL_CONFIG.get('early_stopping_patience', 10)
    conv_window = MODEL_CONFIG.get('convergence_window', 5)
    conv_rel_tol = MODEL_CONFIG.get('convergence_rel_tol', 1e-4)
    min_epochs = MODEL_CONFIG.get('min_epochs', 10)
    reg_lambda = MODEL_CONFIG.get('ridge_lambda', 1e-3)
    resume_state = None

    # 断点目录固定为 TCM/outputs/checkpoint（若未显式指定）
    if checkpoint_dir is None:
        checkpoint_dir = os.path.join("src", "models", "TCM", "outputs", "checkpoint")
    if checkpoint_every is None:
        checkpoint_every = 200  # 默认每200轮保存一次

    # 若用户未指定 resume_from，则在 checkpoint_dir 下自动寻找最新检查点
    if not resume_from:
        auto_ckpt = _find_latest_checkpoint(checkpoint_dir)
        if auto_ckpt:
            resume_from = auto_ckpt
            print(f"\n检测到检查点，自动从断点恢复: {resume_from}")

    if resume_from and os.path.exists(resume_from):
        print(f"\n从断点加载: {resume_from}")
        try:
            ckpt = TensorCompletionModel.load_checkpoint(resume_from)
            resume_state = ckpt.get('model_state', None)
        except Exception as e:
            print(f"断点加载失败: {e}")
    
    if checkpoint_every > 0:
        print(f"\n断点保存设置: 目录={checkpoint_dir}, 每{checkpoint_every}轮保存一次")
    
    # 这里不再直接调用 model.fit（其回调仅用于存断点），而是直接调用 tucker.decompose
    # 以便每个 epoch 输出训练 MAE/R2，并按要求每 200 epoch 做一次测试。
    model.is_fitted = False
    start_iter = 0
    if resume_state is not None:
        model.set_state(resume_state)
        start_iter = int(resume_state.get("epoch", 0))
        print(f"\n从断点恢复: epoch={start_iter}")

    try:
        model.tucker.reg_lambda = float(reg_lambda) if reg_lambda is not None else 0.0
    except Exception:
        model.tucker.reg_lambda = 0.0

    os.makedirs(checkpoint_dir, exist_ok=True)

    outputs_dir = os.path.dirname(save_path) if save_path else os.path.join("src", "models", "TCM", "outputs")
    os.makedirs(outputs_dir if outputs_dir else ".", exist_ok=True)
    best_model_path = os.path.join(outputs_dir, "tcm_best_model.pkl")
    report_path = os.path.join(outputs_dir, "Report_traing_TCM.txt")

    def _fmt_metrics(prefix: str, metrics: dict) -> str:
        if not metrics:
            return f"{prefix}: (empty)"
        mae = metrics.get("MAE", float("nan"))
        r2 = metrics.get("R2", float("nan"))
        rmse = metrics.get("RMSE", float("nan"))
        return f"{prefix}: MAE={mae:.6f} RMSE={rmse:.6f} R2={r2:.6f}"

    def on_epoch_end(epoch: int, mae: float, best_mae: float, no_improve: int):
        nonlocal best_model_state, best_metric, best_epoch, best_metrics_snapshot

        model._last_epoch = int(epoch)
        reconstructed = model.tucker.reconstruct()

        train_metrics = calculate_metrics(tensor[train_mask], reconstructed[train_mask])
        val_metrics = calculate_metrics(tensor[val_mask], reconstructed[val_mask]) if val_mask is not None else None

        # 每个 epoch 输出训练（以及验证）指标
        msg_parts = [f"Epoch {epoch}"]
        msg_parts.append(_fmt_metrics("train", train_metrics))
        if val_metrics is not None:
            msg_parts.append(_fmt_metrics("valid", val_metrics))
        msg = " | ".join(msg_parts)
        print(msg)
        report_lines.append(msg)

        # 以验证集 MAE 作为“最佳”标准；若无验证集则用训练 MAE
        monitor_mae = None
        if val_metrics is not None and "MAE" in val_metrics:
            monitor_mae = float(val_metrics["MAE"])
        elif "MAE" in train_metrics:
            monitor_mae = float(train_metrics["MAE"])

        if monitor_mae is not None and monitor_mae + 1e-12 < best_metric:
            best_metric = monitor_mae
            best_epoch = int(epoch)
            best_model_state = model.get_state()
            best_metrics_snapshot = {
                "train": train_metrics,
                "valid": val_metrics,
            }
            with open(best_model_path, "wb") as f:
                pickle.dump(
                    {
                        "model": model,
                        "data_loader": data_loader,
                        "ranks": ranks,
                        "best_epoch": best_epoch,
                        "best_metric_mae": best_metric,
                        "metrics": best_metrics_snapshot,
                    },
                    f,
                )
            print(f"[Best] epoch={best_epoch}, mae={best_metric:.6f}, 已保存最佳模型到 {best_model_path}")

        # 每隔 checkpoint_every 保存检查点，并做一次测试
        if checkpoint_every and checkpoint_every > 0 and epoch % checkpoint_every == 0:
            ckpt_path = os.path.join(checkpoint_dir, f"tcm_epoch_{epoch}.ckpt")
            model.save_checkpoint(
                ckpt_path,
                epoch=int(epoch),
                train_metrics=train_metrics,
                valid_metrics=val_metrics,
            )
            print(f"[Checkpoint] 已保存到 {ckpt_path}")

            if test_mask is not None:
                test_metrics = calculate_metrics(tensor[test_mask], reconstructed[test_mask])
                test_msg = _fmt_metrics("TEST", test_metrics)
                print(test_msg)
                report_lines.append(f"Epoch {epoch} | {test_msg}")

    # 开始训练
    model.tucker.decompose(
        train_tensor,
        train_mask,
        max_iter=max_iter,
        tol=1e-6,
        verbose=True,  # 保留内部 MAE+耗时输出
        early_stopping_patience=patience,
        convergence_window=conv_window,
        convergence_rel_tol=conv_rel_tol,
        min_epochs=min_epochs,
        start_iter=start_iter,
        on_epoch_end=on_epoch_end,
    )
    model.is_fitted = True
    total_time = time.time() - start_time
    print(f"\n训练总用时: {total_time:.2f} 秒")
    report_lines.append(f"Total training time: {total_time:.2f} s")
    report_lines.append(f"Best epoch: {best_epoch}, best_mae(monitor): {best_metric:.6f}")

    # 最终评估与 tcm_best_model.pkl 一致：加载按监控 MAE 保存的最佳权重（有验证集时为验证 MAE）
    if not use_full_data and best_model_state is not None:
        model.set_state(best_model_state)
        monitor_name = "验证集" if val_mask is not None else "训练集"
        msg = (
            f"\n最终评估已加载{monitor_name} MAE 最优模型 "
            f"(epoch={best_epoch}, 监控 MAE={best_metric:.6f})，与 outputs/tcm_best_model.pkl 对齐。"
        )
        print(msg)
        report_lines.append(msg.strip())

    # 若存在验证/测试划分，训练后分别评估
    if not use_full_data:
        reconstructed = model.tucker.reconstruct()
        # 使用重构值在对应掩码位置做评估，避免从原始张量拷贝造成泄漏
        val_metrics = None
        if val_mask is not None:
            val_metrics = calculate_metrics(tensor[val_mask], reconstructed[val_mask])
            print("\n验证集评估指标（最佳模型）:")
            for k, v in val_metrics.items():
                print(f"  {k}: {v:.6f}")
            report_lines.append(_fmt_metrics("Final VAL (best)", val_metrics))
        test_metrics = None
        if test_mask is not None:
            test_metrics = calculate_metrics(tensor[test_mask], reconstructed[test_mask])
            print("\n测试集评估指标（最佳模型）:")
            for k, v in test_metrics.items():
                print(f"  {k}: {v:.6f}")
            report_lines.append(_fmt_metrics("Final TEST (best)", test_metrics))

        if val_metrics is not None or test_metrics is not None:
            eval_path = os.path.join(outputs_dir, "TCM_eval.json")
            eval_payload = {
                "ranks": list(ranks),
                "best_epoch": best_epoch,
                "best_metric_mae": best_metric,
            }
            if val_metrics is not None:
                eval_payload["val_metrics"] = val_metrics
            if test_metrics is not None:
                eval_payload["test_metrics"] = test_metrics
            with open(eval_path, "w", encoding="utf-8") as f:
                json.dump(eval_payload, f, ensure_ascii=False, indent=2)
            print(f"\n评估结果已保存到: {eval_path}")
    
    # 保存模型
    if save_path:
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
        with open(save_path, 'wb') as f:
            pickle.dump({
                'model': model,
                'data_loader': data_loader,
                'ranks': ranks
            }, f)
        print(f"\n模型已保存到: {save_path}")

    # 保存训练报告
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines) + "\n")
    print(f"\n训练报告已保存到: {report_path}")
    
    return model


def rank_selection(data_loader: DataLoader, rank_candidates: list, 
                   n_folds: int = 5, max_iter: int = 50):
    """
    选择最优的秩组合
    
    参数:
        data_loader: 数据加载器
        rank_candidates: 候选秩组合列表，例如 [(3,3,2), (5,5,2), (7,7,3)]
        n_folds: 用于秩选择的折数（可以使用较少的折数以加快速度）
        max_iter: 最大迭代次数
        
    返回:
        最优秩组合和对应的指标
    """
    print(f"\n开始秩选择...")
    print(f"候选秩组合: {rank_candidates}")
    
    best_ranks = None
    best_wmse = float('inf')
    results = []
    
    for ranks in rank_candidates:
        print(f"\n{'='*60}")
        print(f"测试秩组合: {ranks}")
        print(f"{'='*60}")
        
        try:
            all_results, avg_metrics = cross_validation(
                data_loader, n_folds=n_folds, ranks=ranks, max_iter=max_iter
            )
            
            wmse = avg_metrics['wMSE']
            results.append({
                'ranks': ranks,
                'metrics': avg_metrics
            })
            
            print(f"\n秩组合 {ranks} 的 wMSE: {wmse:.6f}")
            
            if wmse < best_wmse:
                best_wmse = wmse
                best_ranks = ranks
                print(f"✓ 新的最佳秩组合!")
        
        except Exception as e:
            print(f"秩组合 {ranks} 训练失败: {e}")
            continue
    
    print(f"\n{'='*60}")
    print("秩选择总结")
    print(f"{'='*60}")
    print(f"最佳秩组合: {best_ranks}")
    print(f"最佳 wMSE: {best_wmse:.6f}")
    
    return best_ranks, results


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='训练张量补全模型')
    parser.add_argument(
        '--data_path',
        type=str,
        default=os.path.join('dataset', 'all', 'all_merged_train.csv'),
                       help='数据文件路径（CSV格式）')
    parser.add_argument(
        '--valid_path',
        type=str,
        default=None,
        help='验证集 CSV 路径（默认与 data_path 同目录下 all_merged_valid.csv）',
    )
    parser.add_argument(
        '--test_path',
        type=str,
        default=None,
        help='测试集 CSV 路径（默认与 data_path 同目录下 all_merged_test.csv）',
    )
    parser.add_argument('--mode', type=str, default='train',
                       choices=['train', 'cv', 'rank_selection'],
                       help='运行模式: train=训练完整模型, cv=交叉验证, rank_selection=秩选择')
    parser.add_argument('--ranks', type=int, nargs=3, default=[5, 5, 2],
                       help='Tucker分解的秩 (r1 r2 r3)')
    parser.add_argument('--max_iter', type=int, default=0,
                       help='最大迭代次数（设为 0 表示不限定，直到早停触发）')
    parser.add_argument('--use_full_data', action='store_true',
                       help='是否使用全部数据进行训练（不划分验证/测试）')
    parser.add_argument('--train_ratio', type=float, default=0.7,
                       help='系统级训练集比例（与 val_ratio, test_ratio 之和为 1）')
    parser.add_argument('--val_ratio', type=float, default=0.15,
                       help='系统级验证集比例')
    parser.add_argument('--test_ratio', type=float, default=0.15,
                       help='系统级测试集比例')
    parser.add_argument('--n_folds', type=int, default=10,
                       help='交叉验证折数')
    parser.add_argument(
        '--save_path',
        type=str,
        default=os.path.join('src', 'models', 'TCM', 'outputs', 'tcm_model.pkl'),
                       help='模型保存路径')
    parser.add_argument(
        '--checkpoint_dir',
        type=str,
        default=os.path.join('src', 'models', 'TCM', 'outputs', 'checkpoint'),
                       help='断点保存目录（默认: TCM/outputs/checkpoint）')
    parser.add_argument('--checkpoint_every', type=int, default=200,
                       help='每多少个epoch保存一次断点并测试（默认: 200，设为0表示不保存）')
    parser.add_argument('--resume_from', type=str, default=None,
                       help='从指定断点文件恢复训练（.ckpt）')
    
    args = parser.parse_args()

    # 每次运行训练前，先在 checkpoint_dir 下检查是否存在检查点；若存在且未指定 --resume_from，则自动续训
    if args.mode == "train":
        args.checkpoint_dir = args.checkpoint_dir or os.path.join("src", "models", "TCM", "outputs", "checkpoint")
        auto_ckpt = _find_latest_checkpoint(args.checkpoint_dir)
        if auto_ckpt and not args.resume_from:
            args.resume_from = auto_ckpt
            print(f"\n[AutoResume] 检测到检查点: {auto_ckpt}")
            print("[AutoResume] 未指定 --resume_from，已自动从该检查点开始训练。")
    
    # 加载数据
    print("加载数据...")
    if args.data_path and os.path.exists(args.data_path):
        # 从文件加载数据（按文件切分 train/valid/test）
        train_path = args.data_path
        base_dir = os.path.dirname(train_path)
        valid_path = args.valid_path or os.path.join(base_dir, "all_merged_valid.csv")
        test_path = args.test_path or os.path.join(base_dir, "all_merged_test.csv")

        df_train = _standardize_df_columns(pd.read_csv(train_path))
        df_valid = _standardize_df_columns(pd.read_csv(valid_path)) if os.path.exists(valid_path) else None
        df_test = _standardize_df_columns(pd.read_csv(test_path)) if os.path.exists(test_path) else None

        # 用 train/valid/test 的并集构建“主张量”，再用三个文件分别生成 mask
        dfs = [df_train]
        if df_valid is not None:
            dfs.append(df_valid)
        if df_test is not None:
            dfs.append(df_test)
        df_all = pd.concat(dfs, ignore_index=True)

        print(f"训练集: {train_path}")
        print(f"验证集: {valid_path if df_valid is not None else '(not found)'}")
        print(f"测试集: {test_path if df_test is not None else '(not found)'}")
    else:
        # 使用示例数据
        print("使用示例数据...")
        df_all = load_example_data()
        df_train = df_all.copy()
        df_valid = None
        df_test = None
    
    # 创建数据加载器
    loader = DataLoader()
    loader.load_from_dataframe(df_all)
    loader.create_temperature_bins(bin_width=1.0)  # 根据论文，使用1K bins

    # 如果提供了文件切分，则覆盖训练/验证/测试 mask（否则沿用 train_full_model 内部按比例划分）
    file_split_masks = None
    if args.mode == "train" and (df_valid is not None or df_test is not None):
        train_mask = _build_subset_mask(loader, df_train, bin_width=1.0)
        val_mask = _build_subset_mask(loader, df_valid, bin_width=1.0) if df_valid is not None else None
        test_mask = _build_subset_mask(loader, df_test, bin_width=1.0) if df_test is not None else None

        # 用同样 shape 的空 mask 占位，避免 None 分支复杂化
        if val_mask is None:
            val_mask = None
        if test_mask is None:
            test_mask = None

        file_split_masks = (train_mask, val_mask, test_mask)
        print(f"按文件划分 mask: train={int(np.sum(train_mask))}, "
              f"valid={(int(np.sum(val_mask)) if val_mask is not None else 0)}, "
              f"test={(int(np.sum(test_mask)) if test_mask is not None else 0)}")
    
    # 根据模式执行相应操作
    if args.mode == 'train':
        # 训练完整模型
        # 若提供 train/valid/test 文件切分，则用 file_split_masks 覆盖比例划分
        model = train_full_model(
            loader,
            ranks=tuple(args.ranks),
            max_iter=args.max_iter,
            save_path=args.save_path,
            use_full_data=args.use_full_data,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio,
            resume_from=args.resume_from,
            checkpoint_dir=args.checkpoint_dir,
            checkpoint_every=args.checkpoint_every,
            fixed_masks=file_split_masks,
        )
    
    elif args.mode == 'cv':
        # 交叉验证
        all_results, avg_metrics = cross_validation(
            loader,
            n_folds=args.n_folds,
            ranks=tuple(args.ranks),
            max_iter=args.max_iter
        )
        
        # 保存结果
        results_path = os.path.join('src', 'models', 'TCM', 'outputs', 'cv_results.json')
        os.makedirs(os.path.dirname(results_path), exist_ok=True)
        with open(results_path, 'w') as f:
            json.dump({
                'all_results': all_results,
                'avg_metrics': avg_metrics,
                'ranks': args.ranks
            }, f, indent=2)
        print(f"\n交叉验证结果已保存到: {results_path}")
    
    elif args.mode == 'rank_selection':
        # 秩选择
        rank_candidates = [
            (3, 3, 2), (4, 4, 2), (5, 5, 2), (6, 6, 2),
            (5, 5, 3), (7, 7, 2), (7, 7, 3)
        ]
        best_ranks, results = rank_selection(
            loader,
            rank_candidates=rank_candidates,
            n_folds=5,  # 使用较少的折数以加快速度
            max_iter=50
        )
        
        # 保存结果
        results_path = os.path.join('src', 'models', 'TCM', 'outputs', 'rank_selection_results.json')
        os.makedirs(os.path.dirname(results_path), exist_ok=True)
        with open(results_path, 'w') as f:
            json.dump({
                'best_ranks': best_ranks,
                'all_results': results
            }, f, indent=2)
        print(f"\n秩选择结果已保存到: {results_path}")


if __name__ == "__main__":
    main()


