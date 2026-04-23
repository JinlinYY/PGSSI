"""
训练脚本
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import yaml
import os
import argparse
import json
import glob
import re
from tqdm import tqdm
import numpy as np
from datetime import datetime
import time
import sys
from typing import Optional, Tuple

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from models import SoluteSolventGraphModel  # noqa: E402
from utils import _PairDataset, _read_csv_rows, calculate_metrics  # noqa: E402
from torch_geometric.loader import DataLoader as PyGDataLoader  # noqa: E402


def _latest_epoch_checkpoint(save_dir: str) -> Optional[str]:
    """Return path to checkpoint_epoch_*.pth with largest epoch number, if any."""
    pattern = os.path.join(save_dir, "checkpoint_epoch_*.pth")
    files = glob.glob(pattern)
    if not files:
        return None

    def epoch_num(path: str) -> int:
        base = os.path.basename(path)
        m = re.match(r"checkpoint_epoch_(\d+)\.pth$", base)
        return int(m.group(1)) if m else -1

    return max(files, key=epoch_num)


def resolve_resume_checkpoint(resume_arg: str, fresh: bool, save_dir: str) -> Tuple[Optional[str], str]:
    """
    Returns (checkpoint_path_or_None, human_readable_note).
    resume_arg: 'auto' | 'none' | explicit file path
    """
    if fresh:
        return None, "fresh (--fresh): train from scratch"
    s = (resume_arg or "").strip()
    low = s.lower()
    if low in ("none", "", "false", "0"):
        return None, "explicit no resume"
    if low != "auto":
        p = os.path.abspath(s)
        if os.path.isfile(p):
            return p, p
        raise FileNotFoundError(f"--resume path not found: {p}")

    # Prefer best_model.pth under outputs root (new default), then fall back to save_dir for backward compatibility.
    outputs_root = os.path.abspath(os.path.join(save_dir, ".."))
    best_outputs = os.path.join(outputs_root, "best_model.pth")
    if os.path.isfile(best_outputs):
        return best_outputs, best_outputs
    best_legacy = os.path.join(save_dir, "best_model.pth")
    if os.path.isfile(best_legacy):
        return best_legacy, best_legacy
    latest = _latest_epoch_checkpoint(save_dir)
    if latest:
        return latest, latest
    return None, "auto: no checkpoint in save_dir, training from scratch"


def write_training_stop_info(
    log_dir: str,
    payload: dict,
) -> Tuple[str, str]:
    """Write timestamped JSON plus training_stop_latest.json for easy discovery."""
    os.makedirs(log_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path_ts = os.path.join(log_dir, f"training_stop_{ts}.json")
    path_latest = os.path.join(log_dir, "training_stop_latest.json")
    with open(path_ts, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    with open(path_latest, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return path_ts, path_latest


def train_epoch(model, dataloader, criterion, optimizer, device):
    """训练一个epoch"""
    model.train()
    total_loss = 0.0
    all_preds = []
    all_labels = []
    num_batches = 0
    skipped_batches = 0
    total_samples = 0
    skipped_samples = 0
    
    for solute_data, solvent_data, labels, temperatures in tqdm(dataloader, desc="训练中"):
        # 移动到设备
        solute_data = solute_data.to(device)
        solvent_data = solvent_data.to(device)
        labels = labels.to(device)
        temperatures = temperatures.to(device)
        
        batch_size = labels.size(0)
        total_samples += batch_size
        
        # 前向传播
        optimizer.zero_grad()
        predictions, _ = model(solute_data, solvent_data, temperatures)
        predictions = predictions.view(-1)
        labels = labels.view(-1)
        
        # 检查预测值中是否有 NaN 或 Inf
        if torch.isnan(predictions).any() or torch.isinf(predictions).any():
            skipped_batches += 1
            skipped_samples += batch_size
            continue
        
        # 计算损失
        loss = criterion(predictions, labels)
        
        # 检查损失是否为 NaN 或 Inf
        if torch.isnan(loss) or torch.isinf(loss):
            skipped_batches += 1
            skipped_samples += batch_size
            continue
        
        # 反向传播
        loss.backward()
        
        # 检查梯度是否有NaN/Inf（优化：只在检测到时才处理）
        # 先快速检查是否有NaN/Inf，避免遍历所有参数
        has_nan_grad = False
        # 只检查前几个参数，如果发现NaN/Inf再处理所有参数
        param_list = list(model.parameters())
        if len(param_list) > 0 and param_list[0].grad is not None:
            # 快速检查：只检查第一个参数的梯度
            first_grad = param_list[0].grad
            if torch.isnan(first_grad).any() or torch.isinf(first_grad).any():
                has_nan_grad = True
                # 如果发现NaN/Inf，处理所有参数
                for param in model.parameters():
                    if param.grad is not None:
                        param.grad = torch.nan_to_num(param.grad, nan=0.0, posinf=1e6, neginf=-1e6)
        
        if has_nan_grad:
            print(f"警告: 检测到 NaN 或 Inf 梯度，已替换为有限值")
        
        # 增强梯度裁剪（更严格的裁剪）
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
        optimizer.step()
        
        # 记录（只有成功处理的批次才记录）
        total_loss += loss.item()
        pred_np = predictions.detach().cpu().numpy().reshape(-1)
        label_np = labels.detach().cpu().numpy().reshape(-1)
        
        # 过滤掉 NaN 和 Inf 值
        valid_mask = np.isfinite(pred_np) & np.isfinite(label_np)
        if valid_mask.sum() > 0:
            all_preds.extend(pred_np[valid_mask])
            all_labels.extend(label_np[valid_mask])
        
        num_batches += 1
    
    # 计算平均损失（只基于成功处理的批次）
    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    metrics = calculate_metrics(all_labels, all_preds)
    
    # 添加统计信息
    if skipped_batches > 0:
        skip_ratio = skipped_batches / (num_batches + skipped_batches) * 100
        sample_skip_ratio = skipped_samples / total_samples * 100 if total_samples > 0 else 0
        print(f"  警告: 跳过了 {skipped_batches} 个批次 ({skip_ratio:.1f}%), {skipped_samples} 个样本 ({sample_skip_ratio:.1f}%)")
    
    return avg_loss, metrics


def validate_epoch(model, dataloader, criterion, device):
    """验证一个epoch"""
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []
    num_batches = 0
    skipped_batches = 0
    total_samples = 0
    skipped_samples = 0
    
    with torch.no_grad():
        for solute_data, solvent_data, labels, temperatures in tqdm(dataloader, desc="验证中"):
            # 移动到设备
            solute_data = solute_data.to(device)
            solvent_data = solvent_data.to(device)
            labels = labels.to(device)
            temperatures = temperatures.to(device)
            
            batch_size = labels.size(0)
            total_samples += batch_size
            
            # 前向传播
            predictions, _ = model(solute_data, solvent_data, temperatures)
            predictions = predictions.view(-1)
            labels = labels.view(-1)
            
            # 检查预测值中是否有 NaN 或 Inf
            if torch.isnan(predictions).any() or torch.isinf(predictions).any():
                skipped_batches += 1
                skipped_samples += batch_size
                continue
            
            # 计算损失
            loss = criterion(predictions, labels)
            
            # 检查损失是否为 NaN 或 Inf
            if torch.isnan(loss) or torch.isinf(loss):
                skipped_batches += 1
                skipped_samples += batch_size
                continue
            
            # 记录（只有成功处理的批次才记录）
            total_loss += loss.item()
            pred_np = predictions.detach().cpu().numpy().reshape(-1)
            label_np = labels.detach().cpu().numpy().reshape(-1)
            
            # 过滤掉 NaN 和 Inf 值
            valid_mask = np.isfinite(pred_np) & np.isfinite(label_np)
            if valid_mask.sum() > 0:
                all_preds.extend(pred_np[valid_mask])
                all_labels.extend(label_np[valid_mask])
            
            num_batches += 1
    
    # 计算平均损失（只基于成功处理的批次）
    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    metrics = calculate_metrics(all_labels, all_preds)
    
    # 添加统计信息
    if skipped_batches > 0:
        skip_ratio = skipped_batches / (num_batches + skipped_batches) * 100
        sample_skip_ratio = skipped_samples / total_samples * 100 if total_samples > 0 else 0
        print(f"  警告: 跳过了 {skipped_batches} 个批次 ({skip_ratio:.1f}%), {skipped_samples} 个样本 ({sample_skip_ratio:.1f}%)")
    
    return avg_loss, metrics


def _repo_root() -> str:
    return os.path.abspath(os.path.join(_HERE, "..", "..", ".."))


def _make_pyg_loader(rows, batch_size: int, shuffle: bool) -> PyGDataLoader:
    return PyGDataLoader(
        _PairDataset(rows),
        batch_size=int(batch_size),
        shuffle=bool(shuffle),
        num_workers=0,
    )


@torch.no_grad()
def evaluate_only(model, dataloader, device):
    model.eval()
    all_preds = []
    all_labels = []

    for solute_data, solvent_data, labels, temperatures in tqdm(dataloader, desc="测试中"):
        solute_data = solute_data.to(device)
        solvent_data = solvent_data.to(device)
        labels = labels.to(device).view(-1)
        temperatures = temperatures.to(device)

        predictions, _ = model(solute_data, solvent_data, temperatures)
        predictions = predictions.view(-1)

        pred_np = predictions.detach().cpu().numpy().reshape(-1)
        label_np = labels.detach().cpu().numpy().reshape(-1)
        valid_mask = np.isfinite(pred_np) & np.isfinite(label_np)
        if valid_mask.sum() > 0:
            all_preds.extend(pred_np[valid_mask].tolist())
            all_labels.extend(label_np[valid_mask].tolist())

    metrics = calculate_metrics(all_labels, all_preds)
    return {
        "num_samples": int(len(all_labels)),
        "MAE": float(metrics.get("MAE", float("nan"))),
        "R2": float(metrics.get("R2", float("nan"))),
        "RMSE": float(metrics.get("RMSE", float("nan"))),
    }


def main():
    parser = argparse.ArgumentParser(description='训练溶质-溶剂交互图学习模型')
    parser.add_argument('--config', type=str, default='config.yaml',
                       help='配置文件路径')
    parser.add_argument(
        '--resume',
        type=str,
        default='auto',
        help="默认 auto：从 logging.save_dir 下 best_model.pth 或最新 checkpoint_epoch_*.pth 恢复；"
             "也可传入具体 .pth 路径；传 none 表示从头训练",
    )
    parser.add_argument(
        '--fresh',
        action='store_true',
        help='强制从头训练（忽略本地 checkpoint，等价于 --resume none）',
    )
    args = parser.parse_args()
    
    # 加载配置
    config_path = args.config
    if not os.path.isabs(config_path):
        # Prefer config next to this script if user passes a relative path.
        local_cfg = os.path.join(_HERE, config_path)
        if os.path.exists(local_cfg):
            config_path = local_cfg
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    # 将相对路径解析为绝对路径：
    # - 训练数据固定为 dataset/all/all_merged_train.csv
    # - 验证/测试在同目录 dataset/all 下，分别为 all_merged_valid.csv/all_merged_test.csv
    # - 输出路径：相对仓库根目录
    repo_root = _repo_root()
    for k in ["save_dir", "log_dir"]:
        p = config["logging"][k]
        if not os.path.isabs(p):
            config["logging"][k] = os.path.abspath(os.path.join(repo_root, p))
    
    # 设置设备
    device = torch.device(config['training']['device'] 
                         if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 创建保存目录
    os.makedirs(config['logging']['save_dir'], exist_ok=True)
    os.makedirs(config['logging']['log_dir'], exist_ok=True)

    # 训练过程“最佳模型”同时在 outputs 根目录保留一份（便于直接使用）
    outputs_root_dir = os.path.join(repo_root, "src", "models", "MPNN-Cat-GH", "outputs")
    os.makedirs(outputs_root_dir, exist_ok=True)
    outputs_best_model_path = os.path.join(outputs_root_dir, "best_model.pth")
    
    # 加载数据（固定 all 数据集三份 CSV）
    print("加载数据...")
    all_dir = os.path.join(repo_root, "dataset", "all")
    train_csv = os.path.join(all_dir, "all_merged_train.csv")
    val_csv = os.path.join(all_dir, "all_merged_valid.csv")
    test_csv = os.path.join(all_dir, "all_merged_test.csv")
    for p in (train_csv, val_csv, test_csv):
        if not os.path.isfile(p):
            raise FileNotFoundError(f"未找到数据文件: {p}")

    train_rows = _read_csv_rows(train_csv)
    val_rows = _read_csv_rows(val_csv)
    test_rows = _read_csv_rows(test_csv)

    batch_size = int(config["training"]["batch_size"])
    train_dataloader = _make_pyg_loader(train_rows, batch_size=batch_size, shuffle=True)
    val_dataloader = _make_pyg_loader(val_rows, batch_size=batch_size, shuffle=False)
    test_dataloader = _make_pyg_loader(test_rows, batch_size=batch_size, shuffle=False)
    
    # 创建模型
    print("创建模型...")
    model = SoluteSolventGraphModel(
        input_dim=128,  # 根据实际数据调整
        hidden_dim=config['model']['hidden_dim'],
        num_layers=config['model']['num_layers'],
        num_heads=config['model']['num_heads'],
        dropout=config['model']['dropout'],
        use_batch_norm=config['model']['use_batch_norm'],
        output_dim=1
    ).to(device)
    
    print(f"模型参数数量: {sum(p.numel() for p in model.parameters()):,}")
    
    # 损失函数和优化器
    criterion = nn.MSELoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=float(config['training']['learning_rate']),
        weight_decay=float(config['training']['weight_decay'])
    )
    scheduler = ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )
    
    resume_ckpt, resume_note = resolve_resume_checkpoint(
        args.resume, args.fresh, config['logging']['save_dir']
    )
    print(f"恢复策略: {resume_note}")

    # 恢复训练
    start_epoch = 0
    best_val_loss = float('inf')

    if resume_ckpt:
        print(f"从检查点恢复训练: {resume_ckpt}")
        checkpoint = torch.load(resume_ckpt, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = int(checkpoint['epoch'])
        best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        if 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        print(f"从第 {start_epoch} 个 epoch 之后继续（下一轮为 epoch {start_epoch + 1}），历史最佳验证损失: {best_val_loss:.4f}")
    
    # 训练循环
    print("开始训练...")
    log_file = os.path.join(
        config['logging']['log_dir'],
        f"train_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    )
    
    with open(log_file, 'w', encoding='utf-8') as f:
        f.write(
            "Epoch\tTrain Loss\tTrain MAE\tTrain R2\t"
            "Val Loss\tVal MAE\tVal R2\t"
            "Test Loss\tTest MAE\tTest R2\n"
        )
    
    # 记录训练开始时间
    training_start_time = time.time()
    target_mae = float(
        config['training'].get('target_mae', config['training'].get('target_train_mae', 0.11))
    )
    stopped_by: Optional[str] = None
    best_train_mae = float("inf")
    best_test_mae = float("inf")

    report_path = os.path.join(
        repo_root, "src", "models", "MPNN-Cat-GH", "outputs", "Report_training_MPNN-Cat-GH.txt"
    )
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as rf:
        rf.write(
            f"MPNN-Cat-GH training report\n"
            f"start_time: {datetime.now().isoformat(timespec='seconds')}\n"
            f"train_csv: {train_csv}\n"
            f"val_csv: {val_csv}\n"
            f"test_csv: {test_csv}\n"
            f"save_dir: {config['logging']['save_dir']}\n"
            f"log_dir: {config['logging']['log_dir']}\n"
            f"resume: {resume_note}\n"
            f"device: {device}\n"
            f"batch_size: {batch_size}\n"
            f"num_epochs: {config['training']['num_epochs']}\n"
            f"checkpoint_every_epochs: 30\n"
            f"----------------------------------------\n"
            f"epoch\ttrain_loss\ttrain_mae\ttrain_r2\tbest_train_mae\tval_loss\tval_mae\tval_r2\t"
            f"test_mae\ttest_r2\tbest_test_mae\tcheckpoint\n"
        )

    for epoch in range(start_epoch, config['training']['num_epochs']):
        current_epoch = epoch + 1
        print(f"\n{'='*60}")
        print(f"第 {current_epoch}/{config['training']['num_epochs']} 轮训练")
        print(f"{'='*60}")
        
        # 训练
        train_loss, train_metrics = train_epoch(
            model, train_dataloader, criterion, optimizer, device
        )
        
        # 验证
        val_loss, val_metrics = validate_epoch(
            model, val_dataloader, criterion, device
        )
        scheduler.step(val_loss)

        train_mae = float(train_metrics.get("MAE", float("nan")))
        train_r2 = float(train_metrics.get("R2", float("nan")))
        if np.isfinite(train_mae) and train_mae < best_train_mae:
            best_train_mae = train_mae

        # 每个 epoch：打印训练 MAE/R2/BestMAE（按你的要求）
        print(f"[Epoch {current_epoch}] TrainMAE={train_mae:.6f}, TrainR2={train_r2:.6f}, BestMAE={best_train_mae:.6f}")

        # 默认不每个 epoch 跑测试；只在 checkpoint 间隔跑一次
        test_metrics = {"MAE": float("nan"), "R2": float("nan")}
        checkpoint_note = ""
        if current_epoch % 30 == 0:
            # 保存检查点
            checkpoint_path = os.path.join(
                config['logging']['save_dir'], f'checkpoint_epoch_{current_epoch}.pth'
            )
            torch.save({
                'epoch': current_epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_val_loss': best_val_loss,
                'patience_counter': 0,
                'config': config
            }, checkpoint_path)
            checkpoint_note = os.path.basename(checkpoint_path)
            print(f"保存检查点到 {checkpoint_path}")

            # 同时进行一次测试，并打印 TestMAE/TestR2/Best Test Mae
            _test_loss, test_metrics = validate_epoch(
                model, test_dataloader, criterion, device
            )
            test_mae = float(test_metrics.get("MAE", float("nan")))
            test_r2 = float(test_metrics.get("R2", float("nan")))
            if np.isfinite(test_mae) and test_mae < best_test_mae:
                best_test_mae = test_mae
            print(f"[Epoch {current_epoch}] TestMAE={test_mae:.6f}, TestR2={test_r2:.6f}, Best Test Mae={best_test_mae:.6f}")
        
        # 记录日志
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(
                f"{current_epoch}\t{train_loss:.4f}\t{train_metrics['MAE']:.4f}\t{train_metrics['R2']:.4f}\t"
                f"{val_loss:.4f}\t{val_metrics['MAE']:.4f}\t{val_metrics['R2']:.4f}\t"
                f"nan\t{test_metrics.get('MAE', float('nan')):.4f}\t{test_metrics.get('R2', float('nan')):.4f}\n"
            )

        with open(report_path, "a", encoding="utf-8") as rf:
            rf.write(
                f"{current_epoch}\t{train_loss:.6f}\t{train_mae:.6f}\t{train_r2:.6f}\t{best_train_mae:.6f}\t"
                f"{val_loss:.6f}\t{float(val_metrics.get('MAE', float('nan'))):.6f}\t{float(val_metrics.get('R2', float('nan'))):.6f}\t"
                f"{float(test_metrics.get('MAE', float('nan'))):.6f}\t{float(test_metrics.get('R2', float('nan'))):.6f}\t{best_test_mae:.6f}\t"
                f"{checkpoint_note}\n"
            )
        
        # 保存最佳模型（按验证损失）
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            payload = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_val_loss': best_val_loss,
                'patience_counter': 0,
                'config': config
            }
            torch.save(payload, outputs_best_model_path)
            print(f"保存最佳模型到 {outputs_best_model_path}")

        # 早停：训练 MAE <= 阈值 且验证 MAE <= 阈值
        val_mae = float(val_metrics.get('MAE', float('nan')))
        mae_ok_train = np.isfinite(train_mae) and train_mae <= target_mae
        mae_ok_val = np.isfinite(val_mae) and val_mae <= target_mae
        if mae_ok_train and mae_ok_val:
            stopped_by = 'target_mae_train_and_val'
            best_model_path = os.path.join(config['logging']['save_dir'], 'best_model.pth')
            elapsed = time.time() - training_start_time
            stop_payload = {
                'stop_reason': stopped_by,
                'message': (
                    f'早停条件满足: train_mae={train_mae:.6f}<={target_mae}，且 '
                    f'val_mae={val_mae:.6f}<={target_mae}'
                ),
                'epoch_completed': epoch + 1,
                'target_mae': target_mae,
                'train_loss': float(train_loss),
                'train_mae': train_mae,
                'train_r2': float(train_metrics.get('R2', float('nan'))),
                'val_loss': float(val_loss),
                'val_mae': val_mae,
                'val_r2': float(val_metrics.get('R2', float('nan'))),
                'val_mae_meets_threshold': mae_ok_val,
                'best_val_loss_so_far': float(best_val_loss),
                'resume_checkpoint': resume_ckpt,
                'resume_note': resume_note,
                'log_file': log_file,
                'best_model_path': best_model_path if os.path.isfile(best_model_path) else None,
                'elapsed_seconds': elapsed,
                'timestamp': datetime.now().isoformat(timespec='seconds'),
            }
            path_ts, path_latest = write_training_stop_info(
                config['logging']['log_dir'], stop_payload
            )
            print(f"\n早停触发: {stop_payload['message']}")
            print(f"已写入停止信息: {path_ts}")
            print(f"最新摘要副本: {path_latest}")
            with open(log_file, 'a', encoding='utf-8') as f:
                f.write(
                    f"# EARLY_STOP\t{stop_payload['stop_reason']}\t"
                    f"train_mae={train_mae:.6f}\tval_mae={val_mae:.6f}\t"
                    f"threshold={target_mae}\tjson={path_ts}\n"
                )
            break
    
    # 记录训练结束时间并计算总时间
    training_end_time = time.time()
    total_training_time = training_end_time - training_start_time
    
    # 格式化时间输出
    hours = int(total_training_time // 3600)
    minutes = int((total_training_time % 3600) // 60)
    seconds = int(total_training_time % 60)
    
    print("\n" + "="*60)
    print("训练完成！")
    print("="*60)
    if stopped_by == 'target_mae_train_and_val':
        print(
            "结束原因: 训练 MAE 与验证 MAE 均达到阈值早停"
            "（详见 outputs/logs 下 training_stop_*.json）"
        )
    print(f"最佳验证损失: {best_val_loss:.4f}")
    print(f"\n总训练时间: {hours}小时 {minutes}分钟 {seconds}秒")
    print(f"总训练时间: {total_training_time:.2f} 秒")
    print("="*60)
    print(f"训练报告已写入: {report_path}")

    # 训练完成后：在 all_merged_test.csv 上自动评测，并落盘为 outputs/MPNN-Cat-GH_eval.json
    try:
        eval_out_path = os.path.join(
            repo_root, "src", "models", "MPNN-Cat-GH", "outputs", "MPNN-Cat-GH_eval.json"
        )
        os.makedirs(os.path.dirname(eval_out_path), exist_ok=True)

        best_path = outputs_best_model_path
        model_used = "in_memory_model"
        if os.path.isfile(best_path):
            ckpt = torch.load(best_path, map_location=device)
            if "model_state_dict" in ckpt:
                model.load_state_dict(ckpt["model_state_dict"])
                model_used = os.path.abspath(best_path)

        test_result = evaluate_only(model, test_dataloader, device)
        payload = {
            "model": "MPNN-Cat-GH",
            "model_path": model_used,
            "device": str(device),
            "test_csv": os.path.abspath(test_csv),
            "batch_size": int(batch_size),
            "metrics": test_result,
            "best_val_loss": float(best_val_loss),
            "timestamp": datetime.now().isoformat(timespec="seconds"),
        }
        with open(eval_out_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        print(f"测试结果已写入: {eval_out_path}")
    except Exception as e:
        print(f"警告: 训练后自动测试/写入 JSON 失败: {e}")


if __name__ == '__main__':
    main()

