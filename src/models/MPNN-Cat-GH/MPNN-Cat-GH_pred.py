"""
评估和预测脚本
包含完整评估和快速评估功能
"""

import torch
import yaml
import argparse
import os
import numpy as np
from tqdm import tqdm
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from models import SoluteSolventGraphModel  # noqa: E402
from utils import GraphDataLoader, calculate_metrics, plot_predictions  # noqa: E402


def evaluate(model, dataloader, device, save_dir=None):
    """评估模型"""
    model.eval()
    all_preds = []
    all_labels = []
    all_attention_weights = []
    
    with torch.no_grad():
        for solute_data, solvent_data, labels, temperatures in tqdm(dataloader, desc="评估中"):
            # 移动到设备
            solute_data = solute_data.to(device)
            solvent_data = solvent_data.to(device)
            labels = labels.to(device)
            temperatures = temperatures.to(device)
            
            # 前向传播
            predictions, attention_weights = model(solute_data, solvent_data, temperatures)
            predictions = predictions.squeeze()
            
            # 记录
            all_preds.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_attention_weights.append(attention_weights)
    
    # 转换为numpy数组
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    # 过滤掉 NaN 和 Inf 值
    valid_mask = np.isfinite(all_preds) & np.isfinite(all_labels)
    if valid_mask.sum() == 0:
        print("警告: 没有有效数据点")
        return {}, all_preds, all_labels, all_attention_weights
    
    all_preds = all_preds[valid_mask]
    all_labels = all_labels[valid_mask]
    
    # 计算指标
    metrics = calculate_metrics(all_labels, all_preds)
    
    # 计算绝对误差
    abs_errors = np.abs(all_preds - all_labels)
    mae = np.mean(abs_errors)
    
    # 计算不同阈值下的准确率
    threshold_01 = np.sum(abs_errors < 0.1) / len(abs_errors) * 100
    threshold_02 = np.sum(abs_errors < 0.2) / len(abs_errors) * 100
    threshold_03 = np.sum(abs_errors < 0.3) / len(abs_errors) * 100
    
    # 打印结果
    print("\n评估结果:")
    print("=" * 50)
    for metric_name, metric_value in metrics.items():
        print(f"{metric_name}: {metric_value:.4f}")
    print("-" * 50)
    print(f"平均绝对误差 (MAE): {mae:.4f}")
    print(f"绝对误差 < 0.1 的数据占比: {threshold_01:.2f}%")
    print(f"绝对误差 < 0.2 的数据占比: {threshold_02:.2f}%")
    print(f"绝对误差 < 0.3 的数据占比: {threshold_03:.2f}%")
    print("=" * 50)
    
    # 绘制预测结果
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        plot_path = os.path.join(save_dir, 'predictions.png')
        plot_predictions(all_labels, all_preds, save_path=plot_path)
        print(f"\n预测结果图已保存到: {plot_path}")
    
    return metrics, all_preds, all_labels, all_attention_weights


def evaluate_test(model_path='checkpoints/checkpoint_epoch_170.pth'):
    """快速评估测试集"""
    # 加载配置
    with open(os.path.join(_HERE, 'config.yaml'), 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    repo_root = os.path.abspath(os.path.join(_HERE, "..", "..", ".."))
    data_path = config["data"]["data_path"]
    if not os.path.isabs(data_path):
        config["data"]["data_path"] = os.path.abspath(os.path.join(repo_root, data_path))
    
    # 设置设备
    device = torch.device(config['training']['device'] 
                         if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 加载模型
    print(f"\n加载模型从 {model_path}...")
    checkpoint = torch.load(model_path, map_location=device)
    
    model = SoluteSolventGraphModel(
        input_dim=128,
        hidden_dim=config['model']['hidden_dim'],
        num_layers=config['model']['num_layers'],
        num_heads=config['model']['num_heads'],
        dropout=config['model']['dropout'],
        use_batch_norm=config['model']['use_batch_norm'],
        output_dim=1
    ).to(device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print(f"模型加载完成 (Epoch: {checkpoint.get('epoch', 'N/A')})")
    
    # 加载数据
    print("\n加载数据...")
    data_loader = GraphDataLoader(
        data_path=config['data']['data_path'],
        batch_size=config['training']['batch_size'],
        train_ratio=config['data']['train_ratio'],
        val_ratio=config['data']['val_ratio'],
        test_ratio=config['data']['test_ratio'],
        random_seed=config['data']['random_seed']
    )
    
    # 评估测试集
    print("\n评估测试集...")
    dataloader = data_loader.get_dataloader('test')
    
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for solute_data, solvent_data, labels, temperatures in tqdm(dataloader, desc="评估中"):
            solute_data = solute_data.to(device)
            solvent_data = solvent_data.to(device)
            labels = labels.to(device)
            temperatures = temperatures.to(device)
            
            predictions, _ = model(solute_data, solvent_data, temperatures)
            predictions = predictions.squeeze()
            
            all_preds.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # 转换为numpy数组
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    # 过滤掉 NaN 和 Inf 值
    valid_mask = np.isfinite(all_preds) & np.isfinite(all_labels)
    if valid_mask.sum() == 0:
        print("警告: 没有有效数据点")
        return
    
    all_preds = all_preds[valid_mask]
    all_labels = all_labels[valid_mask]
    
    # 计算指标
    metrics = calculate_metrics(all_labels, all_preds)
    
    # 计算绝对误差
    abs_errors = np.abs(all_preds - all_labels)
    mae = np.mean(abs_errors)
    
    # 计算不同阈值下的准确率
    threshold_01 = np.sum(abs_errors < 0.1) / len(abs_errors) * 100
    threshold_02 = np.sum(abs_errors < 0.2) / len(abs_errors) * 100
    threshold_03 = np.sum(abs_errors < 0.3) / len(abs_errors) * 100
    
    # 打印结果
    print("\n" + "=" * 60)
    print("测试集评估结果")
    print("=" * 60)
    for metric_name, metric_value in metrics.items():
        print(f"{metric_name}: {metric_value:.4f}")
    print("-" * 60)
    print(f"平均绝对误差 (MAE): {mae:.4f}")
    print(f"绝对误差 < 0.1 的数据占比: {threshold_01:.2f}%")
    print(f"绝对误差 < 0.2 的数据占比: {threshold_02:.2f}%")
    print(f"绝对误差 < 0.3 的数据占比: {threshold_03:.2f}%")
    print("=" * 60)
    
    # 保存结果
    os.makedirs('results', exist_ok=True)
    results_file = 'results/evaluation_test_epoch170.txt'
    with open(results_file, 'w', encoding='utf-8') as f:
        f.write(f"测试集评估结果 (使用Epoch {checkpoint.get('epoch', 'N/A')}的模型)\n")
        f.write("=" * 60 + "\n")
        for metric_name, metric_value in metrics.items():
            f.write(f"{metric_name}: {metric_value:.4f}\n")
        f.write("-" * 60 + "\n")
        f.write(f"平均绝对误差 (MAE): {mae:.4f}\n")
        f.write(f"绝对误差 < 0.1 的数据占比: {threshold_01:.2f}%\n")
        f.write(f"绝对误差 < 0.2 的数据占比: {threshold_02:.2f}%\n")
        f.write(f"绝对误差 < 0.3 的数据占比: {threshold_03:.2f}%\n")
        f.write("=" * 60 + "\n")
    
    print(f"\n评估结果已保存到: {results_file}")


def main():
    parser = argparse.ArgumentParser(description='评估溶质-溶剂交互图学习模型')
    parser.add_argument('--model_path', type=str, default='checkpoints/checkpoint_epoch_170.pth',
                       help='模型检查点路径（默认: checkpoints/checkpoint_epoch_170.pth）')
    parser.add_argument('--config', type=str, default='config.yaml',
                       help='配置文件路径')
    parser.add_argument('--split', type=str, default='test',
                       choices=['train', 'val', 'test'],
                       help='评估数据集')
    parser.add_argument('--save_dir', type=str, default='results',
                       help='结果保存目录')
    parser.add_argument('--quick', action='store_true',
                       help='快速评估模式（仅评估测试集）')
    args = parser.parse_args()
    
    if args.quick:
        # 快速评估模式
        evaluate_test(args.model_path)
    else:
        # 完整评估模式
        # 加载配置
        config_path = args.config
        if not os.path.isabs(config_path):
            local_cfg = os.path.join(_HERE, config_path)
            if os.path.exists(local_cfg):
                config_path = local_cfg
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)

        repo_root = os.path.abspath(os.path.join(_HERE, "..", "..", ".."))
        data_path = config["data"]["data_path"]
        if not os.path.isabs(data_path):
            config["data"]["data_path"] = os.path.abspath(os.path.join(repo_root, data_path))
        if not os.path.isabs(args.save_dir):
            args.save_dir = os.path.abspath(os.path.join(repo_root, args.save_dir))
        
        # 设置设备
        device = torch.device(config['training']['device'] 
                             if torch.cuda.is_available() else 'cpu')
        print(f"使用设备: {device}")
        
        # 加载模型
        print(f"加载模型从 {args.model_path}...")
        checkpoint = torch.load(args.model_path, map_location=device)
        
        # 创建模型
        model = SoluteSolventGraphModel(
            input_dim=128,  # 根据实际数据调整
            hidden_dim=config['model']['hidden_dim'],
            num_layers=config['model']['num_layers'],
            num_heads=config['model']['num_heads'],
            dropout=config['model']['dropout'],
            use_batch_norm=config['model']['use_batch_norm'],
            output_dim=1
        ).to(device)
        
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        print("模型加载完成")
        
        # 加载数据
        print("加载数据...")
        data_loader = GraphDataLoader(
            data_path=config['data']['data_path'],
            batch_size=config['training']['batch_size'],
            train_ratio=config['data']['train_ratio'],
            val_ratio=config['data']['val_ratio'],
            test_ratio=config['data']['test_ratio'],
            random_seed=config['data']['random_seed']
        )
        
        # 评估
        dataloader = data_loader.get_dataloader(args.split)
        metrics, predictions, labels, attention_weights = evaluate(
            model, dataloader, device, args.save_dir
        )
        
        # 保存结果
        os.makedirs(args.save_dir, exist_ok=True)
        results_file = os.path.join(args.save_dir, f'evaluation_{args.split}.txt')
        
        # 计算绝对误差统计（用于保存）
        all_preds_array = np.array(predictions)
        all_labels_array = np.array(labels)
        valid_mask = np.isfinite(all_preds_array) & np.isfinite(all_labels_array)
        if valid_mask.sum() > 0:
            all_preds_array = all_preds_array[valid_mask]
            all_labels_array = all_labels_array[valid_mask]
            abs_errors = np.abs(all_preds_array - all_labels_array)
            mae = np.mean(abs_errors)
            threshold_01 = np.sum(abs_errors < 0.1) / len(abs_errors) * 100
            threshold_02 = np.sum(abs_errors < 0.2) / len(abs_errors) * 100
            threshold_03 = np.sum(abs_errors < 0.3) / len(abs_errors) * 100
        else:
            mae = np.nan
            threshold_01 = threshold_02 = threshold_03 = 0.0
        
        with open(results_file, 'w', encoding='utf-8') as f:
            f.write(f"评估数据集: {args.split}\n")
            f.write("=" * 50 + "\n")
            for metric_name, metric_value in metrics.items():
                f.write(f"{metric_name}: {metric_value:.4f}\n")
            f.write("-" * 50 + "\n")
            f.write(f"平均绝对误差 (MAE): {mae:.4f}\n")
            f.write(f"绝对误差 < 0.1 的数据占比: {threshold_01:.2f}%\n")
            f.write(f"绝对误差 < 0.2 的数据占比: {threshold_02:.2f}%\n")
            f.write(f"绝对误差 < 0.3 的数据占比: {threshold_03:.2f}%\n")
            f.write("=" * 50 + "\n")
        
        print(f"\n评估结果已保存到: {results_file}")


if __name__ == '__main__':
    main()
