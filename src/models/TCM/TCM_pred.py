"""
测试脚本
用于在测试集上评估模型性能
计算MAE以及绝对误差<0.1, <0.2, <0.3的二元组合占比百分比
"""
import numpy as np
import pandas as pd
import pickle
from data_loader import DataLoader
from tensor_completion import calculate_metrics
import os
from scipy import stats


def load_model(model_path: str):
    """加载训练好的模型"""
    with open(model_path, 'rb') as f:
        data = pickle.load(f)
    return data['model'], data['data_loader']


def load_test_data(test_csv_path: str):
    """加载测试集CSV文件"""
    df = pd.read_csv(test_csv_path)
    # 重命名列以匹配数据加载器的期望格式
    df = df.rename(columns={
        'Solute_SMILES': 'solute',
        'Solvent_SMILES': 'solvent',
        'log-gamma': 'ln_gamma_inf',
        'T': 'temperature'
    })
    return df


def estimate_error_distribution_from_mae(mae: float, thresholds: list = [0.1, 0.2, 0.3]):
    """
    基于MAE值估算绝对误差分布
    
    假设误差服从正态分布，MAE ≈ 0.798 * σ (标准差)
    对于正态分布，MAE = sqrt(2/π) * σ ≈ 0.798 * σ
    
    参数:
        mae: 平均绝对误差
        thresholds: 误差阈值列表
        
    返回:
        每个阈值对应的百分比
    """
    # 从MAE估算标准差
    # MAE = sqrt(2/π) * σ，所以 σ = MAE / sqrt(2/π)
    sigma = mae / np.sqrt(2 / np.pi)
    
    # 对于绝对误差 |X|，其分布是折叠正态分布
    # P(|X| < t) = P(-t < X < t) = 2 * Φ(t/σ) - 1
    results = {}
    for threshold in thresholds:
        # 使用标准正态分布的累积分布函数
        prob = 2 * stats.norm.cdf(threshold / sigma) - 1
        percentage = prob * 100
        results[threshold] = percentage
    
    return results, sigma


def test_model_on_testset():
    """在测试集上测试模型"""
    print("="*60)
    print("在测试集上评估模型")
    print("="*60)
    
    # 1. 加载训练好的模型
    model_path = 'models/tcm_model.pkl'
    print(f"\n1. 加载模型: {model_path}")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"模型文件不存在: {model_path}")
    
    model, data_loader = load_model(model_path)
    print("   模型加载完成!")
    
    # 2. 加载测试集
    test_csv_path = r'D:\化工预测\论文复现结果\TCM\data\molecule_test.csv'
    print(f"\n2. 加载测试集: {test_csv_path}")
    test_df = load_test_data(test_csv_path)
    print(f"   测试集数据形状: {test_df.shape}")
    print(f"   测试集前5行数据:")
    print(test_df.head())
    
    # 3. 获取模型的张量和索引映射
    tensor, mask = data_loader.get_tensor()
    solute_to_idx = {solute: idx for idx, solute in enumerate(data_loader.solutes)}
    solvent_to_idx = {solvent: idx for idx, solvent in enumerate(data_loader.solvents)}
    temp_to_idx = {temp: idx for idx, temp in enumerate(data_loader.temperatures)}
    
    # 4. 对测试集进行预测
    print("\n3. 对测试集进行预测...")
    reconstructed = model.predict(tensor, mask)
    
    # 5. 收集测试集的真实值和预测值
    y_true_list = []
    y_pred_list = []
    system_errors = {}  # {(solute, solvent): [errors]}
    
    valid_count = 0
    invalid_count = 0
    
    for _, row in test_df.iterrows():
        solute = row['solute']
        solvent = row['solvent']
        temp = row['temperature']
        true_value = row['ln_gamma_inf']
        
        # 检查是否在模型的索引中
        if solute not in solute_to_idx or solvent not in solvent_to_idx:
            invalid_count += 1
            continue
        
        # 找到最接近的温度bin
        temp_idx = None
        min_diff = float('inf')
        for t, idx in temp_to_idx.items():
            diff = abs(t - temp)
            if diff < min_diff:
                min_diff = diff
                temp_idx = idx
        
        if temp_idx is None:
            invalid_count += 1
            continue
        
        i = solute_to_idx[solute]
        j = solvent_to_idx[solvent]
        k = temp_idx
        
        pred_value = reconstructed[i, j, k]
        
        # 检查预测值是否有效
        if np.isnan(pred_value):
            invalid_count += 1
            continue
        
        y_true_list.append(true_value)
        y_pred_list.append(pred_value)
        
        # 记录每个二元组合的误差
        system_key = (solute, solvent)
        if system_key not in system_errors:
            system_errors[system_key] = []
        system_errors[system_key].append(abs(true_value - pred_value))
        
        valid_count += 1
    
    print(f"   有效预测数量: {valid_count}")
    print(f"   无效预测数量: {invalid_count}")
    
    if valid_count == 0:
        print("   错误: 没有有效的预测数据!")
        return
    
    # 6. 计算整体MAE
    print("\n4. 计算评估指标...")
    y_true = np.array(y_true_list)
    y_pred = np.array(y_pred_list)
    
    mae = np.mean(np.abs(y_true - y_pred))
    print(f"   整体MAE: {mae:.6f}")
    
    # 7. 计算每个二元组合的平均绝对误差
    print("\n5. 计算二元组合的平均绝对误差...")
    system_mae = {}
    for system_key, errors in system_errors.items():
        system_mae[system_key] = np.mean(errors)
    
    print(f"   二元组合总数: {len(system_mae)}")
    
    # 8. 统计绝对误差<0.1, <0.2, <0.3的二元组合占比
    print("\n6. 统计绝对误差阈值占比...")
    thresholds = [0.1, 0.2, 0.3]
    mae_values = list(system_mae.values())
    
    for threshold in thresholds:
        count = sum(1 for mae_val in mae_values if mae_val < threshold)
        percentage = (count / len(mae_values)) * 100
        print(f"   绝对误差 < {threshold}: {count}/{len(mae_values)} = {percentage:.2f}%")
    
    # 9. 输出详细结果
    print("\n" + "="*60)
    print("评估结果总结")
    print("="*60)
    print(f"测试集总样本数: {len(test_df)}")
    print(f"有效预测数: {valid_count}")
    print(f"无效预测数: {invalid_count}")
    print(f"\n整体MAE: {mae:.6f}")
    print(f"\n二元组合统计:")
    print(f"  总二元组合数: {len(system_mae)}")
    for threshold in thresholds:
        count = sum(1 for mae_val in mae_values if mae_val < threshold)
        percentage = (count / len(mae_values)) * 100
        print(f"  绝对误差 < {threshold}: {count} ({percentage:.2f}%)")
    
    print("\n" + "="*60)
    print("评估完成!")
    print("="*60)
    
    return {
        'mae': mae,
        'valid_count': valid_count,
        'invalid_count': invalid_count,
        'system_count': len(system_mae),
        'thresholds': {f'<{t}': sum(1 for mae_val in mae_values if mae_val < t) / len(mae_values) * 100 
                      for t in thresholds}
    }


def estimate_from_reference_data(mae: float = 0.085736):
    """
    基于参考数据表格进行插值估算绝对误差分布
    
    参考数据:
    - GHGEAT: MAE=0.07, AE≤0.1=81.92%, AE≤0.2=92.39%, AE≤0.3=95.73%
    - GNNCat: MAE=0.08, AE≤0.1=81.07%, AE≤0.2=92.06%, AE≤0.3=94.88%
    - SolvGNNCat: MAE=0.09, AE≤0.1=78.96%, AE≤0.2=91.03%, AE≤0.3=95.02%
    - SolvGNNGH_wo: MAE=0.10, AE≤0.1=74.87%, AE≤0.2=89.10%, AE≤0.3=93.71%
    - GHGEAT_wo: MAE=0.09, AE≤0.1=74.07%, AE≤0.2=89.85%, AE≤0.3=93.42%
    - MPNN-cat-GH: MAE=0.11, AE≤0.1=76.43%, AE≤0.2=91.22%, AE≤0.3=95.21%
    - NCF: MAE=0.16, AE≤0.1=60.05%, AE≤0.2=81.64%, AE≤0.3=89.53%
    
    参数:
        mae: TCM模型的平均绝对误差值
    """
    # 参考数据：按MAE排序，取平均值（对于相同MAE的多个模型）
    reference_data = [
        {'mae': 0.07, 'ae_0.1': 81.92, 'ae_0.2': 92.39, 'ae_0.3': 95.73},  # GHGEAT
        {'mae': 0.08, 'ae_0.1': 81.07, 'ae_0.2': 92.06, 'ae_0.3': 94.88},  # GNNCat
        {'mae': 0.09, 'ae_0.1': 76.52, 'ae_0.2': 90.44, 'ae_0.3': 94.22},  # SolvGNNCat和GHGEAT_wo的平均
        {'mae': 0.10, 'ae_0.1': 74.87, 'ae_0.2': 89.10, 'ae_0.3': 93.71},  # SolvGNNGH_wo
        {'mae': 0.11, 'ae_0.1': 76.43, 'ae_0.2': 91.22, 'ae_0.3': 95.21},  # MPNN-cat-GH
        {'mae': 0.16, 'ae_0.1': 60.05, 'ae_0.2': 81.64, 'ae_0.3': 89.53},  # NCF
    ]
    
    # 线性插值
    def interpolate(mae_val, threshold_key):
        # 找到MAE值所在区间
        for i in range(len(reference_data) - 1):
            mae_low = reference_data[i]['mae']
            mae_high = reference_data[i + 1]['mae']
            
            if mae_low <= mae_val <= mae_high:
                # 线性插值
                ratio = (mae_val - mae_low) / (mae_high - mae_low)
                value_low = reference_data[i][threshold_key]
                value_high = reference_data[i + 1][threshold_key]
                interpolated = value_low + ratio * (value_high - value_low)
                return interpolated
        
        # 如果超出范围，使用最近的值
        if mae_val < reference_data[0]['mae']:
            return reference_data[0][threshold_key]
        else:
            return reference_data[-1][threshold_key]
    
    results = {
        0.1: interpolate(mae, 'ae_0.1'),
        0.2: interpolate(mae, 'ae_0.2'),
        0.3: interpolate(mae, 'ae_0.3')
    }
    
    print("="*60)
    print("基于参考数据插值估算TCM模型的绝对误差分布")
    print("="*60)
    print(f"\nTCM模型MAE: {mae:.6f}")
    print(f"\n估算的绝对误差分布（基于参考数据插值）:")
    print(f"  绝对误差 < 0.1: {results[0.1]:.2f}%")
    print(f"  绝对误差 < 0.2: {results[0.2]:.2f}%")
    print(f"  绝对误差 < 0.3: {results[0.3]:.2f}%")
    
    print("\n参考数据对比:")
    print("  GHGEAT (MAE=0.07): AE≤0.1=81.92%, AE≤0.2=92.39%, AE≤0.3=95.73%")
    print("  GNNCat (MAE=0.08): AE≤0.1=81.07%, AE≤0.2=92.06%, AE≤0.3=94.88%")
    print("  SolvGNNCat (MAE=0.09): AE≤0.1=78.96%, AE≤0.2=91.03%, AE≤0.3=95.02%")
    print("="*60)
    
    return results


def estimate_from_mae_only(mae: float = 0.085736):
    """
    仅基于MAE值估算绝对误差分布（用于快速估算）
    
    参数:
        mae: 平均绝对误差值
    """
    print("="*60)
    print("基于MAE值估算绝对误差分布")
    print("="*60)
    print(f"\n输入MAE: {mae:.6f}")
    
    thresholds = [0.1, 0.2, 0.3]
    results, sigma = estimate_error_distribution_from_mae(mae, thresholds)
    
    print(f"\n估算的标准差 (σ): {sigma:.6f}")
    print(f"\n估算的绝对误差分布:")
    for threshold in thresholds:
        percentage = results[threshold]
        print(f"  绝对误差 < {threshold}: {percentage:.2f}%")
    
    print("\n注意: 这是基于正态分布假设的理论估算，实际结果可能有所不同")
    print("="*60)
    
    return results


if __name__ == "__main__":
    import sys
    
    # 如果提供了--estimate参数，仅进行估算
    if len(sys.argv) > 1 and sys.argv[1] == '--estimate':
        # 使用图片中显示的MAE值
        estimate_from_mae_only(mae=0.085736)
    elif len(sys.argv) > 1 and sys.argv[1] == '--reference':
        # 基于参考数据插值估算
        estimate_from_reference_data(mae=0.085736)
    else:
        # 运行完整的测试脚本获取真实数据
        test_model_on_testset()

