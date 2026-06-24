import os
import pandas as pd
import numpy as np
from phasepy.actmodels import nrtl
from scipy.optimize import least_squares
import warnings

# 忽略数值计算中可能产生的警告
warnings.filterwarnings('ignore')

# ================= 配置区 =================
# 待批量评估的 4 个数据集
DATASETS = [
    "PGSSI_Lazzaroni_2023.csv",
    "PGSSI_Lazzaroni_Original.csv",
    "PGSSI_OrginW.csv",
    "PGSSI_WinOrg.csv"
]

# PhasePy NRTL 核心常量
ALPHA_MATRIX = np.array([[0.0, 0.3], [0.3, 0.0]])
X_INF = np.array([1e-6, 1.0 - 1e-6])

# 双参数拟合边界 (b12, b21)
PARAM_BOUNDS = ([-8000.0, -8000.0], [8000.0, 8000.0])
INITIAL_GUESS = [0.0, 0.0]
# ==========================================

def get_nrtl_ln_gamma(T, b12, b21):
    """
    核心预测函数：强制设定 a12=0, a21=0。仅通过焓参量(b)描述温度效应。
    """
    g_matrix = np.array([[0.0, float(b12)], 
                         [float(b21), 0.0]])
    g1_matrix = np.array([[0.0, 0.0], 
                          [0.0, 0.0]])
    
    ln_gamma_array = nrtl(X_INF, float(T), ALPHA_MATRIX, g_matrix, g1_matrix)
    return ln_gamma_array[0]

def residuals_2param(params, T_array, ln_gamma_exp_array):
    """
    双参数最小二乘法残差函数
    """
    b12, b21 = params
    calc_vals = []
    for T in T_array:
        try:
            calc_vals.append(get_nrtl_ln_gamma(T, b12, b21))
        except:
            calc_vals.append(1e6) # 计算失败时给极大惩罚
    return np.array(calc_vals) - ln_gamma_exp_array

def process_single_dataset(input_file):
    base_name = input_file.replace('.csv', '')
    output_pred_file = f"{base_name}_Predictions_Extrap_Clean.csv"
    output_params_file = f"{base_name}_nrtl_params_2param.csv"

    print(f"\n>>>> 正在处理数据集: {input_file} ...")
    if not os.path.exists(input_file):
        print(f"[错误] 未找到输入文件 {input_file}，跳过此数据集。")
        return

    df = pd.read_csv(input_file)
    
    # 清洗字符串格式
    df['Solute_SMILES'] = df['Solute_SMILES'].astype(str).str.strip()
    df['Solvent_SMILES'] = df['Solvent_SMILES'].astype(str).str.strip()

    # 创建外推预测列
    df['ln_gamma_pred_Extrap'] = np.nan 
    final_params_list = []

    grouped = df.groupby(['Solute_SMILES', 'Solvent_SMILES'])
    
    for (solute, solvent), group in grouped:
        indices = group.index.values
        T_data = group['T_K'].values
        ln_gamma_data = group['ln_gamma_inf'].values
        
        # 拦截：数据点少于 3 个无法执行外推验证（至少2个点拟合，1个点测试）
        if len(T_data) < 3:
            continue 
            
        # ============ 严格温度外推策略 ============
        # 1. 按温度升序排列
        sort_idx = np.argsort(T_data)
        T_sorted = T_data[sort_idx]
        ln_gamma_sorted = ln_gamma_data[sort_idx]
        idx_sorted = indices[sort_idx]

        # 2. 仅选取最冷的 2 个点作为训练/拟合样本
        train_n = 2 
        T_train = T_sorted[:train_n]
        ln_gamma_train = ln_gamma_sorted[:train_n]

        T_test = T_sorted[train_n:]
        idx_test = idx_sorted[train_n:]

        # 3. 拟合双参数 (仅依靠低温区)
        res_extrap = least_squares(residuals_2param, INITIAL_GUESS, 
                                   bounds=PARAM_BOUNDS, 
                                   args=(T_train, ln_gamma_train))
        
        # 4. 外推预测所有较高温度的数据点
        for i, T_val in enumerate(T_test):
            try:
                pred_val = get_nrtl_ln_gamma(T_val, res_extrap.x[0], res_extrap.x[1])
                df.loc[idx_test[i], 'ln_gamma_pred_Extrap'] = pred_val
            except:
                pass
        # ==========================================

        # 全量数据拟合供以后使用的参数（保持原样，导出各物系的最佳参数表）
        res_final = least_squares(residuals_2param, INITIAL_GUESS, 
                                  bounds=PARAM_BOUNDS, 
                                  args=(T_data, ln_gamma_data))
        
        final_params_list.append({
            'Solute_SMILES': solute,
            'Solvent_SMILES': solvent,
            'a12': 0.0,
            'b12': res_final.x[0],
            'a21': 0.0,
            'b21': res_final.x[1],
            'data_points_used': len(T_data)
        })

    # 清理未获得外推预测值的行（即过滤掉低温拟合点以及数据点不足的体系）
    df_clean = df.dropna(subset=['ln_gamma_pred_Extrap']).copy()

    if len(df_clean) > 0:
        absolute_errors = np.abs(df_clean['ln_gamma_pred_Extrap'] - df_clean['ln_gamma_inf'])
        mae = absolute_errors.mean()
        rmse = np.sqrt((absolute_errors**2).mean())
        
        gamma_inf_exp = np.exp(df_clean['ln_gamma_inf'])
        gamma_inf_pred = np.exp(df_clean['ln_gamma_pred_Extrap'])
        mape = np.mean(np.abs((gamma_inf_exp - gamma_inf_pred) / gamma_inf_exp)) * 100
        
        print("-" * 45)
        print(f" 数据集评估报告: {input_file}")
        print(f" 高温外推测试行数: {len(df_clean)}")
        print(f" ▶ ln(γ∞) 均方根误差 (RMSE): {rmse:.4f}")
        print(f" ▶ ln(γ∞) 平均绝对误差 (MAE): {mae:.4f}")
        print(f" ▶ γ∞ 平均绝对百分比误差 (MAPE): {mape:.2f}%")
        print("-" * 45)
    else:
        print(f"[提示] 数据集 {input_file} 清洗后无有效外推数据。")

    # 自动导出对应的文件
    df_clean.to_csv(output_pred_file, index=False)
    pd.DataFrame(final_params_list).to_csv(output_params_file, index=False)
    print(f" 预测集已保存: {output_pred_file}")
    print(f" 参数表已保存: {output_params_file}\n")


def main():
    print("=====================================================")
    print("        NRTL 模型多数据集自动化批量外推评估程序       ")
    print("=====================================================")
    
    for dataset in DATASETS:
        process_single_dataset(dataset)
        
    print("=====================================================")
    print("✅ 所有数据集的高温外推计算全部结束！")
    print("=====================================================")

if __name__ == "__main__":
    main()