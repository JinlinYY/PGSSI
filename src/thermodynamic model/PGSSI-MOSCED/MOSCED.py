import math
import openpyxl
import pandas as pd
import numpy as np

# ==========================================
# 1. 从原包中复制出的核心 MOSCED 计算算法
# ==========================================
def calculate_mosced(v1, v2, lambda1, lambda2, tau1, tau2, rho1, rho2, alpha1, alpha2, beta1, beta2, T):
    """
    原作者的计算单点温度 T 的算法
    注意：在原代码的算法中，rho (ρ) 就是参数表里的极性参数 q。
    """
    R = 8.3144598
    
    def powerT1(n, T):
        return n * pow((293 / T), 0.8)

    def powerT2(n, T):
        return n * pow((293 / T), 0.4)

    try:
        POL = pow(rho1, 4) * (1.15 - 1.15 * math.exp(-0.002337 * pow(powerT2(tau1, T), 3))) + 1
        xi1 = 0.68 * (POL - 1) + pow(3.4 - (2.4 * math.exp(-0.002687 * pow(alpha1 * beta1, 1.5))), pow((293 / T), 2))
        psi1 = POL + 0.002629 * powerT1(alpha1, T) * powerT1(beta1, T)
        aa = 0.953 - 0.002314 * (pow(powerT2(tau2, T), 2) + powerT1(alpha2, T) * powerT1(beta2, T))
        
        d12 = math.log(pow(v2 / v1, aa)) + 1 - pow(v2 / v1, aa)
        activity_coefficient = (v2 / (R * T)) * (pow(lambda1 - lambda2, 2) +
                                                 (pow(rho1, 2) * pow(rho2, 2) * pow(powerT2(tau1, T) - powerT2(tau2, T), 2)) / psi1 +
                                                 (powerT1(alpha1, T) - powerT1(alpha2, T)) * (powerT1(beta1, T) - powerT1(beta2, T)) / xi1) + d12
        return math.exp(activity_coefficient)
    except (ZeroDivisionError, OverflowError, ValueError):
        return np.nan


# ==========================================
# 2. 加载参数库：提取第 D 列(SMILES) 作为键
# ==========================================
print("正在加载 MOSCED 参数库...")
wb = openpyxl.load_workbook('MOSCED_Data.xlsx', data_only=True)
sheet = wb['Data'] 

smiles_param_dict = {}

# openpyxl 索引对应：
# row[2] 是 D列 (SMILES)
# row[3] 到 row[9] 对应 E列 到 J列 (v, lambda, tau, q, alpha, beta)
for row in sheet['B3':'K134']:
    smiles_val = row[2].value
    if smiles_val:
        clean_smiles = str(smiles_val).strip()
        param_list = [obj.value for obj in row[3:9]]
        smiles_param_dict[clean_smiles] = param_list

print(f"参数库加载完毕！成功通过自带 SMILES 加载了 {len(smiles_param_dict)} 种物质的物性。")


# ==========================================
# 3. 读取你的数据集并进行批量预测
# ==========================================
csv_path = 'PGSSI_WinOrg.csv'
print(f"正在读取数据集: {csv_path} ...")
df = pd.read_csv(csv_path)

solute_col = 'Solute_SMILES'
solvent_col = 'Solvent_SMILES'
temp_col = 'T_K'

predictions = []
matched_count = 0

for idx, row in df.iterrows():
    solute_smiles = str(row[solute_col]).strip()
    solvent_smiles = str(row[solvent_col]).strip()
    T = float(row[temp_col])
    
    if solute_smiles in smiles_param_dict and solvent_smiles in smiles_param_dict:
        p_solute = smiles_param_dict[solute_smiles]
        p_solvent = smiles_param_dict[solvent_smiles]
        
        try:
            gamma_inf = calculate_mosced(
                v1=float(p_solute[0]),      v2=float(p_solvent[0]),
                lambda1=float(p_solute[1]), lambda2=float(p_solvent[1]),
                tau1=float(p_solute[2]),    tau2=float(p_solvent[2]),
                rho1=float(p_solute[3]),    rho2=float(p_solvent[3]),
                alpha1=float(p_solute[4]),  alpha2=float(p_solvent[4]),
                beta1=float(p_solute[5]),   beta2=float(p_solvent[5]),
                T=T
            )
            ln_gamma_inf = math.log(gamma_inf) if gamma_inf > 0 else np.nan
            predictions.append(ln_gamma_inf)
            matched_count += 1
        except:
            predictions.append(np.nan)
    else:
        predictions.append(np.nan)

# ==========================================
# 4. 保存预测结果
# ==========================================
df['ln_gamma_inf_pred_mosced'] = predictions
output_file = 'PGSSI_Predicted_Results.csv'
df.to_csv(output_file, index=False)

print(f"\n批量预测完成！")
print(f"总数据量: {len(df)} 条")
print(f"原生 SMILES 成功匹配并预测: {matched_count} 条")
print(f"结果已成功保存至: {output_file}")


# ==========================================
# 5. 缺失数据统计与导出 (新增部分)
# ==========================================
print("\n" + "="*40)
print("正在统计未能匹配到 MOSCED 参数的缺失数据...")
print("="*40)

# 找出由于没匹配到参数导致预测值为 NaN 的行
missing_df = df[df['ln_gamma_inf_pred_mosced'].isna()]

if len(missing_df) > 0:
    print(f"共有 {len(df) - matched_count} 条数据因缺失参数未被预测。")
    print("前 10 个未匹配成功的溶质和溶剂 SMILES 体系如下：")
    
    # 去重并打印前10行
    unique_missing = missing_df[[solute_col, solvent_col]].drop_duplicates()
    print(unique_missing.head(10))
    
    # 自动把所有未匹配成功的独特体系单独导出一个 CSV，方便后续研究或用 AI/GNN 填补参数
    missing_output_file = 'MOSCED_Missing_Substances.csv'
    unique_missing.to_csv(missing_output_file, index=False)
    print(f"\n所有独特的缺失物性体系已自动导出至: {missing_output_file}")
else:
    print("太棒了！数据集中所有物质体系均成功匹配，无缺失数据。")