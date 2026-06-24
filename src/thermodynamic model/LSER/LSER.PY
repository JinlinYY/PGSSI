import pandas as pd
import numpy as np
from rdkit import Chem
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GroupKFold
from sklearn.cluster import KMeans
from sklearn.metrics import mean_absolute_error, r2_score
import warnings
import os

warnings.filterwarnings('ignore')

# ==========================================
# 1. 辅助函数：SMILES 标准化
# ==========================================
def canonicalize_smiles(smiles):
    s_str = str(smiles).strip()
    if pd.isna(smiles) or s_str == '' or 'unknown' in s_str.lower(): return np.nan
    try:
        mol = Chem.MolFromSmiles(s_str)
        if mol is not None: return Chem.MolToSmiles(mol, isomericSmiles=False)
        return np.nan
    except: return np.nan

# ==========================================
# 2. 全局预加载：提取 Abraham 物理描述符作为“分子基因库”
# ==========================================
print("==========================================")
print("正在全局加载 Abraham 物理描述符 (只需加载一次)...")
try: df_abraham = pd.read_excel('Dataset_and_Predictions.xlsx', sheet_name='dataset_and_outliers')
except: df_abraham = pd.read_excel('Dataset_and_Predictions.xlsx', sheet_name=0)

df_abraham.columns = [str(col).strip().replace('\n', ' ') for col in df_abraham.columns]

def find_col(letter, columns):
    for col in columns:
        if col.upper().replace(' ', '').startswith(f"{letter}EXPERIMENT"): return col
    for col in columns:
        if col.upper().startswith(letter): return col
    raise ValueError(f"缺少列 {letter}")

col_E, col_S, col_A, col_B, col_V = find_col('E', df_abraham.columns), find_col('S', df_abraham.columns), find_col('A', df_abraham.columns), find_col('B', df_abraham.columns), find_col('V', df_abraham.columns)

df_abraham['SMILES_can'] = df_abraham['SMILES'].apply(canonicalize_smiles)
df_abraham_unique = df_abraham.dropna(subset=['SMILES_can']).drop_duplicates(subset=['SMILES_can'])

# 修改点 1：使用严格过滤规则构建物理基因库
feature_pool = {}
for _, row in df_abraham_unique.iterrows():
    feats = [row[col_E], row[col_S], row[col_A], row[col_B], row[col_V]]
    # 严格把关：只有当5个描述符全部存在（非NaN）时，才将其纳入基因库
    if not any(pd.isna(x) for x in feats):
        feature_pool[row['SMILES_can']] = feats

print(f"💡 成功构建包含 {len(feature_pool)} 个分子的物理基因库！")
print("==========================================\n")

# ==========================================
# 3. 定义批量处理的输入输出列表
# ==========================================
input_files = [
    'PGSSI_WinOrg.csv',
    'PGSSI_OrginW.csv',
    'PGSSI_Lazzaroni_Original.csv',
    'PGSSI_Lazzaroni_2023.csv'
]

output_files = [
    'IdacRecJaubert+TdeWinOrg.csv',
    'IdacRecJaubert+TdeOrginW.csv',
    'IdacRecLazzaroniDb.csv',
    'IdacRecLazzaroniDb2023.csv'
]

# --- 这里可以选择你想要的 Baseline 强度 (默认 MLR 传统基准) ---
METHOD_TYPE = "MLR"  # 可改为 "RF"

# ==========================================
# 4. 批量循环处理每个数据集
# ==========================================
for in_file, out_file in zip(input_files, output_files):
    print(f"\n>>> 开始处理数据集: {in_file}")
    
    if not os.path.exists(in_file):
        print(f"❌ 找不到文件 {in_file}，将跳过。")
        continue

    try:
        df_user = pd.read_csv(in_file)
    except Exception as e:
        print(f"❌ 读取 {in_file} 失败: {e}")
        continue

    df_user['Solute_SMILES_can'] = df_user['Solute_SMILES'].apply(canonicalize_smiles)
    df_user['Solvent_SMILES_can'] = df_user['Solvent_SMILES'].apply(canonicalize_smiles)

    X_list, y_list, kept_rows = [], [], []
    for _, row in df_user.iterrows():
        if row['Solute_SMILES_can'] in feature_pool and row['Solvent_SMILES_can'] in feature_pool:
            t_val = row['T_K']
            y_val = row['ln_gamma_inf']
            
            # 修改点 2：双重保险，如果目标文件里的温度或目标预测值有空缺，也跳过
            if pd.isna(t_val) or pd.isna(y_val):
                continue
                
            X_list.append(feature_pool[row['Solute_SMILES_can']] + feature_pool[row['Solvent_SMILES_can']] + [t_val])
            y_list.append(y_val)
            kept_rows.append(row)

    if len(X_list) == 0:
        print(f"⚠️ {in_file} 中没有找到任何能够匹配基因库的数据，跳过。")
        continue

    X = np.array(X_list)
    y = np.array(y_list)
    print(f"✅ 成功匹配并解锁了 {len(X)} 条体系数据！")

    # --- 聚类空间外推逻辑 ---
    solute_groups = [str(row['Solute_SMILES_can']) for row in kept_rows]
    solvent_groups = [str(row['Solvent_SMILES_can']) for row in kept_rows]

    if len(set(solute_groups)) >= len(set(solvent_groups)) and len(set(solute_groups)) >= 5:
        target_smiles = solute_groups
        target_type = "溶质 (Solute)"
    else:
        target_smiles = solvent_groups
        target_type = "溶剂 (Solvent)"

    unique_smiles = list(set(target_smiles))
    unique_features = [feature_pool[s] for s in unique_smiles]
    
    # 动态适应簇的数量 (如果独立分子数小于5，就聚少一点)
    n_clusters = min(5, len(unique_smiles))
    print(f"=> 聚类目标：{target_type}，聚为 {n_clusters} 簇")

    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(unique_features)
    smiles_to_cluster = dict(zip(unique_smiles, cluster_labels))

    groups = np.array([smiles_to_cluster[s] for s in target_smiles])

    # 根据可用簇的数量进行 K 折
    n_splits = min(5, n_clusters)
    if n_splits < 2:
        print(f"⚠️ 分子种类仅为 {len(unique_smiles)} 种，无法进行交叉验证，跳过训练。")
        continue

    gkf = GroupKFold(n_splits=n_splits)

    oof_predictions = np.zeros(len(X))
    fold = 1

    for train_idx, test_idx in gkf.split(X, y, groups=groups):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        if METHOD_TYPE == "RF":
            model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        else:
            model = LinearRegression()
            
        model.fit(X_train, y_train)
        oof_predictions[test_idx] = model.predict(X_test)
        
        fold_mae = mean_absolute_error(y_test, oof_predictions[test_idx])
        fold_r2 = r2_score(y_test, oof_predictions[test_idx])
        fold += 1

    final_mae = mean_absolute_error(y, oof_predictions)
    final_r2 = r2_score(y, oof_predictions)

    print(f"📊 {in_file} 最终指标 ({METHOD_TYPE} 算法，化学簇外推):")
    print(f"   Overall MAE: {final_mae:.4f}")
    print(f"   Overall R2 Score: {final_r2:.4f}")

    # 保存文件
    df_result = pd.DataFrame(kept_rows)
    df_result['QSPR_ln_gamma_inf_pred'] = oof_predictions
    df_result.to_csv(out_file, index=False)
    print(f"💾 已保存预测结果至: {out_file}")

print("\n🎉 所有数据集处理完毕！")