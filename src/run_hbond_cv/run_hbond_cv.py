import json
import subprocess
import time
from pathlib import Path
import numpy as np
import pandas as pd
from rdkit import Chem
from sklearn.model_selection import KFold

# 路径配置
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_PATH = PROJECT_ROOT / "dataset" / "all" / "all_merged.csv"
OUTPUT_DIR = PROJECT_ROOT / "runs" / "pgssi_hbond_cv"
CACHE_DIR = PROJECT_ROOT / "cache" / "pgssi_hbond_cv"

def is_hbond_rich(smiles):
    """
    检查分子是否包含 O、N 或 F（典型的氢键供体/受体）。
    """
    mol = Chem.MolFromSmiles(smiles)
    if not mol:
        return False
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() in (7, 8, 9):  # 7:N, 8:O, 9:F
            return True
    return False

def filter_dataset(df):
    """筛选富含氢键的体系子集"""
    print("正在通过 RDKit 自动筛选富含氢键的体系子集 (O/N/F)...")
    
    mask = df.apply(lambda row: is_hbond_rich(row['Solvent_SMILES']) and is_hbond_rich(row['Solute_SMILES']), axis=1)
    filtered_df = df[mask].copy()
    filtered_df.reset_index(drop=True, inplace=True)
    
    print(f"原始数据集大小: {len(df)}")
    print(f"富氢键子集大小: {len(filtered_df)}")
    return filtered_df

def run_cross_validation(df, n_splits=5):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    metrics_list = []
    
    # 记录总开始时间
    total_start_time = time.time()

    for fold, (train_valid_idx, test_idx) in enumerate(kf.split(df)):
        fold_start_time = time.time()
        print(f"\n{'='*50}")
        print(f" 🚀 正在启动 第 {fold + 1}/{n_splits} 折训练")
        print(f"{'='*50}")
        
        # 数据集划分
        np.random.seed(42 + fold)
        np.random.shuffle(train_valid_idx)
        split_point = int(len(train_valid_idx) * 0.8)
        train_idx = train_valid_idx[:split_point]
        valid_idx = train_valid_idx[split_point:]
        
        train_df = df.iloc[train_idx]
        valid_df = df.iloc[valid_idx]
        test_df = df.iloc[test_idx]
        
        fold_dir = OUTPUT_DIR / f"fold_{fold}"
        fold_dir.mkdir(exist_ok=True)
        
        train_path = fold_dir / "train.csv"
        valid_path = fold_dir / "valid.csv"
        test_path = fold_dir / "test.csv"
        
        train_df.to_csv(train_path, index=False)
        valid_df.to_csv(valid_path, index=False)
        test_df.to_csv(test_path, index=False)
        
        # 针对 3090 的超大显存和多核进行提速：batch_size=128, workers=8
        cmd = [
            "python", str(PROJECT_ROOT / "src" / "models" / "PGSSI" / "PGSSI_train.py"),
            "--train-path", str(train_path),
            "--valid-path", str(valid_path),
            "--test-path", str(test_path),
            "--run-dir", str(fold_dir),
            "--cache-dir", str(CACHE_DIR / f"fold_{fold}"),
            "--model-name", "PGSSI_HBond",
            "--n-epochs", "200", 
            "--batch-size", "128", 
            "--train-num-workers", "8",
            "--valid-num-workers", "8"
        ]
        
        subprocess.run(cmd, check=True)
        
        # 读取指标
        metrics_file = fold_dir / "train_PGSSI_HBond_test_metrics.json"
        with open(metrics_file, 'r', encoding='utf-8') as f:
            fold_metrics = json.load(f)
            metrics_list.append(fold_metrics)
            print(f"✅ 第 {fold + 1} 折完成 | 测试集 MAE: {fold_metrics['mae']:.4f}")
            
        # 计算进度和剩余时间预估
        fold_elapsed = time.time() - fold_start_time
        total_elapsed = time.time() - total_start_time
        avg_time_per_fold = total_elapsed / (fold + 1)
        folds_left = n_splits - (fold + 1)
        eta_seconds = avg_time_per_fold * folds_left
        
        print(f"⏱️ 本折耗时: {fold_elapsed/60:.1f} 分钟")
        if folds_left > 0:
            print(f"⏳ 预计全部跑完还需: {eta_seconds/60:.1f} 分钟 (约 {eta_seconds/3600:.1f} 小时)")

    # 汇总输出
    print(f"\n{'='*50}")
    print(" 🎉 5 折交叉验证结果汇总 (富氢键子集)")
    print(f"{'='*50}")
    
    mae_scores = [m['mae'] for m in metrics_list]
    rmse_scores = [m['rmse'] for m in metrics_list]
    r2_scores = [m['r2'] for m in metrics_list]
    
    print(f"MAE:  {np.mean(mae_scores):.4f} ± {np.std(mae_scores):.4f}")
    print(f"RMSE: {np.mean(rmse_scores):.4f} ± {np.std(rmse_scores):.4f}")
    print(f"R2:   {np.mean(r2_scores):.4f} ± {np.std(r2_scores):.4f}")

    summary_df = pd.DataFrame(metrics_list)
    summary_df.loc['Mean'] = summary_df.mean(numeric_only=True)
    summary_df.loc['Std'] = summary_df.std(numeric_only=True)
    summary_df.to_csv(OUTPUT_DIR / "hbond_cv_summary.csv", index=False)
    
    total_time_hours = (time.time() - total_start_time) / 3600
    print(f"\n✅ 实验全部结束！总耗时: {total_time_hours:.2f} 小时。")
    print(f"📊 详细结果已保存至: {OUTPUT_DIR / 'hbond_cv_summary.csv'}")

if __name__ == "__main__":
    full_df = pd.read_csv(DATA_PATH)
    hbond_df = filter_dataset(full_df)
    run_cross_validation(hbond_df, n_splits=5)