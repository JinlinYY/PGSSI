import os
import subprocess
import pandas as pd
from sklearn.model_selection import KFold

# 待评估的数据集列表
DATASETS = [
    "PGSSI_Lazzaroni_2023.csv",
    "PGSSI_Lazzaroni_Original.csv",
    "PGSSI_OrginW.csv",
    "PGSSI_WinOrg.csv"
]

def run_pgssi_5fold_cv():
    # 设定随机种子，保证每次五折划分的一致性，方便消融实验对比
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    for ds_name in DATASETS:
        print(f"\n================ 启动 {ds_name} 的五折交叉验证 ================")
        df = pd.read_csv(ds_name)

        # 核心修复：数据列名对齐，适配 PGSSI_data.py 的特征提取逻辑
        if "T_K" in df.columns:
            df.rename(columns={"T_K": "T"}, inplace=True)
        if "ln_gamma_inf" in df.columns:
            df.rename(columns={"ln_gamma_inf": "log-gamma"}, inplace=True)

        out_dir_base = f"cv_results/{ds_name.replace('.csv', '')}"
        os.makedirs(out_dir_base, exist_ok=True)

        for fold, (train_idx, test_idx) in enumerate(kf.split(df)):
            fold_id = fold + 1
            print(f"\n---> 正在训练 Fold {fold_id}/5 ...")

            # 划分训练集和独立的测试集 (80% / 20%)
            train_val_df = df.iloc[train_idx]
            test_df = df.iloc[test_idx]

            # 进一步划分验证集 (从训练集中分出约10-12.5%用于 Early Stopping)
            valid_df = train_val_df.sample(frac=0.1, random_state=42)
            train_df = train_val_df.drop(valid_df.index)

            # 写入临时文件，供 PGSSI_train.py 读取
            train_path = f"tmp_train_{fold_id}.csv"
            valid_path = f"tmp_valid_{fold_id}.csv"
            test_path = f"tmp_test_{fold_id}.csv"

            train_df.to_csv(train_path, index=False)
            valid_df.to_csv(valid_path, index=False)
            test_df.to_csv(test_path, index=False)

            fold_out_dir = f"{out_dir_base}/fold_{fold_id}"

            # 组装 CLI 运行指令
            # 请根据你的硬件配置（如 RTX 3090/4090 的显存）适当调高 batch_size 以加速训练
            cmd = [
                "python", "PGSSI_train.py",
                "--train_path", train_path,
                "--valid_path", valid_path,
                "--test_path", test_path,
                "--output_dir", fold_out_dir,
                "--batch_size", "128", 
                "--epochs", "300" # 假设代码内配置了早停，epoch可设置较大基数
            ]

            try:
                # 阻塞式运行子进程
                subprocess.run(cmd, check=True)
            except subprocess.CalledProcessError as e:
                print(f"Fold {fold_id} 训练出错，退出码: {e.returncode}")

            # 清理该折的临时文件
            for f in [train_path, valid_path, test_path]:
                if os.path.exists(f):
                    os.remove(f)

        print(f"================ {ds_name} 实验完成 ================\n")

if __name__ == "__main__":
    run_pgssi_5fold_cv()