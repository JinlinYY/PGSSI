import os
import json
import subprocess
import pandas as pd
import numpy as np
import datetime
import glob
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# 待评估的数据集列表
DATASETS = [
    "IdacRecLazzaroniDb2023.csv",
    "IdacRecLazzaroniDb.csv",
    "IdacRecJaubert+TdeOrginW.csv",
    "IdacRecJaubert+TdeWinOrg.csv"
]

def run_pgssi_5fold_cv():
    # 核心升级 1：生成带有时间戳的独立实验根目录，防止多轮实验互相覆盖
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_root_dir = f"cv_results/exp_{timestamp}"
    os.makedirs(exp_root_dir, exist_ok=True)
    
    # 设定随机种子，保证每次五折划分的一致性，方便消融实验对比
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    
    # 用于收集所有数据集、每一折的指标结果
    all_metrics = []

    # 核心升级 2：获取当前进程号，防止在服务器上多开脚本时临时文件冲突
    pid = os.getpid()

    for ds_name in DATASETS:
        print(f"\n================ 启动 {ds_name} 的五折交叉验证 ================")
        df = pd.read_csv(ds_name)

        # 数据列名对齐，适配 PGSSI_data.py 的特征提取逻辑
        if "T_K" in df.columns:
            df.rename(columns={"T_K": "T"}, inplace=True)
        if "ln_gamma_inf" in df.columns:
            df.rename(columns={"ln_gamma_inf": "log-gamma"}, inplace=True)

        # 结果保存在当前时间戳专属目录下
        out_dir_base = f"{exp_root_dir}/{ds_name.replace('.csv', '')}"
        os.makedirs(out_dir_base, exist_ok=True)

        for fold, (train_idx, test_idx) in enumerate(kf.split(df)):
            fold_id = fold + 1
            print(f"\n---> 正在训练 Fold {fold_id}/5 (实验编号: {timestamp}) ...")

            # 划分训练集和独立的测试集 (80% / 20%)
            train_val_df = df.iloc[train_idx]
            test_df = df.iloc[test_idx]

            # 进一步划分验证集 (从训练集中分出约10-12.5%用于 Early Stopping)
            valid_df = train_val_df.sample(frac=0.1, random_state=42)
            train_df = train_val_df.drop(valid_df.index)

            # 提取数据集的前缀名，用于区分临时文件
            ds_stem = ds_name.replace('.csv', '')

            # 写入临时文件，文件名加入 ds_stem 和进程 PID 彻底防冲突
            train_path = f"tmp_train_{ds_stem}_pid{pid}_fold{fold_id}.csv"
            valid_path = f"tmp_valid_{ds_stem}_pid{pid}_fold{fold_id}.csv"
            test_path = f"tmp_test_{ds_stem}_pid{pid}_fold{fold_id}.csv"

            train_df.to_csv(train_path, index=False)
            valid_df.to_csv(valid_path, index=False)
            test_df.to_csv(test_path, index=False)

            fold_out_dir = f"{out_dir_base}/fold_{fold_id}"

            # 组装 CLI 运行指令（已添加 --cache-dir 隔离 PyG 缓存）
            cmd = [
                "python", "PGSSI_train.py",
                "--train-path", train_path,
                "--valid-path", valid_path,
                "--test-path", test_path,
                "--run-dir", fold_out_dir,
                "--cache-dir", fold_out_dir,  # 将缓存完全隔离在此次实验的当前折文件夹中
                "--batch-size", "128", 
                "--n-epochs", "300"
            ]

            try:
                # 阻塞式运行子进程
                subprocess.run(cmd, check=True)
                
                # ==========================================
                # 1. 计算当前折 PGL 模型的指标 (基于测试集)
                # ==========================================
                fold_results = {"Dataset": ds_name, "Fold": fold_id}
                
                # 【需修改】请将 "PGL_pred" 替换为数据集中实际的 PGL 预测值列名
                # 修改前: pgl_col = "PGL_pred"
                pgl_col = "ln_PGL_Predicted" 
                if pgl_col in test_df.columns and "log-gamma" in test_df.columns:
                    y_true = test_df["log-gamma"]
                    y_pgl = test_df[pgl_col]
                    fold_results["PGL_RMSE"] = np.sqrt(mean_squared_error(y_true, y_pgl))
                    fold_results["PGL_MAE"] = mean_absolute_error(y_true, y_pgl)
                    fold_results["PGL_R2"] = r2_score(y_true, y_pgl)
                
                # ==========================================
                # 2. 读取当前折 PGSSI 模型的指标 (自动适配 test_summary.json)
                # ==========================================
                # 使用通配符查找文件夹下的 *_test_summary.json 文件
                summary_files = glob.glob(os.path.join(fold_out_dir, "*_test_summary.json"))
                
                if summary_files:
                    metrics_path = summary_files[0] # 取匹配到的第一个文件
                    try:
                        with open(metrics_path, "r", encoding="utf-8") as f:
                            pgssi_data = json.load(f)
                            
                            # 因为数据被 test_name 包裹了一层，我们需要剥开它
                            if pgssi_data and isinstance(pgssi_data, dict):
                                # 提取里面那层真正包含 rmse、mae 的字典
                                inner_metrics = list(pgssi_data.values())[0]
                                
                                # 兼容大小写键名，确保万无一失
                                fold_results["PGSSI_RMSE"] = inner_metrics.get("rmse", inner_metrics.get("RMSE"))
                                fold_results["PGSSI_MAE"] = inner_metrics.get("mae", inner_metrics.get("MAE"))
                                fold_results["PGSSI_R2"] = inner_metrics.get("r2", inner_metrics.get("R2"))
                    except json.JSONDecodeError:
                        print(f"⚠️ 警告: {metrics_path} 文件损坏，无法解析。")
                else:
                    print(f"⚠️ 提示: 未在 {fold_out_dir} 找到任何 *_test_summary.json 文件。")

                all_metrics.append(fold_results)

            except subprocess.CalledProcessError as e:
                print(f"❌ Fold {fold_id} 训练出错，退出码: {e.returncode}")

            # 清理该折的临时文件
            for f in [train_path, valid_path, test_path]:
                if os.path.exists(f):
                    os.remove(f)

        print(f"================ {ds_name} 实验完成 ================\n")

    # ==========================================
    # 3. 汇总所有结果，计算平均值并输出表格
    # ==========================================
    if all_metrics:
        results_df = pd.DataFrame(all_metrics)
        
        # 计算每个数据集的五折平均指标
        summary_list = []
        # 获取所有数值类型的列（排除 Fold 编号）
        numeric_cols = results_df.select_dtypes(include='number').columns.drop('Fold', errors='ignore')
        
        for ds in DATASETS:
            ds_data = results_df[results_df["Dataset"] == ds]
            if not ds_data.empty:
                mean_metrics = ds_data[numeric_cols].mean().to_dict()
                mean_metrics["Dataset"] = ds
                mean_metrics["Fold"] = "Average"
                summary_list.append(mean_metrics)
                
        # 将平均值追加到数据表中
        summary_df = pd.DataFrame(summary_list)
        final_df = pd.concat([results_df, summary_df], ignore_index=True)
        
        # 为了美观，对表格进行排序：按数据集排在一起，Average 放在每个数据集的最后
        final_df["Fold_sort"] = final_df["Fold"].apply(lambda x: 999 if x == "Average" else x)
        final_df = final_df.sort_values(by=["Dataset", "Fold_sort"]).drop(columns=["Fold_sort"])
        
        # 将最终汇总表格也保存在带时间戳的实验根目录中
        final_csv_path = f"{exp_root_dir}/all_datasets_summary_metrics.csv"
        final_df.to_csv(final_csv_path, index=False)
        print(f"✅ 所有实验全部结束！最终指标对比汇总表已安全保存至: {final_csv_path}\n")
        
        # 在控制台打印预览
        print(final_df.to_string(index=False))

if __name__ == "__main__":
    run_pgssi_5fold_cv()