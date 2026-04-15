"""PGSSI 预测入口。

输入一份包含 solvent / solute / T 的表格，输出：
1. 最终的 log-gamma 预测值
2. 模型内部预测的 K1、K2 参数
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_absolute_error, r2_score
from torch_geometric.loader import DataLoader

try:
    from PGSSI_3D_architecture import PGSSIModel
    from PGSSI_train import build_dataset
except ImportError:
    from src.models.PGSSI.PGSSI_3D_architecture import PGSSIModel
    from src.models.PGSSI.PGSSI_train import build_dataset


PROJECT_ROOT = Path(__file__).resolve().parents[3]
MODEL_DIR = Path(__file__).resolve().parent
CACHE_DIR = PROJECT_ROOT / "cache" / "pgssi_shared"


def pred_PGSSI_T(df, model_name, hyperparameters, T=None, cache_key=None):
    """对 dataframe 做预测。

    如果显式给了 T，就会覆盖输入表中的温度列。
    """
    df = df.copy()
    if T is not None:
        df["T"] = T
    elif "T" not in df.columns:
        raise ValueError("Input dataframe must include column 'T' when T is not provided.")

    # 预测阶段也复用训练时的数据构图逻辑和缓存格式。
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_suffix = "pred" if not cache_key else cache_key
    dataset = build_dataset(df, str(CACHE_DIR / f"{model_name}_{cache_suffix}.pt"))
    loader = DataLoader(dataset, batch_size=hyperparameters.get("batch_size", 32), shuffle=False, num_workers=0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PGSSIModel(
        hidden_dim=hyperparameters["hidden_dim"],
        enable_cross_interaction=hyperparameters.get("enable_cross_interaction", True),
    ).to(device)

    model_path = Path(hyperparameters["model_path"])
    if not model_path.is_absolute():
        model_path = PROJECT_ROOT / model_path
    if not model_path.exists():
        raise FileNotFoundError(f"Model weights not found: {model_path}")

    checkpoint = torch.load(model_path, map_location=device)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint
    model.load_state_dict(state_dict, strict=False)

    predictions = []
    k1_predictions = []
    k2_predictions = []
    sample_indices = []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            if batch is None:
                continue
            batch = batch.to(device)
            # 训练时最终监督的是 log-gamma；预测时把中间参数也一起导出。
            out = model(batch, return_dict=True)
            predictions.append(out["log_gamma"].view(-1).cpu().numpy())
            k1_predictions.append(out["k1"].view(-1).cpu().numpy())
            k2_predictions.append(out["k2"].view(-1).cpu().numpy())
            if hasattr(batch, "sample_index"):
                sample_indices.append(batch.sample_index.view(-1).cpu().numpy())

    df[model_name] = np.nan
    df[f"{model_name}_K1"] = np.nan
    df[f"{model_name}_K2"] = np.nan

    pred_values = np.concatenate(predictions) if predictions else np.array([], dtype=np.float32)
    k1_values = np.concatenate(k1_predictions) if k1_predictions else np.array([], dtype=np.float32)
    k2_values = np.concatenate(k2_predictions) if k2_predictions else np.array([], dtype=np.float32)
    row_indices = np.concatenate(sample_indices) if sample_indices else np.array([], dtype=np.int64)

    if row_indices.size > 0:
        df.loc[row_indices, model_name] = pred_values
        df.loc[row_indices, f"{model_name}_K1"] = k1_values
        df.loc[row_indices, f"{model_name}_K2"] = k2_values
    return df


def parse_args():
    """命令行参数定义。"""
    parser = argparse.ArgumentParser(description="PGSSI prediction")
    parser.add_argument("--input-path", type=str, required=True)
    parser.add_argument("--output-path", type=str, required=True)
    parser.add_argument("--model-path", type=str, default=str(MODEL_DIR / "PGSSI_best.pth"))
    parser.add_argument("--model-name", type=str, default="PGSSI_prediction")
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--disable-cross-interaction", action="store_true")
    parser.add_argument("--temperature", type=float, default=None)
    return parser.parse_args()


def main():
    """命令行入口。"""
    args = parse_args()
    df = pd.read_csv(args.input_path)

    hyperparameters = {
        "hidden_dim": args.hidden_dim,
        "batch_size": args.batch_size,
        "model_path": args.model_path,
        "enable_cross_interaction": not args.disable_cross_interaction,
    }

    input_path = Path(args.input_path)
    cache_prefix = input_path.stem
    cache_hash = hashlib.md5(str(input_path.resolve()).encode("utf-8")).hexdigest()[:8]
    cache_key = f"{cache_prefix}_pred_{cache_hash}"
    df_pred = pred_PGSSI_T(df, args.model_name, hyperparameters, T=args.temperature, cache_key=cache_key)

    output_path = Path(args.output_path)
    if not output_path.is_absolute():
        output_path = PROJECT_ROOT / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_pred.to_csv(output_path, index=False)

    print(f"Prediction completed. Results saved to: {output_path}")
    if "log-gamma" in df_pred.columns and args.model_name in df_pred.columns:
        valid_mask = df_pred[args.model_name].notna()
        y_true = df_pred.loc[valid_mask, "log-gamma"].to_numpy()
        y_pred = df_pred.loc[valid_mask, args.model_name].to_numpy()
        if y_true.size > 0:
            print(f"MAE: {mean_absolute_error(y_true, y_pred):.6f}")
            print(f"R2 : {r2_score(y_true, y_pred):.6f}")
        print(f"Predicted rows: {int(valid_mask.sum())}/{len(df_pred)}")


if __name__ == "__main__":
    main()
