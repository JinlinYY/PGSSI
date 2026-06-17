from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold


PROJECT_ROOT = Path(__file__).resolve().parents[2]
TRAIN_SCRIPT = PROJECT_ROOT / "src" / "models" / "PGSSI" / "PGSSI_train.py"
DEFAULT_INPUT = PROJECT_ROOT / "dataset" / "all" / "all_merged.csv"
DEFAULT_OUTPUT = PROJECT_ROOT / "runs" / "water_nonwater_cv"
DEFAULT_CACHE = PROJECT_ROOT / "cache" / "water_nonwater_cv"
REQUIRED_COLUMNS = ["Solvent_SMILES", "Solute_SMILES", "T", "log-gamma"]


def parse_args():
    parser = argparse.ArgumentParser(description="Water/non-water grouped CV experiment for PGSSI.")
    parser.add_argument("--input-path", type=str, default=str(DEFAULT_INPUT))
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT))
    parser.add_argument("--cache-dir", type=str, default=str(DEFAULT_CACHE))
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--valid-size", type=float, default=0.125)
    parser.add_argument("--max-folds", type=int, default=None)
    parser.add_argument("--limit-rows", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--model-name", type=str, default="PGSSI_WaterCV")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=2.5e-4)
    parser.add_argument("--weight-decay", type=float, default=2e-5)
    parser.add_argument("--early-stopping-patience", type=int, default=20)
    parser.add_argument("--checkpoint-interval", type=int, default=10)
    parser.add_argument("--train-num-workers", type=int, default=0)
    parser.add_argument("--valid-num-workers", type=int, default=0)
    parser.add_argument("--force-cpu", action="store_true")
    parser.add_argument("--reuse-existing", action="store_true")
    parser.add_argument("--quiet-progress", action="store_true")
    return parser.parse_args()


def canonical_smiles(smiles: str) -> str | None:
    try:
        mol = Chem.MolFromSmiles(str(smiles))
        if mol is None:
            return None
        return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
    except Exception:
        return None


def is_water_smiles(smiles: str) -> bool:
    return canonical_smiles(smiles) == "O"


def add_water_flags(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["is_water_solvent"] = out["Solvent_SMILES"].map(is_water_smiles)
    out["is_water_solute"] = out["Solute_SMILES"].map(is_water_smiles)
    out["is_any_water"] = out["is_water_solvent"] | out["is_water_solute"]
    out["is_non_water"] = ~out["is_any_water"]
    return out


def validate_columns(df: pd.DataFrame):
    missing = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(f"Input dataset is missing required columns: {missing}")


def make_target_bins(target: pd.Series, n_bins: int = 10) -> np.ndarray:
    effective_bins = max(2, min(n_bins, int(target.nunique())))
    try:
        bins = pd.qcut(target, q=effective_bins, labels=False, duplicates="drop")
    except ValueError:
        bins = pd.Series(np.zeros(len(target), dtype=int), index=target.index)
    return bins.astype(int).to_numpy()


def pair_groups(df: pd.DataFrame) -> np.ndarray:
    return (df["Solvent_SMILES"].astype(str) + "||" + df["Solute_SMILES"].astype(str)).to_numpy()


def verify_no_pair_overlap(train_df: pd.DataFrame, valid_df: pd.DataFrame, test_df: pd.DataFrame):
    split_pairs = {
        "train": set(zip(train_df["Solvent_SMILES"], train_df["Solute_SMILES"])),
        "valid": set(zip(valid_df["Solvent_SMILES"], valid_df["Solute_SMILES"])),
        "test": set(zip(test_df["Solvent_SMILES"], test_df["Solute_SMILES"])),
    }
    names = list(split_pairs)
    for i, left in enumerate(names):
        for right in names[i + 1 :]:
            overlap = split_pairs[left].intersection(split_pairs[right])
            if overlap:
                raise ValueError(f"Pair leakage detected between {left} and {right}: {len(overlap)} pairs")


def split_train_valid(train_valid_df: pd.DataFrame, valid_size: float, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    groups = pair_groups(train_valid_df)
    splitter = GroupShuffleSplit(n_splits=1, test_size=valid_size, random_state=seed)
    train_idx, valid_idx = next(splitter.split(train_valid_df, groups=groups))
    train_df = train_valid_df.iloc[train_idx].reset_index(drop=True)
    valid_df = train_valid_df.iloc[valid_idx].reset_index(drop=True)
    return train_df, valid_df


def metric_row(df: pd.DataFrame, subset_name: str, fold: int, pred_column: str = "pred_log-gamma") -> dict:
    subset = df.dropna(subset=["log-gamma", pred_column])
    row = {
        "fold": fold,
        "subset": subset_name,
        "n_samples": int(len(subset)),
        "n_pairs": int(subset.groupby(["Solvent_SMILES", "Solute_SMILES"]).ngroups) if len(subset) else 0,
        "mae": np.nan,
        "rmse": np.nan,
        "r2": np.nan,
        "ae_le_01_pct": np.nan,
        "ae_le_02_pct": np.nan,
        "ae_le_03_pct": np.nan,
    }
    if subset.empty:
        return row

    y_true = subset["log-gamma"].to_numpy(dtype=float)
    y_pred = subset[pred_column].to_numpy(dtype=float)
    abs_error = np.abs(y_true - y_pred)
    row.update(
        {
            "mae": float(mean_absolute_error(y_true, y_pred)),
            "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
            "r2": float(r2_score(y_true, y_pred)) if len(subset) >= 2 else np.nan,
            "ae_le_01_pct": float(np.mean(abs_error <= 0.1) * 100.0),
            "ae_le_02_pct": float(np.mean(abs_error <= 0.2) * 100.0),
            "ae_le_03_pct": float(np.mean(abs_error <= 0.3) * 100.0),
        }
    )
    return row


def water_subset_metrics(pred_df: pd.DataFrame, fold: int) -> list[dict]:
    flagged = add_water_flags(pred_df)
    return [
        metric_row(flagged, "all", fold),
        metric_row(flagged[flagged["is_water_solvent"]], "water_as_solvent", fold),
        metric_row(flagged[flagged["is_water_solute"]], "water_as_solute", fold),
        metric_row(flagged[flagged["is_any_water"]], "any_water", fold),
        metric_row(flagged[flagged["is_non_water"]], "non_water", fold),
    ]


def summarize_metrics(metrics_df: pd.DataFrame) -> pd.DataFrame:
    metric_columns = ["n_samples", "n_pairs", "mae", "rmse", "r2", "ae_le_01_pct", "ae_le_02_pct", "ae_le_03_pct"]
    rows = []
    for subset, group in metrics_df.groupby("subset", sort=False):
        row = {"subset": subset, "folds": int(group["fold"].nunique())}
        for column in metric_columns:
            row[f"{column}_mean"] = float(group[column].mean(skipna=True))
            row[f"{column}_std"] = float(group[column].std(skipna=True, ddof=1)) if len(group[column].dropna()) > 1 else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def run_training(
    args,
    fold: int,
    fold_dir: Path,
    fold_cache_dir: Path,
    train_path: Path,
    valid_path: Path,
    test_path: Path,
) -> Path:
    prediction_path = fold_dir / f"{train_path.stem}_{args.model_name}_{test_path.stem}_predictions.csv"
    if args.reuse_existing and prediction_path.exists():
        return prediction_path

    cmd = [
        sys.executable,
        str(TRAIN_SCRIPT),
        "--train-path",
        str(train_path),
        "--valid-path",
        str(valid_path),
        "--test-path",
        str(test_path),
        "--run-dir",
        str(fold_dir),
        "--cache-dir",
        str(fold_cache_dir),
        "--model-name",
        args.model_name,
        "--hidden-dim",
        str(args.hidden_dim),
        "--lr",
        str(args.lr),
        "--n-epochs",
        str(args.epochs),
        "--batch-size",
        str(args.batch_size),
        "--weight-decay",
        str(args.weight_decay),
        "--early-stopping-patience",
        str(args.early_stopping_patience),
        "--checkpoint-interval",
        str(args.checkpoint_interval),
        "--train-num-workers",
        str(args.train_num_workers),
        "--valid-num-workers",
        str(args.valid_num_workers),
        "--seed",
        str(args.seed + fold),
    ]
    if args.quiet_progress:
        cmd.append("--quiet-progress")

    env = os.environ.copy()
    env.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    if args.force_cpu:
        env["CUDA_VISIBLE_DEVICES"] = "-1"

    print(f"\n=== Fold {fold}: training PGSSI ===", flush=True)
    print(" ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=PROJECT_ROOT, env=env, check=True)
    return prediction_path


def main():
    args = parse_args()
    input_path = Path(args.input_path)
    output_dir = Path(args.output_dir)
    cache_dir = Path(args.cache_dir)
    split_dir = output_dir / "splits"
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)
    split_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_path)
    validate_columns(df)
    df = df[REQUIRED_COLUMNS].dropna().reset_index(drop=True)
    if args.limit_rows is not None:
        df = df.sample(n=min(args.limit_rows, len(df)), random_state=args.seed).reset_index(drop=True)

    flagged_df = add_water_flags(df)
    dataset_summary = {
        "input_path": str(input_path.resolve()),
        "n_samples": int(len(flagged_df)),
        "n_pairs": int(flagged_df.groupby(["Solvent_SMILES", "Solute_SMILES"]).ngroups),
        "water_as_solvent_samples": int(flagged_df["is_water_solvent"].sum()),
        "water_as_solute_samples": int(flagged_df["is_water_solute"].sum()),
        "any_water_samples": int(flagged_df["is_any_water"].sum()),
        "non_water_samples": int(flagged_df["is_non_water"].sum()),
    }
    with open(output_dir / "water_nonwater_dataset_summary.json", "w", encoding="utf-8") as fh:
        json.dump(dataset_summary, fh, indent=2, ensure_ascii=False)
    pd.DataFrame([dataset_summary]).to_csv(output_dir / "water_nonwater_dataset_summary.csv", index=False)
    print(json.dumps(dataset_summary, indent=2, ensure_ascii=False), flush=True)

    groups = pair_groups(flagged_df)
    bins = make_target_bins(flagged_df["log-gamma"])
    splitter = StratifiedGroupKFold(n_splits=args.n_folds, shuffle=True, random_state=args.seed)

    fold_metric_rows = []
    for fold, (train_valid_idx, test_idx) in enumerate(splitter.split(flagged_df, bins, groups), start=1):
        if args.max_folds is not None and fold > args.max_folds:
            break

        train_valid_df = flagged_df.iloc[train_valid_idx].reset_index(drop=True)
        test_df = flagged_df.iloc[test_idx].reset_index(drop=True)
        train_df, valid_df = split_train_valid(train_valid_df, valid_size=args.valid_size, seed=args.seed + fold)
        verify_no_pair_overlap(train_df, valid_df, test_df)

        fold_dir = output_dir / f"fold_{fold}"
        fold_cache_dir = cache_dir / f"fold_{fold}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        fold_cache_dir.mkdir(parents=True, exist_ok=True)

        train_path = split_dir / f"fold_{fold}_train.csv"
        valid_path = split_dir / f"fold_{fold}_valid.csv"
        test_path = split_dir / f"fold_{fold}_test.csv"
        train_df[REQUIRED_COLUMNS].to_csv(train_path, index=False)
        valid_df[REQUIRED_COLUMNS].to_csv(valid_path, index=False)
        test_df[REQUIRED_COLUMNS].to_csv(test_path, index=False)

        fold_summary = {
            "fold": fold,
            "train_samples": int(len(train_df)),
            "valid_samples": int(len(valid_df)),
            "test_samples": int(len(test_df)),
            "train_pairs": int(train_df.groupby(["Solvent_SMILES", "Solute_SMILES"]).ngroups),
            "valid_pairs": int(valid_df.groupby(["Solvent_SMILES", "Solute_SMILES"]).ngroups),
            "test_pairs": int(test_df.groupby(["Solvent_SMILES", "Solute_SMILES"]).ngroups),
        }
        with open(fold_dir / "fold_split_summary.json", "w", encoding="utf-8") as fh:
            json.dump(fold_summary, fh, indent=2, ensure_ascii=False)

        prediction_path = run_training(args, fold, fold_dir, fold_cache_dir, train_path, valid_path, test_path)
        pred_df = pd.read_csv(prediction_path)
        current_rows = water_subset_metrics(pred_df, fold)
        fold_metric_rows.extend(current_rows)
        pd.DataFrame(current_rows).to_csv(fold_dir / "water_nonwater_metrics.csv", index=False)

    metrics_df = pd.DataFrame(fold_metric_rows)
    summary_df = summarize_metrics(metrics_df)
    metrics_path = output_dir / "water_nonwater_fold_metrics.csv"
    summary_path = output_dir / "water_nonwater_summary_metrics.csv"
    metrics_df.to_csv(metrics_path, index=False)
    summary_df.to_csv(summary_path, index=False)
    metrics_df.to_markdown(output_dir / "water_nonwater_fold_metrics.md", index=False)
    summary_df.to_markdown(output_dir / "water_nonwater_summary_metrics.md", index=False)

    print(f"\nSaved fold metrics to: {metrics_path}", flush=True)
    print(f"Saved summary metrics to: {summary_path}", flush=True)
    print(summary_df.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
