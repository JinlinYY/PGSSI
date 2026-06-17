from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold


PROJECT_ROOT = Path(__file__).resolve().parents[2]
TRAIN_SCRIPT = PROJECT_ROOT / "src" / "models" / "PGSSI" / "PGSSI_train.py"
DEFAULT_INPUT = PROJECT_ROOT / "dataset" / "all" / "all_merged.csv"
DEFAULT_OUTPUT = PROJECT_ROOT / "runs" / "pgssi_ablation_strong_cv"
DEFAULT_CACHE = PROJECT_ROOT / "cache" / "pgssi_ablation_strong_cv"
REQUIRED_COLUMNS = ["Solvent_SMILES", "Solute_SMILES", "T", "log-gamma"]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run five-fold grouped cross-validation for the Table 2 PGSSI ablation study."
    )
    parser.add_argument("--input-path", type=str, default=str(DEFAULT_INPUT))
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT))
    parser.add_argument("--cache-dir", type=str, default=str(DEFAULT_CACHE))
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--valid-size", type=float, default=0.125)
    parser.add_argument("--max-folds", type=int, default=None)
    parser.add_argument("--max-experiments", type=int, default=None)
    parser.add_argument(
        "--experiment-names",
        type=str,
        default=None,
        help="Comma-separated experiment names to run. Defaults to all Table 2 variants.",
    )
    parser.add_argument(
        "--skip-full-model",
        action="store_true",
        help="Skip Full PGSSI when existing full-model CV results are reused.",
    )
    parser.add_argument(
        "--existing-split-dir",
        type=str,
        default=None,
        help="Directory containing fold_N_train/valid/test.csv files to reuse instead of regenerating CV splits.",
    )
    parser.add_argument("--limit-rows", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--model-name", type=str, default="PGSSI_StrongAblationCV")
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


def experiment_grid() -> list[dict]:
    return [
        {
            "name": "full_model",
            "label": "Full PGSSI",
            "flags": [],
            "description": "Full PGSSI model.",
            "table_order": 1,
        },
        {
            "name": "no_joint_3d_geometry",
            "label": "No Joint 3D Geometry",
            "flags": ["--disable-joint-3d-geometry"],
            "description": (
                "Remove joint 3D geometry by zeroing coordinates, intramolecular geometry attributes, "
                "orientation/dipole vectors, and replacing distance-derived cross-contact edges with "
                "deterministic pseudo cross edges."
            ),
            "table_order": 2,
        },
        {
            "name": "no_explicit_cross_interaction",
            "label": "No Explicit Cross Interaction",
            "flags": ["--disable-cross-interaction", "--disable-pair-readout"],
            "description": (
                "Remove explicit solute-solvent cross message passing and final solvent-solute "
                "pair contrast readout terms."
            ),
            "table_order": 3,
        },
        {
            "name": "no_physics_guided_cues",
            "label": "No Physics-Guided Cues",
            "flags": ["--disable-physics-prior", "--disable-physics-cues"],
            "description": (
                "Remove LJ/Coulomb priors and zero cross-interaction scalar physics cues, including "
                "distance, inverse distance, charge terms, and typed interaction indicators."
            ),
            "table_order": 4,
        },
        {
            "name": "direct_lngamma_regression",
            "label": "Direct ln gamma-infinity Regression",
            "flags": ["--direct-loggamma-head"],
            "description": "Predict log-gamma directly instead of using the inverse-temperature output structure.",
            "table_order": 5,
        },
    ]


def select_experiments(args) -> list[dict]:
    experiments = experiment_grid()
    if args.skip_full_model:
        experiments = [exp for exp in experiments if exp["name"] != "full_model"]
    if args.experiment_names:
        requested = [name.strip() for name in args.experiment_names.split(",") if name.strip()]
        known = {exp["name"] for exp in experiments}
        unknown = sorted(set(requested) - known)
        if unknown:
            raise ValueError(f"Unknown experiment name(s): {unknown}. Known names: {sorted(known)}")
        requested_set = set(requested)
        experiments = [exp for exp in experiments if exp["name"] in requested_set]
    if args.max_experiments is not None:
        experiments = experiments[: args.max_experiments]
    return experiments


def validate_columns(df: pd.DataFrame):
    missing = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(f"Input dataset is missing required columns: {missing}")


def pair_groups(df: pd.DataFrame) -> np.ndarray:
    return (df["Solvent_SMILES"].astype(str) + "||" + df["Solute_SMILES"].astype(str)).to_numpy()


def make_target_bins(target: pd.Series, n_bins: int = 10) -> np.ndarray:
    effective_bins = max(2, min(n_bins, int(target.nunique())))
    try:
        bins = pd.qcut(target, q=effective_bins, labels=False, duplicates="drop")
    except ValueError:
        bins = pd.Series(np.zeros(len(target), dtype=int), index=target.index)
    return bins.astype(int).to_numpy()


def split_train_valid(train_valid_df: pd.DataFrame, valid_size: float, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    groups = pair_groups(train_valid_df)
    splitter = GroupShuffleSplit(n_splits=1, test_size=valid_size, random_state=seed)
    train_idx, valid_idx = next(splitter.split(train_valid_df, groups=groups))
    train_df = train_valid_df.iloc[train_idx].reset_index(drop=True)
    valid_df = train_valid_df.iloc[valid_idx].reset_index(drop=True)
    return train_df, valid_df


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


def load_existing_split(source_split_dir: Path, fold: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    paths = {
        "train": source_split_dir / f"fold_{fold}_train.csv",
        "valid": source_split_dir / f"fold_{fold}_valid.csv",
        "test": source_split_dir / f"fold_{fold}_test.csv",
    }
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing existing split file(s) for fold {fold}: {missing}")
    split_frames = {}
    for name, path in paths.items():
        frame = pd.read_csv(path)
        validate_columns(frame)
        split_frames[name] = frame[REQUIRED_COLUMNS].dropna().reset_index(drop=True)
    return split_frames["train"], split_frames["valid"], split_frames["test"]


def prediction_path(run_dir: Path, train_path: Path, model_name: str, test_path: Path) -> Path:
    return run_dir / f"{train_path.stem}_{model_name}_{test_path.stem}_predictions.csv"


def metrics_path(run_dir: Path, train_path: Path, model_name: str, test_path: Path) -> Path:
    return run_dir / f"{train_path.stem}_{model_name}_{test_path.stem}_metrics.json"


def training_csv_path(run_dir: Path, train_path: Path, model_name: str) -> Path:
    return run_dir / f"{train_path.stem}_{model_name}_training.csv"


def run_training(
    args,
    exp: dict,
    fold: int,
    run_dir: Path,
    fold_cache_dir: Path,
    train_path: Path,
    valid_path: Path,
    test_path: Path,
) -> tuple[Path, Path, Path]:
    pred_path = prediction_path(run_dir, train_path, args.model_name, test_path)
    metric_path = metrics_path(run_dir, train_path, args.model_name, test_path)
    train_csv = training_csv_path(run_dir, train_path, args.model_name)

    if args.reuse_existing and pred_path.exists() and metric_path.exists() and train_csv.exists():
        print(f"\n=== Reusing fold {fold} | {exp['label']} ===", flush=True)
        return pred_path, metric_path, train_csv

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
        str(run_dir),
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
        str(args.seed + (fold * 100) + int(exp["table_order"])),
    ] + exp["flags"]
    if args.quiet_progress:
        cmd.append("--quiet-progress")

    env = os.environ.copy()
    env.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    if args.force_cpu:
        env["CUDA_VISIBLE_DEVICES"] = "-1"

    print(f"\n=== Fold {fold} | {exp['label']} ===", flush=True)
    print(" ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=PROJECT_ROOT, env=env, check=True)
    return pred_path, metric_path, train_csv


def metric_row(pred_df: pd.DataFrame, fold: int, exp: dict, train_csv: Path, metric_path: Path) -> dict:
    pred_column = "pred_log-gamma"
    subset = pred_df.dropna(subset=["log-gamma", pred_column])
    y_true = subset["log-gamma"].to_numpy(dtype=float)
    y_pred = subset[pred_column].to_numpy(dtype=float)
    abs_error = np.abs(y_true - y_pred)

    train_df = pd.read_csv(train_csv)
    best_idx = train_df["MAE_Valid"].idxmin()

    metrics = {}
    if metric_path.exists():
        with open(metric_path, "r", encoding="utf-8") as fh:
            metrics = json.load(fh)

    return {
        "fold": int(fold),
        "table_order": int(exp["table_order"]),
        "experiment": exp["label"],
        "name": exp["name"],
        "description": exp["description"],
        "flags": " ".join(exp["flags"]),
        "epochs_requested": int(metrics.get("epochs_requested", len(train_df))),
        "epochs_ran": int(len(train_df)),
        "best_valid_mae": float(train_df.loc[best_idx, "MAE_Valid"]),
        "best_valid_r2": float(train_df.loc[best_idx, "R2_Valid"]),
        "n_samples": int(len(subset)),
        "n_pairs": int(subset.groupby(["Solvent_SMILES", "Solute_SMILES"]).ngroups) if len(subset) else 0,
        "test_mae": float(mean_absolute_error(y_true, y_pred)) if len(subset) else np.nan,
        "test_rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))) if len(subset) else np.nan,
        "test_r2": float(r2_score(y_true, y_pred)) if len(subset) >= 2 else np.nan,
        "test_ae_le_01_pct": float(np.mean(abs_error <= 0.1) * 100.0) if len(subset) else np.nan,
        "test_ae_le_02_pct": float(np.mean(abs_error <= 0.2) * 100.0) if len(subset) else np.nan,
        "test_ae_le_03_pct": float(np.mean(abs_error <= 0.3) * 100.0) if len(subset) else np.nan,
        "test_num_skipped": int(metrics.get("num_skipped", 0)),
    }


def summarize_metrics(fold_df: pd.DataFrame) -> pd.DataFrame:
    metric_columns = [
        "n_samples",
        "n_pairs",
        "epochs_ran",
        "best_valid_mae",
        "best_valid_r2",
        "test_mae",
        "test_rmse",
        "test_r2",
        "test_ae_le_01_pct",
        "test_ae_le_02_pct",
        "test_ae_le_03_pct",
        "test_num_skipped",
    ]
    rows = []
    for (_, experiment, name), group in fold_df.groupby(["table_order", "experiment", "name"], sort=True):
        row = {
            "experiment": experiment,
            "name": name,
            "folds": int(group["fold"].nunique()),
        }
        for column in metric_columns:
            row[f"{column}_mean"] = float(group[column].mean(skipna=True))
            row[f"{column}_std"] = float(group[column].std(skipna=True, ddof=1)) if len(group[column].dropna()) > 1 else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def foldwise_r2_table(fold_df: pd.DataFrame) -> pd.DataFrame:
    pivot = fold_df.pivot_table(
        index=["table_order", "experiment", "name"],
        columns="fold",
        values="test_r2",
        aggfunc="first",
    ).reset_index()
    pivot.columns = [f"fold_{column}_r2" if isinstance(column, int) else column for column in pivot.columns]
    fold_columns = [column for column in pivot.columns if column.startswith("fold_")]
    pivot["mean_r2"] = pivot[fold_columns].mean(axis=1, skipna=True)
    pivot["std_r2"] = pivot[fold_columns].std(axis=1, skipna=True, ddof=1)
    return pivot.sort_values("table_order").drop(columns=["table_order"])


def write_outputs(output_dir: Path, fold_rows: list[dict]):
    if not fold_rows:
        return
    fold_df = pd.DataFrame(fold_rows).sort_values(["table_order", "fold"])
    summary_df = summarize_metrics(fold_df)
    r2_df = foldwise_r2_table(fold_df)

    fold_path = output_dir / "ablation_cv_fold_metrics.csv"
    summary_path = output_dir / "ablation_cv_summary_metrics.csv"
    r2_path = output_dir / "ablation_cv_r2_scores.csv"

    fold_df.to_csv(fold_path, index=False)
    summary_df.to_csv(summary_path, index=False)
    r2_df.to_csv(r2_path, index=False)
    fold_df.to_markdown(output_dir / "ablation_cv_fold_metrics.md", index=False)
    summary_df.to_markdown(output_dir / "ablation_cv_summary_metrics.md", index=False)
    r2_df.to_markdown(output_dir / "ablation_cv_r2_scores.md", index=False)

    print(f"\nSaved fold metrics to: {fold_path}", flush=True)
    print(f"Saved summary metrics to: {summary_path}", flush=True)
    print(f"Saved R2 scores to: {r2_path}", flush=True)
    print(summary_df.to_string(index=False), flush=True)


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

    dataset_summary = {
        "input_path": str(input_path.resolve()),
        "n_samples": int(len(df)),
        "n_pairs": int(df.groupby(["Solvent_SMILES", "Solute_SMILES"]).ngroups),
        "n_folds": int(args.n_folds),
        "valid_size_within_train_valid": float(args.valid_size),
        "splitter": "StratifiedGroupKFold grouped by solvent-solute pair and stratified by log-gamma quantile bins",
        "existing_split_dir": str(Path(args.existing_split_dir).resolve()) if args.existing_split_dir else None,
    }
    with open(output_dir / "ablation_cv_dataset_summary.json", "w", encoding="utf-8") as fh:
        json.dump(dataset_summary, fh, indent=2, ensure_ascii=False)
    pd.DataFrame([dataset_summary]).to_csv(output_dir / "ablation_cv_dataset_summary.csv", index=False)
    print(json.dumps(dataset_summary, indent=2, ensure_ascii=False), flush=True)

    experiments = select_experiments(args)
    print(
        "Selected experiments: " + ", ".join(exp["name"] for exp in experiments),
        flush=True,
    )

    fold_specs = []
    if args.existing_split_dir:
        source_split_dir = Path(args.existing_split_dir)
        if not source_split_dir.is_absolute():
            source_split_dir = PROJECT_ROOT / source_split_dir
        max_fold = args.max_folds if args.max_folds is not None else args.n_folds
        for fold in range(1, max_fold + 1):
            train_df, valid_df, test_df = load_existing_split(source_split_dir, fold)
            verify_no_pair_overlap(train_df, valid_df, test_df)
            fold_specs.append((fold, train_df, valid_df, test_df))
    else:
        groups = pair_groups(df)
        bins = make_target_bins(df["log-gamma"])
        splitter = StratifiedGroupKFold(n_splits=args.n_folds, shuffle=True, random_state=args.seed)
        for fold, (train_valid_idx, test_idx) in enumerate(splitter.split(df, bins, groups), start=1):
            if args.max_folds is not None and fold > args.max_folds:
                break
            train_valid_df = df.iloc[train_valid_idx].reset_index(drop=True)
            test_df = df.iloc[test_idx].reset_index(drop=True)
            train_df, valid_df = split_train_valid(train_valid_df, valid_size=args.valid_size, seed=args.seed + fold)
            verify_no_pair_overlap(train_df, valid_df, test_df)
            fold_specs.append((fold, train_df, valid_df, test_df))

    fold_rows = []
    for fold, train_df, valid_df, test_df in fold_specs:
        train_path = split_dir / f"fold_{fold}_train.csv"
        valid_path = split_dir / f"fold_{fold}_valid.csv"
        test_path = split_dir / f"fold_{fold}_test.csv"
        train_df[REQUIRED_COLUMNS].to_csv(train_path, index=False)
        valid_df[REQUIRED_COLUMNS].to_csv(valid_path, index=False)
        test_df[REQUIRED_COLUMNS].to_csv(test_path, index=False)

        fold_summary = {
            "fold": int(fold),
            "train_samples": int(len(train_df)),
            "valid_samples": int(len(valid_df)),
            "test_samples": int(len(test_df)),
            "train_pairs": int(train_df.groupby(["Solvent_SMILES", "Solute_SMILES"]).ngroups),
            "valid_pairs": int(valid_df.groupby(["Solvent_SMILES", "Solute_SMILES"]).ngroups),
            "test_pairs": int(test_df.groupby(["Solvent_SMILES", "Solute_SMILES"]).ngroups),
        }
        with open(output_dir / f"fold_{fold}_split_summary.json", "w", encoding="utf-8") as fh:
            json.dump(fold_summary, fh, indent=2, ensure_ascii=False)

        for exp in experiments:
            run_dir = output_dir / exp["name"] / f"fold_{fold}"
            fold_cache_dir = cache_dir / f"fold_{fold}"
            run_dir.mkdir(parents=True, exist_ok=True)
            fold_cache_dir.mkdir(parents=True, exist_ok=True)

            pred_path, metric_path, train_csv = run_training(
                args,
                exp,
                fold,
                run_dir,
                fold_cache_dir,
                train_path,
                valid_path,
                test_path,
            )
            pred_df = pd.read_csv(pred_path)
            fold_rows.append(metric_row(pred_df, fold, exp, train_csv, metric_path))
            write_outputs(output_dir, fold_rows)

    write_outputs(output_dir, fold_rows)


if __name__ == "__main__":
    main()
