from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, train_test_split


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = Path(__file__).resolve().parent
TRAIN_SCRIPT = PROJECT_ROOT / "src" / "models" / "PGSSI" / "PGSSI_train.py"
DEFAULT_DATA_PATH = PROJECT_ROOT / "dataset" / "all" / "all_merged.csv"
DEFAULT_FOLDS_DIR = SCRIPT_DIR / "folds"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "outputs_5fold"
DEFAULT_CACHE_DIR = SCRIPT_DIR / "cache" / "datasets"
DEFAULT_PAIR_CACHE_DIR = SCRIPT_DIR / "cache" / "pair_graphs"
REQUIRED_COLUMNS = ["Solvent_SMILES", "Solute_SMILES", "T", "log-gamma"]

INTERACTION_MASKS = [
    "full",
    "no_hbond",
    "no_aromatic_pi",
    "no_dipole",
    "no_hydrophobic_polar",
    "no_all_interaction_types",
]

MASK_LABELS = {
    "full": "Full PGSSI",
    "no_hbond": "No H-bond tendency",
    "no_aromatic_pi": "No aromatic/pi",
    "no_dipole": "No dipole alignment",
    "no_hydrophobic_polar": "No hydrophobic/polar",
    "no_all_interaction_types": "No interaction-type features",
}

MASKED_FEATURES = {
    "full": "None",
    "no_hbond": "5 hbond_tendency",
    "no_aromatic_pi": "6 aromatic_pair; 7 pi_stacking_align",
    "no_dipole": "8 dipole_align; 9 dipole_opposition",
    "no_hydrophobic_polar": "10 hydrophobic_pair; 11 polar_pair; 12 hydrophobic_polar",
    "no_all_interaction_types": "5-12 all interaction-type indicators",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run 5-fold PGSSI interaction-type ablation experiments.")
    parser.add_argument("--data-path", type=str, default=str(DEFAULT_DATA_PATH))
    parser.add_argument("--folds-dir", type=str, default=str(DEFAULT_FOLDS_DIR))
    parser.add_argument("--fold-prefix", type=str, default="interaction_type")
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--folds", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    parser.add_argument("--interaction-mask-names", nargs="+", choices=INTERACTION_MASKS, default=INTERACTION_MASKS)
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--cache-dir", type=str, default=str(DEFAULT_CACHE_DIR))
    parser.add_argument("--pair-cache-dir", type=str, default=str(DEFAULT_PAIR_CACHE_DIR))
    parser.add_argument("--python-executable", type=str, default=sys.executable)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--early-stopping-patience", type=int, default=50)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--checkpoint-interval", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-num-workers", type=int, default=0)
    parser.add_argument("--valid-num-workers", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--force", action="store_true", help="Run even if the expected metrics file already exists.")
    parser.add_argument("--force-regenerate-folds", action="store_true")
    return parser.parse_args()


def resolve_path(path: str) -> Path:
    value = Path(path)
    if not value.is_absolute():
        value = PROJECT_ROOT / value
    return value


def fold_paths(folds_dir: Path, fold_prefix: str, fold: int) -> tuple[Path, Path, Path]:
    fold_tag = f"{fold:02d}"
    train_path = folds_dir / f"{fold_prefix}_fold{fold_tag}_train.csv"
    valid_path = folds_dir / f"{fold_prefix}_fold{fold_tag}_valid.csv"
    test_path = folds_dir / f"{fold_prefix}_fold{fold_tag}_test.csv"
    missing = [path for path in (train_path, valid_path, test_path) if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing fold file(s): {missing}")
    return train_path, valid_path, test_path


def validate_columns(df: pd.DataFrame) -> None:
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Input dataset is missing required columns: {missing}")


def make_bins(values: pd.Series, n_bins: int = 10) -> pd.Series:
    effective_bins = max(2, min(n_bins, int(values.nunique())))
    try:
        bins = pd.qcut(values, q=effective_bins, labels=False, duplicates="drop")
    except ValueError:
        bins = pd.Series(np.zeros(len(values), dtype=int), index=values.index)
    return bins.astype(int)


def maybe_stratify(values: pd.Series):
    counts = values.value_counts()
    if len(counts) < 2 or int(counts.min()) < 2:
        return None
    return values


def write_split_summary(
    rows: list[dict],
    dataset_name: str,
    fold: int,
    split_name: str,
    split_df: pd.DataFrame,
    path: Path,
) -> None:
    pair_count = split_df[["Solvent_SMILES", "Solute_SMILES"]].drop_duplicates().shape[0]
    rows.append(
        {
            "dataset": dataset_name,
            "fold": int(fold),
            "split": split_name,
            "path": str(path),
            "samples": int(len(split_df)),
            "unique_pairs": int(pair_count),
            "unique_solute": int(split_df["Solute_SMILES"].nunique()),
            "unique_solvent": int(split_df["Solvent_SMILES"].nunique()),
            "log_gamma_mean": float(split_df["log-gamma"].mean()),
            "log_gamma_std": float(split_df["log-gamma"].std(ddof=0)),
        }
    )


def ensure_interaction_type_folds(args: argparse.Namespace) -> Path:
    data_path = resolve_path(args.data_path)
    folds_dir = resolve_path(args.folds_dir)
    folds_dir.mkdir(parents=True, exist_ok=True)
    expected = [
        folds_dir / f"{args.fold_prefix}_fold{fold:02d}_{split}.csv"
        for fold in args.folds
        for split in ("train", "valid", "test")
    ]
    if not args.force_regenerate_folds and all(path.exists() for path in expected):
        return folds_dir

    df = pd.read_csv(data_path)
    validate_columns(df)
    work_df = df.copy()
    work_df["pair_key"] = list(zip(work_df["Solvent_SMILES"], work_df["Solute_SMILES"]))
    pair_df = (
        work_df.groupby("pair_key", sort=False)
        .agg(
            rows=("pair_key", "size"),
            target_mean=("log-gamma", "mean"),
            solvent=("Solvent_SMILES", "first"),
            solute=("Solute_SMILES", "first"),
        )
        .reset_index()
    )
    pair_df["target_bin"] = make_bins(pair_df["target_mean"], n_bins=10)

    if args.n_folds < 2:
        raise ValueError("--n-folds must be at least 2")
    pair_kfold = KFold(n_splits=args.n_folds, shuffle=True, random_state=args.seed)
    summary_rows = []
    pair_indices = np.arange(len(pair_df))
    requested_folds = set(args.folds)
    for fold_idx, (train_valid_idx, test_idx) in enumerate(pair_kfold.split(pair_indices), start=1):
        if fold_idx not in requested_folds:
            continue
        train_valid_pairs = pair_df.iloc[train_valid_idx].copy()
        test_pairs = pair_df.iloc[test_idx].copy()
        stratify_valid = maybe_stratify(train_valid_pairs["target_bin"])
        train_pairs, valid_pairs = train_test_split(
            train_valid_pairs,
            test_size=0.125,
            random_state=args.seed + fold_idx,
            shuffle=True,
            stratify=stratify_valid,
        )
        split_pairs = {
            "train": set(train_pairs["pair_key"]),
            "valid": set(valid_pairs["pair_key"]),
            "test": set(test_pairs["pair_key"]),
        }
        for split_name, pair_keys in split_pairs.items():
            split_df = work_df[work_df["pair_key"].isin(pair_keys)].drop(columns=["pair_key"]).reset_index(drop=True)
            split_path = folds_dir / f"{args.fold_prefix}_fold{fold_idx:02d}_{split_name}.csv"
            split_df.to_csv(split_path, index=False)
            write_split_summary(summary_rows, args.fold_prefix, fold_idx, split_name, split_df, split_path)

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(folds_dir / f"{args.fold_prefix}_fold_summary.csv", index=False)
    return folds_dir


def metric_artifacts(run_dir: Path, train_path: Path, test_path: Path, model_name: str) -> tuple[Path, Path]:
    artifact_prefix = f"{train_path.stem}_{model_name}"
    metrics_path = run_dir / f"{artifact_prefix}_{test_path.stem}_metrics.json"
    train_csv = run_dir / f"{artifact_prefix}_training.csv"
    return metrics_path, train_csv


def run_one(mask_name: str, fold: int, args: argparse.Namespace) -> dict:
    folds_dir = resolve_path(args.folds_dir)
    output_dir = resolve_path(args.output_dir)
    cache_dir = resolve_path(args.cache_dir)
    pair_cache_dir = resolve_path(args.pair_cache_dir)
    train_path, valid_path, test_path = fold_paths(folds_dir, args.fold_prefix, fold)
    model_name = "PGSSI_InteractionTypeAblation"
    run_dir = output_dir / mask_name / f"fold{fold:02d}"
    run_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)
    pair_cache_dir.mkdir(parents=True, exist_ok=True)
    metrics_path, train_csv = metric_artifacts(run_dir, train_path, test_path, model_name)

    if args.force or not (metrics_path.exists() and train_csv.exists()):
        cmd = [
            args.python_executable,
            str(TRAIN_SCRIPT),
            "--run-dir",
            str(run_dir),
            "--cache-dir",
            str(cache_dir),
            "--pair-cache-dir",
            str(pair_cache_dir),
            "--train-path",
            str(train_path),
            "--valid-path",
            str(valid_path),
            "--test-path",
            str(test_path),
            "--model-name",
            model_name,
            "--cache-model-name",
            f"PGSSI_interaction_type_cv_{args.fold_prefix}_fold{fold:02d}",
            "--interaction-mask-name",
            mask_name,
            "--n-epochs",
            str(args.epochs),
            "--hidden-dim",
            str(args.hidden_dim),
            "--batch-size",
            str(args.batch_size),
            "--train-num-workers",
            str(args.train_num_workers),
            "--valid-num-workers",
            str(args.valid_num_workers),
            "--checkpoint-interval",
            str(args.checkpoint_interval),
            "--early-stopping-patience",
            str(args.early_stopping_patience),
            "--quiet-progress",
            "--seed",
            str(args.seed + fold - 1),
        ]
        if args.resume:
            cmd.append("--resume")
        print(f"\n=== Running {MASK_LABELS[mask_name]} | fold {fold} ===")
        print(" ".join(cmd))
        subprocess.run(cmd, check=True, cwd=PROJECT_ROOT)
    else:
        print(f"\n=== Reusing {MASK_LABELS[mask_name]} | fold {fold} ===")

    with open(metrics_path, "r", encoding="utf-8") as fh:
        metrics = json.load(fh)
    train_df = pd.read_csv(train_csv)
    best_idx = train_df["MAE_Valid"].idxmin()
    return {
        "mask_name": mask_name,
        "experiment": MASK_LABELS[mask_name],
        "masked_features": MASKED_FEATURES[mask_name],
        "fold": int(fold),
        "epochs_ran": int(len(train_df)),
        "best_valid_mae": float(train_df.loc[best_idx, "MAE_Valid"]),
        "best_valid_r2": float(train_df.loc[best_idx, "R2_Valid"]),
        "test_mae": float(metrics.get("mae", float("nan"))),
        "test_rmse": float(metrics.get("rmse", float("nan"))),
        "test_r2": float(metrics.get("r2", float("nan"))),
        "run_dir": str(run_dir),
    }


def fmt_mean_std(mean_value: float, std_value: float) -> str:
    return f"{mean_value:.4f} $\\pm$ {std_value:.4f}"


def build_summary(fold_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for mask_name, group in fold_df.groupby("mask_name", sort=False):
        std_ddof = 1 if len(group) > 1 else 0
        row = {
            "mask_name": mask_name,
            "experiment": MASK_LABELS[mask_name],
            "masked_features": MASKED_FEATURES[mask_name],
            "folds": int(group["fold"].nunique()),
            "mae_mean": float(group["test_mae"].mean()),
            "mae_std": float(group["test_mae"].std(ddof=std_ddof)),
            "rmse_mean": float(group["test_rmse"].mean()),
            "rmse_std": float(group["test_rmse"].std(ddof=std_ddof)),
            "r2_mean": float(group["test_r2"].mean()),
            "r2_std": float(group["test_r2"].std(ddof=std_ddof)),
            "epochs_mean": float(group["epochs_ran"].mean()),
        }
        row["mae_mean_std"] = fmt_mean_std(row["mae_mean"], row["mae_std"])
        row["rmse_mean_std"] = fmt_mean_std(row["rmse_mean"], row["rmse_std"])
        row["r2_mean_std"] = fmt_mean_std(row["r2_mean"], row["r2_std"])
        rows.append(row)
    return pd.DataFrame(rows)


def write_latex_table(summary_df: pd.DataFrame, path: Path) -> None:
    table_df = summary_df[
        ["experiment", "masked_features", "mae_mean_std", "rmse_mean_std", "r2_mean_std"]
    ].rename(
        columns={
            "experiment": "Setting",
            "masked_features": "Masked feature(s)",
            "mae_mean_std": "MAE",
            "rmse_mean_std": "RMSE",
            "r2_mean_std": "$R^2$",
        }
    )
    latex = table_df.to_latex(
        index=False,
        escape=False,
        column_format="llccc",
        caption="Five-fold cross-validation results for PGSSI interaction-type ablations.",
        label="tab:pgssi_interaction_type_ablation",
    )
    path.write_text(latex, encoding="utf-8")


def write_per_fold_latex_table(fold_df: pd.DataFrame, path: Path) -> None:
    table_df = fold_df[
        ["experiment", "fold", "test_mae", "test_rmse", "test_r2"]
    ].rename(
        columns={
            "experiment": "Setting",
            "fold": "Fold",
            "test_mae": "MAE",
            "test_rmse": "RMSE",
            "test_r2": "$R^2$",
        }
    )
    latex = table_df.to_latex(
        index=False,
        escape=False,
        column_format="lrrrr",
        float_format=lambda value: f"{value:.4f}",
        caption="Per-fold test performance for PGSSI interaction-type ablations.",
        label="tab:pgssi_interaction_type_ablation_per_fold",
    )
    path.write_text(latex, encoding="utf-8")


def write_conclusion(summary_df: pd.DataFrame, path: Path, total_runs: int) -> None:
    if "full" not in set(summary_df["mask_name"]):
        text = "\n".join(
            [
                "# Interaction-Type Ablation Conclusion",
                "",
                f"- Total training runs: {total_runs}.",
                "- The `full` baseline was not included in this run, so baseline-relative deltas were not computed.",
                "- Rerun with `--interaction-mask-names full ...` to generate the final paper conclusion.",
            ]
        )
        path.write_text(text + "\n", encoding="utf-8")
        return

    full = summary_df.loc[summary_df["mask_name"] == "full"].iloc[0]
    ablated = summary_df.loc[summary_df["mask_name"] != "full"].copy()
    if ablated.empty:
        text = "\n".join(
            [
                "# Interaction-Type Ablation Conclusion",
                "",
                f"- Total training runs: {total_runs}.",
                f"- Full PGSSI obtains MAE {full['mae_mean_std']}, RMSE {full['rmse_mean_std']}, R2 {full['r2_mean_std']}.",
                "- No ablated setting was included in this run.",
            ]
        )
        path.write_text(text + "\n", encoding="utf-8")
        return

    ablated["delta_mae"] = ablated["mae_mean"] - float(full["mae_mean"])
    ablated["delta_r2"] = ablated["r2_mean"] - float(full["r2_mean"])
    strongest = ablated.sort_values("delta_mae", ascending=False).iloc[0]
    weakest = ablated.sort_values("delta_mae", ascending=True).iloc[0]
    text = "\n".join(
        [
            "# Interaction-Type Ablation Conclusion",
            "",
            f"- Total training runs: {total_runs} ({len(summary_df)} settings x {int(full['folds'])} folds).",
            f"- Full PGSSI obtains MAE {full['mae_mean_std']}, RMSE {full['rmse_mean_std']}, R2 {full['r2_mean_std']}.",
            f"- The largest MAE increase is observed for `{strongest['mask_name']}` "
            f"(Delta MAE = {strongest['delta_mae']:.4f}, Delta R2 = {strongest['delta_r2']:.4f}), "
            "indicating this interaction group contributes most strongly among the tested masks.",
            f"- The smallest MAE change is observed for `{weakest['mask_name']}` "
            f"(Delta MAE = {weakest['delta_mae']:.4f}, Delta R2 = {weakest['delta_r2']:.4f}).",
            "",
            "Interpret the deltas after all folds finish; if a setting has fewer than five folds, rerun with the same command to complete missing folds.",
        ]
    )
    path.write_text(text + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    output_dir = resolve_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    ensure_interaction_type_folds(args)

    rows = []
    for mask_name in args.interaction_mask_names:
        for fold in args.folds:
            rows.append(run_one(mask_name, fold, args))

    fold_df = pd.DataFrame(rows)
    summary_df = build_summary(fold_df)

    fold_csv = output_dir / "interaction_type_ablation_5fold_per_fold.csv"
    summary_csv = output_dir / "interaction_type_ablation_5fold_summary.csv"
    summary_md = output_dir / "interaction_type_ablation_5fold_summary.md"
    latex_path = output_dir / "interaction_type_ablation_5fold_table.tex"
    per_fold_latex_path = output_dir / "interaction_type_ablation_5fold_per_fold_table.tex"
    conclusion_path = output_dir / "interaction_type_ablation_conclusion.md"
    run_summary_path = output_dir / "interaction_type_ablation_run_summary.json"

    fold_df.to_csv(fold_csv, index=False, float_format="%.6f")
    summary_df.to_csv(summary_csv, index=False, float_format="%.6f")
    summary_md_text = summary_df[
        ["experiment", "masked_features", "folds", "mae_mean_std", "rmse_mean_std", "r2_mean_std", "epochs_mean"]
    ].to_markdown(index=False)
    summary_md.write_text(summary_md_text + "\n", encoding="utf-8")
    write_latex_table(summary_df, latex_path)
    write_per_fold_latex_table(fold_df, per_fold_latex_path)
    write_conclusion(summary_df, conclusion_path, total_runs=len(fold_df))

    run_summary = {
        "num_settings": int(len(args.interaction_mask_names)),
        "num_folds": int(len(args.folds)),
        "num_training_runs": int(len(fold_df)),
        "epochs_requested_per_run": int(args.epochs),
        "outputs": {
            "per_fold_csv": str(fold_csv),
            "summary_csv": str(summary_csv),
            "summary_md": str(summary_md),
            "latex_table": str(latex_path),
            "per_fold_latex_table": str(per_fold_latex_path),
            "conclusion_md": str(conclusion_path),
        },
    }
    with open(run_summary_path, "w", encoding="utf-8") as fh:
        json.dump(run_summary, fh, indent=2, ensure_ascii=False)

    print(f"\nSaved per-fold metrics to: {fold_csv}")
    print(f"Saved 5-fold summary to: {summary_csv}")
    print(f"Saved LaTeX table to: {latex_path}")
    print(f"Saved per-fold LaTeX table to: {per_fold_latex_path}")
    print(summary_df[["experiment", "mae_mean_std", "rmse_mean_std", "r2_mean_std"]].to_string(index=False))


if __name__ == "__main__":
    main()
