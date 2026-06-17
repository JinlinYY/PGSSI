from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TRAIN = PROJECT_ROOT / "dataset" / "all" / "all_merged_train.csv"
DEFAULT_VALID = PROJECT_ROOT / "dataset" / "all" / "all_merged_valid.csv"
DEFAULT_TEST = PROJECT_ROOT / "dataset" / "all" / "all_merged_test.csv"
DEFAULT_OUTPUT = PROJECT_ROOT / "runs" / "pgssi_reference_ablation"
DEFAULT_CACHE = PROJECT_ROOT / "cache" / "pgssi_reference_ablation"
TRAIN_SCRIPT = PROJECT_ROOT / "src" / "models" / "PGSSI" / "PGSSI_train.py"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run PGSSI reference-information ablations for reviewer Comment 5."
    )
    parser.add_argument("--train-path", type=str, default=str(DEFAULT_TRAIN))
    parser.add_argument("--valid-path", type=str, default=str(DEFAULT_VALID))
    parser.add_argument("--test-path", type=str, default=str(DEFAULT_TEST))
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT))
    parser.add_argument("--cache-dir", type=str, default=str(DEFAULT_CACHE))
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--early-stopping-patience", type=int, default=50)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--train-num-workers", type=int, default=0)
    parser.add_argument("--valid-num-workers", type=int, default=0)
    parser.add_argument("--checkpoint-interval", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def experiment_grid():
    return [
        {
            "name": "full_model",
            "label": "Full PGSSI",
            "flags": [],
            "description": "Full system readout with solvent, solute, and pair contrast information.",
        },
        {
            "name": "no_solute_readout",
            "label": "No Solute Readout",
            "flags": ["--disable-solute-readout"],
            "description": "Zero solute standalone pooled graph features and solute RDKit descriptors in the final readout; explicit solute-solvent cross contacts are retained.",
        },
        {
            "name": "no_pair_readout",
            "label": "No Pair Contrast Readout",
            "flags": ["--disable-pair-readout"],
            "description": "Zero elementwise product and absolute-difference readout terms between solvent and solute embeddings.",
        },
        {
            "name": "no_solute_reference_and_pair_readout",
            "label": "No Solute Reference + Pair Contrast",
            "flags": ["--disable-solute-readout", "--disable-pair-readout"],
            "description": "Remove solute standalone reference information and solvent-solute readout contrast while retaining explicit cross-contact message passing.",
        },
        {
            "name": "no_solvent_readout_control",
            "label": "No Solvent Readout Control",
            "flags": ["--disable-solvent-readout"],
            "description": "Symmetry control that zeros solvent standalone pooled graph features and solvent RDKit descriptors.",
        },
    ]


def _artifact_prefix(train_path: str, model_name: str) -> str:
    return f"{Path(train_path).stem}_{model_name}"


def _metrics_path(run_dir: Path, train_path: str, test_path: str, model_name: str) -> Path:
    return run_dir / f"{_artifact_prefix(train_path, model_name)}_{Path(test_path).stem}_metrics.json"


def _training_path(run_dir: Path, train_path: str, model_name: str) -> Path:
    return run_dir / f"{_artifact_prefix(train_path, model_name)}_training.csv"


def run_one(exp: dict, args, output_dir: Path, cache_dir: Path) -> dict:
    model_name = "PGSSI_ReferenceAblation"
    run_dir = output_dir / exp["name"]
    run_cache_dir = cache_dir
    run_dir.mkdir(parents=True, exist_ok=True)
    run_cache_dir.mkdir(parents=True, exist_ok=True)

    metrics_path = _metrics_path(run_dir, args.train_path, args.test_path, model_name)
    train_csv = _training_path(run_dir, args.train_path, model_name)

    if not (metrics_path.exists() and train_csv.exists()):
        cmd = [
            sys.executable,
            str(TRAIN_SCRIPT),
            "--run-dir",
            str(run_dir),
            "--cache-dir",
            str(run_cache_dir),
            "--train-path",
            str(args.train_path),
            "--valid-path",
            str(args.valid_path),
            "--test-path",
            str(args.test_path),
            "--model-name",
            model_name,
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
            str(args.seed),
        ] + exp["flags"]

        print(f"\n=== Running {exp['label']} ===")
        print(" ".join(cmd))
        subprocess.run(cmd, check=True, cwd=PROJECT_ROOT)
    else:
        print(f"\n=== Reusing existing result for {exp['label']} ===")

    with open(metrics_path, "r", encoding="utf-8") as fh:
        metrics = json.load(fh)

    train_df = pd.read_csv(train_csv)
    best_valid_mae = float(train_df["MAE_Valid"].min())
    best_valid_r2 = float(train_df.loc[train_df["MAE_Valid"].idxmin(), "R2_Valid"])

    return {
        "experiment": exp["label"],
        "name": exp["name"],
        "description": exp["description"],
        "epochs_requested": int(args.epochs),
        "early_stopping_patience": int(args.early_stopping_patience),
        "epochs_ran": int(len(train_df)),
        "best_valid_mae": best_valid_mae,
        "best_valid_r2": best_valid_r2,
        "test_num_predicted": int(metrics.get("num_samples_predicted", 0)),
        "test_num_skipped": int(metrics.get("num_samples_skipped", 0)),
        "test_mae": float(metrics.get("mae", float("nan"))),
        "test_rmse": float(metrics.get("rmse", float("nan"))),
        "test_r2": float(metrics.get("r2", float("nan"))),
        "test_ae_le_01_pct": float(metrics.get("AE<=0.1", float("nan"))),
        "test_ae_le_02_pct": float(metrics.get("AE<=0.2", float("nan"))),
        "test_ae_le_03_pct": float(metrics.get("AE<=0.3", float("nan"))),
        "run_dir": str(run_dir),
    }


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    cache_dir = Path(args.cache_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    rows = [run_one(exp, args, output_dir, cache_dir) for exp in experiment_grid()]
    result_df = pd.DataFrame(rows)
    full_mae = result_df.loc[result_df["name"] == "full_model", "test_mae"]
    full_r2 = result_df.loc[result_df["name"] == "full_model", "test_r2"]
    if not full_mae.empty:
        result_df["delta_mae_vs_full"] = result_df["test_mae"] - float(full_mae.iloc[0])
    if not full_r2.empty:
        result_df["delta_r2_vs_full"] = result_df["test_r2"] - float(full_r2.iloc[0])

    float_columns = result_df.select_dtypes(include=["float32", "float64"]).columns
    result_df[float_columns] = result_df[float_columns].round(4)
    csv_path = output_dir / "reference_ablation_results.csv"
    md_path = output_dir / "reference_ablation_results.md"
    result_df.to_csv(csv_path, index=False, float_format="%.4f")
    result_df.to_markdown(md_path, index=False)

    print(f"\nSaved reference ablation table to: {csv_path}")
    print(result_df.to_string(index=False))


if __name__ == "__main__":
    main()
