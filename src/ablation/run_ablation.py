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
DEFAULT_OUTPUT = Path(__file__).resolve().parent / "outputs_full_200ep"
TRAIN_SCRIPT = PROJECT_ROOT / "src" / "models" / "PGSSI" / "PGSSI_train.py"


def parse_args():
    parser = argparse.ArgumentParser(description="Run PGSSI ablation experiments.")
    parser.add_argument("--train-path", type=str, default=str(DEFAULT_TRAIN))
    parser.add_argument("--valid-path", type=str, default=str(DEFAULT_VALID))
    parser.add_argument("--test-path", type=str, default=str(DEFAULT_TEST))
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT))
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--early-stopping-patience", type=int, default=50)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--checkpoint-interval", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def experiment_grid():
    return [
        {
            "name": "full_model",
            "label": "Full PGSSI",
            "flags": [],
            "description": "EGNN + typed cross interaction + MoE gate",
        },
        {
            "name": "no_equivariant_intra",
            "label": "No EGNN Intra",
            "flags": ["--num-intra-layers", "0"],
            "description": "Remove equivariant intra-molecular message passing",
        },
        {
            "name": "no_cross_interaction",
            "label": "No Cross Interaction",
            "flags": ["--disable-cross-interaction"],
            "description": "Remove explicit solute-solvent cross interaction",
        },
        {
            "name": "no_interaction_types",
            "label": "No Interaction Typing",
            "flags": ["--disable-interaction-types"],
            "description": "Keep cross edges but remove donor/acceptor, aromatic, dipole, polarity typing",
        },
        {
            "name": "no_moe_gate",
            "label": "No MoE Gate",
            "flags": ["--disable-moe"],
            "description": "Replace mixture-of-experts cross interaction with a single expert",
        },
        {
            "name": "no_physics_prior",
            "label": "No Physics Prior",
            "flags": ["--disable-physics-prior"],
            "description": "Remove LJ/Coulomb-inspired physics priors from cross interaction",
        },
        {
            "name": "no_formula_layer",
            "label": "No Formula Layer",
            "flags": ["--disable-formula-layer"],
            "description": "Predict K1/K2 but remove the K1 + K2 / T_K formula layer",
        },
        {
            "name": "direct_loggamma_head",
            "label": "Direct log-gamma Head",
            "flags": ["--direct-loggamma-head"],
            "description": "Predict log-gamma directly instead of using the parametric K1/K2 head",
        },
        {
            "name": "topology_only",
            "label": "2D Topology Variant",
            "flags": ["--topology-only"],
            "description": "Disable effective 3D geometry updates and use topology-only message passing",
        },
        {
            "name": "no_cross_refine",
            "label": "No Cross Refine",
            "flags": ["--disable-cross-refine"],
            "description": "Keep a single cross interaction layer and remove the refine layer",
        },
    ]


def run_one(exp: dict, args, output_dir: Path) -> dict:
    run_dir = output_dir / exp["name"]
    run_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = run_dir / "all_merged_train_PGSSI_Ablation_all_merged_test_metrics.json"
    train_csv = run_dir / "all_merged_train_PGSSI_Ablation_training.csv"

    if not (metrics_path.exists() and train_csv.exists()):
        cmd = [
            sys.executable,
            str(TRAIN_SCRIPT),
            "--run-dir",
            str(run_dir),
            "--train-path",
            str(args.train_path),
            "--valid-path",
            str(args.valid_path),
            "--test-path",
            str(args.test_path),
            "--model-name",
            "PGSSI_Ablation",
            "--n-epochs",
            str(args.epochs),
            "--hidden-dim",
            str(args.hidden_dim),
            "--batch-size",
            str(args.batch_size),
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
        "test_mae": float(metrics.get("mae", float("nan"))),
        "test_rmse": float(metrics.get("rmse", float("nan"))),
        "test_r2": float(metrics.get("r2", float("nan"))),
        "test_ae_le_01_pct": float(metrics.get("AE<=0.1", float("nan"))),
        "test_ae_le_02_pct": float(metrics.get("AE<=0.2", float("nan"))),
        "test_ae_le_03_pct": float(metrics.get("AE<=0.3", float("nan"))),
        "physics_reg": float(metrics.get("physics_reg", float("nan"))),
        "physics_smoothness": float(metrics.get("physics_smoothness", float("nan"))),
        "physics_short_range_repulsion": float(metrics.get("physics_short_range_repulsion", float("nan"))),
        "physics_long_range_decay": float(metrics.get("physics_long_range_decay", float("nan"))),
        "physics_charge_sign_consistency": float(metrics.get("physics_charge_sign_consistency", float("nan"))),
        "physics_thermo_derivative": float(metrics.get("physics_thermo_derivative", float("nan"))),
        "run_dir": str(run_dir),
    }


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for exp in experiment_grid():
        rows.append(run_one(exp, args, output_dir))

    result_df = pd.DataFrame(rows).sort_values(["test_r2", "best_valid_r2"], ascending=False)
    csv_path = output_dir / "ablation_results.csv"
    md_path = output_dir / "ablation_results.md"
    result_df.to_csv(csv_path, index=False)
    result_df.to_markdown(md_path, index=False)

    print(f"\nSaved ablation table to: {csv_path}")
    print(result_df.to_string(index=False))


if __name__ == "__main__":
    main()
