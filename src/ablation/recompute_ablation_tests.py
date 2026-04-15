from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.PGSSI.PGSSI_train import evaluate_test_file

DEFAULT_OUTPUT = Path(__file__).resolve().parent / "outputs_full_200ep"
DEFAULT_TEST = PROJECT_ROOT / "dataset" / "all" / "all_merged_test.csv"
DEFAULT_CACHE = PROJECT_ROOT / "cache" / "ablation_test_recompute"


def parse_args():
    parser = argparse.ArgumentParser(description="Recompute full-data ablation test metrics with fresh caches.")
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT))
    parser.add_argument("--test-path", type=str, default=str(DEFAULT_TEST))
    parser.add_argument("--cache-dir", type=str, default=str(DEFAULT_CACHE))
    parser.add_argument("--hidden-dim", type=int, default=512)
    return parser.parse_args()


def experiment_grid():
    return [
        {
            "name": "full_model",
            "label": "Full PGSSI",
            "description": "EGNN + typed cross interaction + MoE gate",
            "enable_cross_interaction": True,
            "num_intra_layers": 2,
            "use_interaction_types": True,
            "use_moe": True,
            "use_physics_prior": True,
            "disable_cross_refine": False,
            "topology_only": False,
            "direct_loggamma_head": False,
            "disable_formula_layer": False,
        },
        {
            "name": "no_equivariant_intra",
            "label": "No EGNN Intra",
            "description": "Remove equivariant intra-molecular message passing",
            "enable_cross_interaction": True,
            "num_intra_layers": 0,
            "use_interaction_types": True,
            "use_moe": True,
            "use_physics_prior": True,
            "disable_cross_refine": False,
            "topology_only": False,
            "direct_loggamma_head": False,
            "disable_formula_layer": False,
        },
        {
            "name": "no_cross_interaction",
            "label": "No Cross Interaction",
            "description": "Remove explicit solute-solvent cross interaction",
            "enable_cross_interaction": False,
            "num_intra_layers": 2,
            "use_interaction_types": True,
            "use_moe": True,
            "use_physics_prior": True,
            "disable_cross_refine": False,
            "topology_only": False,
            "direct_loggamma_head": False,
            "disable_formula_layer": False,
        },
        {
            "name": "no_interaction_types",
            "label": "No Interaction Typing",
            "description": "Keep cross edges but remove donor/acceptor, aromatic, dipole, polarity typing",
            "enable_cross_interaction": True,
            "num_intra_layers": 2,
            "use_interaction_types": False,
            "use_moe": True,
            "use_physics_prior": True,
            "disable_cross_refine": False,
            "topology_only": False,
            "direct_loggamma_head": False,
            "disable_formula_layer": False,
        },
        {
            "name": "no_moe_gate",
            "label": "No MoE Gate",
            "description": "Replace mixture-of-experts cross interaction with a single expert",
            "enable_cross_interaction": True,
            "num_intra_layers": 2,
            "use_interaction_types": True,
            "use_moe": False,
            "use_physics_prior": True,
            "disable_cross_refine": False,
            "topology_only": False,
            "direct_loggamma_head": False,
            "disable_formula_layer": False,
        },
        {
            "name": "no_physics_prior",
            "label": "No Physics Prior",
            "description": "Remove LJ/Coulomb-inspired physics priors from cross interaction",
            "enable_cross_interaction": True,
            "num_intra_layers": 2,
            "use_interaction_types": True,
            "use_moe": True,
            "use_physics_prior": False,
            "disable_cross_refine": False,
            "topology_only": False,
            "direct_loggamma_head": False,
            "disable_formula_layer": False,
        },
        {
            "name": "no_formula_layer",
            "label": "No Formula Layer",
            "description": "Predict K1/K2 but remove the K1 + K2 / T_K formula layer",
            "enable_cross_interaction": True,
            "num_intra_layers": 2,
            "use_interaction_types": True,
            "use_moe": True,
            "use_physics_prior": True,
            "disable_cross_refine": False,
            "topology_only": False,
            "direct_loggamma_head": False,
            "disable_formula_layer": True,
        },
        {
            "name": "direct_loggamma_head",
            "label": "Direct log-gamma Head",
            "description": "Predict log-gamma directly instead of using the parametric K1/K2 head",
            "enable_cross_interaction": True,
            "num_intra_layers": 2,
            "use_interaction_types": True,
            "use_moe": True,
            "use_physics_prior": True,
            "disable_cross_refine": False,
            "topology_only": False,
            "direct_loggamma_head": True,
            "disable_formula_layer": False,
        },
        {
            "name": "topology_only",
            "label": "2D Topology Variant",
            "description": "Disable effective 3D geometry updates and use topology-only message passing",
            "enable_cross_interaction": True,
            "num_intra_layers": 2,
            "use_interaction_types": True,
            "use_moe": True,
            "use_physics_prior": True,
            "disable_cross_refine": False,
            "topology_only": True,
            "direct_loggamma_head": False,
            "disable_formula_layer": False,
        },
        {
            "name": "no_cross_refine",
            "label": "No Cross Refine",
            "description": "Keep a single cross interaction layer and remove the refine layer",
            "enable_cross_interaction": True,
            "num_intra_layers": 2,
            "use_interaction_types": True,
            "use_moe": True,
            "use_physics_prior": True,
            "disable_cross_refine": True,
            "topology_only": False,
            "direct_loggamma_head": False,
            "disable_formula_layer": False,
        },
    ]


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    test_path = str(Path(args.test_path))
    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for exp in experiment_grid():
        run_dir = output_dir / exp["name"]
        train_csv = run_dir / "all_merged_train_PGSSI_Ablation_training.csv"
        best_model_path = run_dir / "all_merged_train_PGSSI_Ablation_best.pth"
        if not (train_csv.exists() and best_model_path.exists()):
            print(f"Skipping {exp['label']} because training artifacts are missing.")
            continue

        train_df = pd.read_csv(train_csv)
        best_valid_mae = float(train_df["MAE_Valid"].min())
        best_valid_r2 = float(train_df.loc[train_df["MAE_Valid"].idxmin(), "R2_Valid"])

        metrics = evaluate_test_file(
            model_path=str(best_model_path),
            test_path=test_path,
            test_name="all_merged_test_recomputed",
            model_name="PGSSI_Ablation_Recomputed",
            artifact_prefix="all_merged_train_PGSSI_Ablation",
            hidden_dim=args.hidden_dim,
            enable_cross_interaction=exp["enable_cross_interaction"],
            num_intra_layers=exp["num_intra_layers"],
            use_interaction_types=exp["use_interaction_types"],
            use_moe=exp["use_moe"],
            use_physics_prior=exp["use_physics_prior"],
            disable_cross_refine=exp["disable_cross_refine"],
            topology_only=exp["topology_only"],
            direct_loggamma_head=exp["direct_loggamma_head"],
            disable_formula_layer=exp["disable_formula_layer"],
            output_dir=str(run_dir),
            cache_dir=str(cache_dir),
        )

        rows.append(
            {
                "experiment": exp["label"],
                "name": exp["name"],
                "description": exp["description"],
                "epochs_requested": 200,
                "early_stopping_patience": 50,
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
                "physics_reg": float(metrics.get("physics_reg", float("nan"))),
                "physics_smoothness": float(metrics.get("physics_smoothness", float("nan"))),
                "physics_short_range_repulsion": float(metrics.get("physics_short_range_repulsion", float("nan"))),
                "physics_long_range_decay": float(metrics.get("physics_long_range_decay", float("nan"))),
                "physics_charge_sign_consistency": float(metrics.get("physics_charge_sign_consistency", float("nan"))),
                "physics_thermo_derivative": float(metrics.get("physics_thermo_derivative", float("nan"))),
                "run_dir": str(run_dir),
            }
        )

    result_df = pd.DataFrame(rows).sort_values(["test_r2", "best_valid_r2"], ascending=False)
    float_columns = result_df.select_dtypes(include=["float32", "float64"]).columns
    result_df[float_columns] = result_df[float_columns].round(4)
    corrected_csv = output_dir / "ablation_results_corrected.csv"
    corrected_md = output_dir / "ablation_results_corrected.md"
    result_df.to_csv(corrected_csv, index=False, float_format="%.4f")
    result_df.to_markdown(corrected_md, index=False)

    current_csv = output_dir / "ablation_results.csv"
    current_md = output_dir / "ablation_results.md"
    result_df.to_csv(current_csv, index=False, float_format="%.4f")
    result_df.to_markdown(current_md, index=False)

    summary = {
        "corrected_csv": str(corrected_csv),
        "corrected_md": str(corrected_md),
        "cache_dir": str(cache_dir),
        "num_experiments": int(len(result_df)),
    }
    with open(output_dir / "ablation_results_corrected_summary.json", "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, ensure_ascii=False)

    print(f"Saved corrected ablation table to: {corrected_csv}")
    print(result_df.to_string(index=False))


if __name__ == "__main__":
    main()
