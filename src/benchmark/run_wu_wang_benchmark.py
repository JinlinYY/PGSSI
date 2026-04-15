from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from src.models.PGSSI.PGSSI_train import (
        configure_torch_runtime,
        dataset_cache_prefix,
        evaluate_test_file,
        make_cache_dir,
        make_run_dir,
        print_test_summary,
        resolve_input_path,
        set_seed,
        train_PGSSI,
    )
except ImportError:
    from models.PGSSI.PGSSI_train import (
        configure_torch_runtime,
        dataset_cache_prefix,
        evaluate_test_file,
        make_cache_dir,
        make_run_dir,
        print_test_summary,
        resolve_input_path,
        set_seed,
        train_PGSSI,
    )
DEFAULT_INIT = PROJECT_ROOT / "runs" / "pgssi_train_all_e3typed" / "all_merged_train_PGSSI_E3Typed_best.pth"


WU_BASELINES = [
    {"Model": "SolvGNNCat", "MAE": 0.09, "R2": 0.97, "AE<=0.1": 78.96, "AE<=0.2": 91.03, "AE<=0.3": 95.02},
    {"Model": "SolvGNGNH", "MAE": 0.13, "R2": 0.96, "AE<=0.1": 65.19, "AE<=0.2": 84.45, "AE<=0.3": 91.39},
    {"Model": "TCM", "MAE": 0.09, "R2": 0.89, "AE<=0.1": 78.46, "AE<=0.2": 91.13, "AE<=0.3": 94.50},
    {"Model": "MPNN-cat-GH", "MAE": 0.11, "R2": 0.96, "AE<=0.1": 76.43, "AE<=0.2": 91.22, "AE<=0.3": 95.21},
    {"Model": "NCF", "MAE": 0.16, "R2": 0.96, "AE<=0.1": 60.05, "AE<=0.2": 81.64, "AE<=0.3": 89.53},
    {"Model": "HANNA(legacy)", "MAE": 0.19, "R2": 0.86, "AE<=0.1": 61.49, "AE<=0.2": 81.84, "AE<=0.3": 88.20},
    {"Model": "HANNA", "MAE": 0.20, "R2": 0.89, "AE<=0.1": 53.38, "AE<=0.2": 76.71, "AE<=0.3": 85.97},
    {"Model": "GHGNN(wo)", "MAE": 0.10, "R2": 0.97, "AE<=0.1": 75.00, "AE<=0.2": 89.04, "AE<=0.3": 94.05},
    {"Model": "GHGNN", "MAE": 0.10, "R2": 0.96, "AE<=0.1": 76.51, "AE<=0.2": 86.51, "AE<=0.3": 94.00},
    {"Model": "GHGEAT(wo)", "MAE": 0.09, "R2": 0.96, "AE<=0.1": 74.07, "AE<=0.2": 89.85, "AE<=0.3": 93.42},
    {"Model": "GHGEAT", "MAE": 0.07, "R2": 0.97, "AE<=0.1": 81.92, "AE<=0.2": 92.39, "AE<=0.3": 95.73},
    {"Model": "GEATCat", "MAE": 0.27, "R2": 0.76, "AE<=0.1": 26.90, "AE<=0.2": 50.97, "AE<=0.3": 69.29},
]


IDAC2026_BASELINES = [
    {"Model": "SolvGNNCat", "MAE": 0.53, "R2": 0.5862, "AE<=0.1": 56.59, "AE<=0.2": 73.12, "AE<=0.3": 79.31},
    {"Model": "SolvGNGNH", "MAE": 0.67, "R2": 0.34, "AE<=0.1": 57.31, "AE<=0.2": 72.67, "AE<=0.3": 77.74},
    {"Model": "TCM", "MAE": 1.46, "R2": 0.50, "AE<=0.1": 11.85, "AE<=0.2": 20.49, "AE<=0.3": 27.40},
    {"Model": "MPNN-cat-GH", "MAE": 1.33, "R2": 0.17, "AE<=0.1": 6.41, "AE<=0.2": 12.40, "AE<=0.3": 18.23},
    {"Model": "NCF", "MAE": 1.40, "R2": 0.15, "AE<=0.1": 11.85, "AE<=0.2": 20.81, "AE<=0.3": 28.68},
    {"Model": "HANNA(legacy)", "MAE": 0.74, "R2": 0.47, "AE<=0.1": 22.13, "AE<=0.2": 40.19, "AE<=0.3": 53.72},
    {"Model": "HANNA", "MAE": 0.77, "R2": 0.33, "AE<=0.1": 26.87, "AE<=0.2": 46.33, "AE<=0.3": 59.08},
    {"Model": "GHGNN(wo)", "MAE": 0.59, "R2": 0.5532, "AE<=0.1": 65.90, "AE<=0.2": 76.46, "AE<=0.3": 80.41},
    {"Model": "GHGNN", "MAE": 0.48, "R2": 0.6317, "AE<=0.1": 66.65, "AE<=0.2": 77.03, "AE<=0.3": 80.72},
    {"Model": "GHGEAT(wo)", "MAE": 0.56, "R2": 0.5574, "AE<=0.1": 58.60, "AE<=0.2": 72.35, "AE<=0.3": 76.83},
    {"Model": "GHGEAT", "MAE": 0.34, "R2": 0.9343, "AE<=0.1": 16.72, "AE<=0.2": 31.17, "AE<=0.3": 42.54},
    {"Model": "GEATCat", "MAE": 0.82, "R2": 0.23, "AE<=0.1": 27.25, "AE<=0.2": 46.41, "AE<=0.3": 59.23},
]


def parse_args():
    parser = argparse.ArgumentParser(description="Train PGSSI on Wu2004 and test on Wu2004/IDAC2026.")
    parser.add_argument("--run-dir", type=str, default=str(PROJECT_ROOT / "runs" / "pgssi_wu_benchmark"))
    parser.add_argument("--cache-dir", type=str, default=str(PROJECT_ROOT / "cache" / "pgssi_wu_benchmark"))
    parser.add_argument("--train-path", type=str, default=str(PROJECT_ROOT / "dataset" / "wu_et_al" / "molecule_train.csv"))
    parser.add_argument("--valid-path", type=str, default=str(PROJECT_ROOT / "dataset" / "wu_et_al" / "molecule_valid.csv"))
    parser.add_argument("--wu-test-path", type=str, default=str(PROJECT_ROOT / "dataset" / "wu_et_al" / "molecule_test.csv"))
    parser.add_argument("--idac-test-path", type=str, default=str(PROJECT_ROOT / "dataset" / "wang_et_al" / "IDAC_2026_dataset.csv"))
    parser.add_argument("--model-name", type=str, default="PGSSI_WuBenchmark")
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1.0e-4)
    parser.add_argument("--n-epochs", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--weight-decay", type=float, default=2.0e-5)
    parser.add_argument("--early-stopping-patience", type=int, default=80)
    parser.add_argument("--checkpoint-interval", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--init-model-path", type=str, default=str(DEFAULT_INIT))
    parser.add_argument("--physics-loss-weight", type=float, default=0.0)
    parser.add_argument("--disable-cross-interaction", action="store_true")
    parser.add_argument("--num-intra-layers", type=int, default=2)
    parser.add_argument("--disable-interaction-types", action="store_true")
    parser.add_argument("--disable-moe", action="store_true")
    parser.add_argument("--disable-physics-prior", action="store_true")
    parser.add_argument("--disable-cross-refine", action="store_true")
    parser.add_argument("--topology-only", action="store_true")
    parser.add_argument("--direct-loggamma-head", action="store_true")
    parser.add_argument("--disable-formula-layer", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--quiet-progress", action="store_true")
    return parser.parse_args()


def _metric_row(dataset_name: str, metrics: dict) -> dict:
    return {
        "Model": "PGSSI",
        "Dataset": dataset_name,
        "MAE": float(metrics.get("mae", float("nan"))),
        "R2": float(metrics.get("r2", float("nan"))),
        "AE<=0.1": float(metrics.get("AE<=0.1", float("nan"))),
        "AE<=0.2": float(metrics.get("AE<=0.2", float("nan"))),
        "AE<=0.3": float(metrics.get("AE<=0.3", float("nan"))),
    }


def _baseline_frame(dataset_name: str, rows: list[dict]) -> pd.DataFrame:
    frame = pd.DataFrame(rows).copy()
    frame.insert(1, "Dataset", dataset_name)
    return frame


def build_comparison_tables(wu_metrics: dict, idac_metrics: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    wu_frame = pd.concat(
        [_baseline_frame("Wu2004", WU_BASELINES), pd.DataFrame([_metric_row("Wu2004", wu_metrics)])],
        ignore_index=True,
    )
    idac_frame = pd.concat(
        [_baseline_frame("IDAC2026", IDAC2026_BASELINES), pd.DataFrame([_metric_row("IDAC2026", idac_metrics)])],
        ignore_index=True,
    )
    return wu_frame, idac_frame


def main():
    args = parse_args()
    set_seed(args.seed)
    configure_torch_runtime()

    run_dir = make_run_dir(args.run_dir, prefix="pgssi_wu_benchmark")
    cache_dir = make_cache_dir(args.cache_dir, prefix="pgssi_wu_benchmark")

    train_path = resolve_input_path(args.train_path)
    valid_path = resolve_input_path(args.valid_path)
    wu_test_path = resolve_input_path(args.wu_test_path)
    idac_test_path = resolve_input_path(args.idac_test_path)

    train_df = pd.read_csv(train_path)
    valid_df = pd.read_csv(valid_path)

    enable_cross_interaction = not args.disable_cross_interaction
    use_interaction_types = not args.disable_interaction_types
    use_moe = not args.disable_moe
    use_physics_prior = not args.disable_physics_prior
    artifact_prefix = f"{dataset_cache_prefix(train_path)}_{args.model_name}"

    best_model_path = train_PGSSI(
        train_df=train_df,
        valid_df=valid_df,
        model_name=args.model_name,
        hyperparameters={
            "hidden_dim": args.hidden_dim,
            "lr": args.lr,
            "n_epochs": args.n_epochs,
            "batch_size": args.batch_size,
            "weight_decay": args.weight_decay,
            "early_stopping_patience": args.early_stopping_patience,
            "enable_cross_interaction": enable_cross_interaction,
            "num_intra_layers": args.num_intra_layers,
            "use_interaction_types": use_interaction_types,
            "use_moe": use_moe,
            "use_physics_prior": use_physics_prior,
            "disable_cross_refine": args.disable_cross_refine,
            "topology_only": args.topology_only,
            "direct_loggamma_head": args.direct_loggamma_head,
            "disable_formula_layer": args.disable_formula_layer,
            "train_cache_prefix": dataset_cache_prefix(train_path),
            "valid_cache_prefix": dataset_cache_prefix(valid_path),
            "artifact_prefix": artifact_prefix,
            "checkpoint_interval": args.checkpoint_interval,
            "physics_loss_weight": args.physics_loss_weight,
            "init_model_path": args.init_model_path,
            "quiet_progress": args.quiet_progress,
        },
        output_dir=run_dir,
        cache_dir=cache_dir,
        resume=args.resume,
    )

    wu_metrics = evaluate_test_file(
        model_path=best_model_path,
        test_path=str(wu_test_path),
        test_name=dataset_cache_prefix(wu_test_path),
        model_name=args.model_name,
        artifact_prefix=artifact_prefix,
        hidden_dim=args.hidden_dim,
        enable_cross_interaction=enable_cross_interaction,
        num_intra_layers=args.num_intra_layers,
        use_interaction_types=use_interaction_types,
        use_moe=use_moe,
        use_physics_prior=use_physics_prior,
        disable_cross_refine=args.disable_cross_refine,
        topology_only=args.topology_only,
        direct_loggamma_head=args.direct_loggamma_head,
        disable_formula_layer=args.disable_formula_layer,
        output_dir=run_dir,
        cache_dir=cache_dir,
    )

    idac_metrics = evaluate_test_file(
        model_path=best_model_path,
        test_path=str(idac_test_path),
        test_name=dataset_cache_prefix(idac_test_path),
        model_name=args.model_name,
        artifact_prefix=artifact_prefix,
        hidden_dim=args.hidden_dim,
        enable_cross_interaction=enable_cross_interaction,
        num_intra_layers=args.num_intra_layers,
        use_interaction_types=use_interaction_types,
        use_moe=use_moe,
        use_physics_prior=use_physics_prior,
        disable_cross_refine=args.disable_cross_refine,
        topology_only=args.topology_only,
        direct_loggamma_head=args.direct_loggamma_head,
        disable_formula_layer=args.disable_formula_layer,
        output_dir=run_dir,
        cache_dir=cache_dir,
    )

    print_test_summary(
        {
            "Wu2004": wu_metrics,
            "IDAC2026": idac_metrics,
        }
    )

    wu_frame, idac_frame = build_comparison_tables(wu_metrics, idac_metrics)
    combined = pd.concat([wu_frame, idac_frame], ignore_index=True)

    wu_path = run_dir / "wu2004_comparison.csv"
    idac_path = run_dir / "idac2026_comparison.csv"
    combined_path = run_dir / "wu_idac_benchmark_comparison.csv"
    md_path = run_dir / "wu_idac_benchmark_comparison.md"
    summary_path = run_dir / "wu_idac_benchmark_summary.json"

    wu_frame.to_csv(wu_path, index=False)
    idac_frame.to_csv(idac_path, index=False)
    combined.to_csv(combined_path, index=False)
    combined.to_markdown(md_path, index=False)

    summary = {
        "run_dir": str(run_dir),
        "cache_dir": str(cache_dir),
        "best_model_path": str(best_model_path),
        "wu_metrics": wu_metrics,
        "idac_metrics": idac_metrics,
        "wu_comparison_file": str(wu_path),
        "idac_comparison_file": str(idac_path),
        "combined_comparison_file": str(combined_path),
    }
    with open(summary_path, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, ensure_ascii=False)

    print(f"Saved benchmark summary to: {summary_path}")
    print(combined.to_string(index=False))


if __name__ == "__main__":
    main()
