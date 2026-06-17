"""Run a PGSSI sensitivity sweep across multiple seeds and forcefields.

This script repeatedly invokes ``PGSSI_train.py`` and collects the per-run
test metrics into a machine-readable CSV summary.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
import subprocess
import sys
import concurrent.futures
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
TRAIN_SCRIPT = PROJECT_ROOT / "src" / "models" / "PGSSI" / "PGSSI_train_.py"
DEFAULT_SEEDS = [42, 2026, 3407, 1024, 8888]
DEFAULT_FORCEFIELDS = ["MMFF", "UFF"]


def dataset_prefix(path: str | Path) -> str:
    return Path(path).stem


def add_common_training_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--train-path",
        type=str,
        default=str(PROJECT_ROOT / "dataset" / "all" / "all_merged_train.csv"),
    )
    parser.add_argument(
        "--valid-path",
        type=str,
        default=str(PROJECT_ROOT / "dataset" / "all" / "all_merged_valid.csv"),
    )
    parser.add_argument(
        "--test-path",
        type=str,
        default=str(PROJECT_ROOT / "dataset" / "all" / "all_merged_test.csv"),
    )
    parser.add_argument("--model-name", type=str, default="PGSSI")
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--lr", type=float, default=2.5e-4)
    parser.add_argument("--n-epochs", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--train-num-workers", type=int, default=0)
    parser.add_argument("--valid-num-workers", type=int, default=0)
    parser.add_argument("--weight-decay", type=float, default=2e-5)
    parser.add_argument("--early-stopping-patience", type=int, default=20)
    parser.add_argument("--checkpoint-interval", type=int, default=10)
    parser.add_argument("--num-intra-layers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=None, help="Legacy alias for a single seed run.")
    parser.add_argument("--forcefield", type=str, default=None, choices=["MMFF", "UFF"], help="Legacy alias for a single forcefield run.")
    parser.add_argument("--disable-cross-interaction", action="store_true")
    parser.add_argument("--disable-interaction-types", action="store_true")
    parser.add_argument("--disable-moe", action="store_true")
    parser.add_argument("--disable-physics-prior", action="store_true")
    parser.add_argument("--disable-cross-refine", action="store_true")
    parser.add_argument("--topology-only", action="store_true")
    parser.add_argument("--direct-loggamma-head", action="store_true")
    parser.add_argument("--disable-formula-layer", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--init-model-path", type=str, default=None)
    parser.add_argument("--quiet-progress", action="store_true")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the PGSSI training script for a seed/forcefield sweep and aggregate the results."
    )
    add_common_training_args(parser)
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=DEFAULT_SEEDS,
        help="Seeds to evaluate. Default: 5 seeds used in the robustness study.",
    )
    parser.add_argument(
        "--forcefields",
        type=str,
        nargs="+",
        default=DEFAULT_FORCEFIELDS,
        choices=["MMFF", "UFF"],
        help="Forcefields to evaluate. Default: MMFF and UFF.",
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default=str(PROJECT_ROOT / "runs" / "pgssi_sensitivity"),
        help="Directory that will contain one subfolder per run and the aggregated CSV files.",
    )
    parser.add_argument(
        "--cache-root",
        type=str,
        default=str(PROJECT_ROOT / "cache" / "pgssi_sensitivity"),
        help="Directory that will contain one cache subfolder per run.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit how many seed/forcefield combinations to run. Helpful for smoke tests.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only print the commands that would be executed.",
    )
    parser.add_argument(
        "--stop-on-error",
        action="store_true",
        help="Stop the sweep as soon as a single run fails.",
    )
    return parser.parse_args()


def build_train_command(args: argparse.Namespace, seed: int, forcefield: str, run_dir: Path, cache_dir: Path) -> list[str]:
    cmd = [
        sys.executable,
        str(TRAIN_SCRIPT),
        "--train-path",
        args.train_path,
        "--valid-path",
        args.valid_path,
        "--test-path",
        args.test_path,
        "--run-dir",
        str(run_dir),
        "--cache-dir",
        str(cache_dir),
        "--model-name",
        args.model_name,
        "--hidden-dim",
        str(args.hidden_dim),
        "--lr",
        str(args.lr),
        "--n-epochs",
        str(args.n_epochs),
        "--batch-size",
        str(args.batch_size),
        "--train-num-workers",
        str(args.train_num_workers),
        "--valid-num-workers",
        str(args.valid_num_workers),
        "--weight-decay",
        str(args.weight_decay),
        "--early-stopping-patience",
        str(args.early_stopping_patience),
        "--checkpoint-interval",
        str(args.checkpoint_interval),
        "--num-intra-layers",
        str(args.num_intra_layers),
        "--seed",
        str(seed),
        "--forcefield",
        forcefield,
    ]

    if args.disable_cross_interaction:
        cmd.append("--disable-cross-interaction")
    if args.disable_interaction_types:
        cmd.append("--disable-interaction-types")
    if args.disable_moe:
        cmd.append("--disable-moe")
    if args.disable_physics_prior:
        cmd.append("--disable-physics-prior")
    if args.disable_cross_refine:
        cmd.append("--disable-cross-refine")
    if args.topology_only:
        cmd.append("--topology-only")
    if args.direct_loggamma_head:
        cmd.append("--direct-loggamma-head")
    if args.disable_formula_layer:
        cmd.append("--disable-formula-layer")
    if args.resume:
        cmd.append("--resume")
    if args.init_model_path is not None:
        cmd.extend(["--init-model-path", args.init_model_path])
    if args.quiet_progress:
        cmd.append("--quiet-progress")

    return cmd


def tee_process_output(command: list[str], log_path: Path) -> int:
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"

    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log_file:
        log_file.write("Command:\n")
        log_file.write(" ".join(command) + "\n\n")
        log_file.flush()

        process = subprocess.Popen(
            command,
            cwd=PROJECT_ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=env,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="")
            log_file.write(line)
        return_code = process.wait()
        log_file.write(f"\nReturn code: {return_code}\n")
        log_file.flush()
    return return_code


def load_json_if_exists(path: Path) -> dict:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def extract_test_metrics(output_dir: Path, train_path: str, model_name: str) -> tuple[dict, dict]:
    artifact_prefix = f"{dataset_prefix(train_path)}_{model_name}"
    train_summary = load_json_if_exists(output_dir / f"{artifact_prefix}_summary.json")
    test_summary = load_json_if_exists(output_dir / f"{artifact_prefix}_test_summary.json")
    if not test_summary:
        return train_summary, {}
    test_name, metrics = next(iter(test_summary.items()))
    metrics = dict(metrics)
    metrics["test_name"] = test_name
    return train_summary, metrics


def _metric_stats(values: list[float]) -> dict:
    if not values:
        return {"mean": None, "std": None, "min": None, "max": None, "count": 0}
    if len(values) == 1:
        return {
            "mean": float(values[0]),
            "std": 0.0,
            "min": float(values[0]),
            "max": float(values[0]),
            "count": 1,
        }
    return {
        "mean": float(statistics.mean(values)),
        "std": float(statistics.pstdev(values)),
        "min": float(min(values)),
        "max": float(max(values)),
        "count": len(values),
    }


def build_summary_table(records: list[dict]) -> list[dict]:
    if not records:
        return []

    df = records
    metric_names = ["mae", "rmse", "r2", "AE<=0.1", "AE<=0.2", "AE<=0.3"]
    rows: list[dict] = []

    def append_stats(group_name: str, frame: list[dict]) -> None:
        for metric in metric_names:
            metric_values: list[float] = []
            for row in frame:
                value = row.get(metric)
                if value in (None, ""):
                    continue
                try:
                    metric_values.append(float(value))
                except (TypeError, ValueError):
                    continue
            stats = _metric_stats(metric_values)
            if stats["count"] == 0:
                continue
            rows.append(
                {
                    "group": group_name,
                    "metric": metric,
                    **stats,
                }
            )

    append_stats("overall", df)
    forcefields = sorted({row.get("forcefield") for row in df if row.get("forcefield") is not None})
    for forcefield in forcefields:
        frame = [row for row in df if row.get("forcefield") == forcefield]
        append_stats(f"forcefield={forcefield}", frame)

    return rows


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return

    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def run_sweep(args: argparse.Namespace) -> tuple[list[dict], list[dict]]:
    output_root = Path(args.output_root)
    cache_root = Path(args.cache_root)
    output_root.mkdir(parents=True, exist_ok=True)
    cache_root.mkdir(parents=True, exist_ok=True)

    combinations = [(seed, forcefield) for seed in args.seeds for forcefield in args.forcefields]
    if args.seed is not None or args.forcefield is not None:
        single_seed = args.seed if args.seed is not None else args.seeds[0]
        single_forcefield = args.forcefield if args.forcefield is not None else args.forcefields[0]
        combinations = [(single_seed, single_forcefield)]

    if args.limit is not None:
        combinations = combinations[: max(args.limit, 0)]

    if args.dry_run:
        print(f"Dry run: {len(combinations)} run(s) planned")

    records: list[dict] = []

    def _run_single_experiment(index, seed, forcefield):
        run_name = f"seed{seed}_{forcefield}"
        run_dir = output_root / run_name
        cache_dir = cache_root / run_name
        command = build_train_command(args, seed, forcefield, run_dir, cache_dir)
        
        # 【关键优化】强制关闭 DataLoader 多进程，完全依靠主进程将缓存推给显卡
        command.extend(["--train-num-workers", "0", "--valid-num-workers", "0"])

        print(f"\n[{index}/{len(combinations)}] Starting {run_name} in background...")
        if args.dry_run:
            return {
                "seed": seed, "forcefield": forcefield, "run_name": run_name,
                "status": "dry_run", "run_dir": str(run_dir), "cache_dir": str(cache_dir)
            }

        log_path = run_dir / "train.log"
        return_code = tee_process_output(command, log_path)
        train_summary, test_metrics = extract_test_metrics(run_dir, args.train_path, args.model_name)

        record = {
            "seed": seed,
            "forcefield": forcefield,
            "run_name": run_name,
            "status": "ok" if return_code == 0 else "failed",
            "return_code": return_code,
            "run_dir": str(run_dir),
            "cache_dir": str(cache_dir),
            "log_path": str(log_path),
            "best_model_path": train_summary.get("best_model_path"),
            "checkpoint_path": train_summary.get("checkpoint_path"),
            "output_dir": train_summary.get("output_dir", str(run_dir)),
            "train_summary_file": str(run_dir / f"{dataset_prefix(args.train_path)}_{args.model_name}_summary.json"),
            "test_summary_file": str(run_dir / f"{dataset_prefix(args.train_path)}_{args.model_name}_test_summary.json"),
        }
        record.update(test_metrics)
        return record

    # 【并发加速】基于 RTX 3090 的显存容量，我们开启 3 个任务并行
    max_concurrent_jobs = 3
    print(f"\nRunning up to {max_concurrent_jobs} sweeps concurrently...")
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_concurrent_jobs) as executor:
        futures = [
            executor.submit(_run_single_experiment, idx, seed, ff) 
            for idx, (seed, ff) in enumerate(combinations, start=1)
        ]
        for future in concurrent.futures.as_completed(futures):
            record = future.result()
            records.append(record)
            if record.get("status") != "ok" and args.stop_on_error:
                print("Error detected, cancelling remaining jobs...")
                executor.shutdown(wait=False, cancel_futures=True)
                break

    results_path = output_root / "sweep_results.csv"
    _write_csv(results_path, records)

    summary_df = build_summary_table(records)
    summary_path = output_root / "sweep_summary.csv"
    _write_csv(summary_path, summary_df)

    config = {
        "train_path": args.train_path,
        "valid_path": args.valid_path,
        "test_path": args.test_path,
        "model_name": args.model_name,
        "hidden_dim": args.hidden_dim,
        "lr": args.lr,
        "n_epochs": args.n_epochs,
        "batch_size": args.batch_size,
        "train_num_workers": args.train_num_workers,
        "valid_num_workers": args.valid_num_workers,
        "weight_decay": args.weight_decay,
        "early_stopping_patience": args.early_stopping_patience,
        "checkpoint_interval": args.checkpoint_interval,
        "num_intra_layers": args.num_intra_layers,
        "seeds": args.seeds,
        "forcefields": args.forcefields,
        "output_root": str(output_root),
        "cache_root": str(cache_root),
    }
    with (output_root / "sweep_config.json").open("w", encoding="utf-8") as fh:
        json.dump(config, fh, indent=2, ensure_ascii=False)

    print(f"\nSaved detailed results to: {results_path}")
    print(f"Saved aggregate summary to: {summary_path}")
    if summary_df:
        print("\nAggregate summary:")
        for row in summary_df:
            print(row)

    return records, summary_df


def main() -> int:
    args = parse_args()
    results_df, _ = run_sweep(args)
    if not results_df:
        return 1
    if any(row.get("status") != "ok" for row in results_df) and not args.dry_run:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())