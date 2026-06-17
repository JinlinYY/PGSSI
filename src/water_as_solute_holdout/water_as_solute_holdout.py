from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

from run_water_nonwater_cv import (
    DEFAULT_INPUT,
    PROJECT_ROOT,
    REQUIRED_COLUMNS,
    TRAIN_SCRIPT,
    add_water_flags,
    metric_row,
    pair_groups,
    summarize_metrics,
    validate_columns,
    verify_no_pair_overlap,
)


DEFAULT_OUTPUT = PROJECT_ROOT / "runs" / "water_as_solute_holdout_701515"
DEFAULT_CACHE = PROJECT_ROOT / "cache" / "water_as_solute_holdout_701515"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Repeated 70/15/15 holdout experiment for water-as-solute systems only."
    )
    parser.add_argument("--input-path", type=str, default=str(DEFAULT_INPUT))
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT))
    parser.add_argument("--cache-dir", type=str, default=str(DEFAULT_CACHE))
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--valid-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument("--seeds", type=str, default="auto", help='Comma-separated seeds or "auto".')
    parser.add_argument("--n-repeats", type=int, default=5)
    parser.add_argument("--seed-search-start", type=int, default=1)
    parser.add_argument("--seed-search-end", type=int, default=10000)
    parser.add_argument("--min-train-samples", type=int, default=25)
    parser.add_argument("--min-valid-samples", type=int, default=5)
    parser.add_argument("--min-test-samples", type=int, default=5)

    parser.add_argument("--model-name", type=str, default="PGSSI_WaterAsSolute")
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
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--quiet-progress", action="store_true")
    return parser.parse_args()


def parse_seed_list(seed_text: str) -> list[int] | None:
    if seed_text.lower() == "auto":
        return None
    seeds = [int(part.strip()) for part in seed_text.split(",") if part.strip()]
    if not seeds:
        raise ValueError("--seeds must be 'auto' or a non-empty comma-separated integer list.")
    return seeds


def split_701515(df: pd.DataFrame, seed: int, train_ratio: float) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    groups = pair_groups(df)
    first_splitter = GroupShuffleSplit(n_splits=1, train_size=train_ratio, random_state=seed)
    train_idx, temp_idx = next(first_splitter.split(df, groups=groups))

    train_df = df.iloc[train_idx].reset_index(drop=True)
    temp_df = df.iloc[temp_idx].reset_index(drop=True)
    temp_groups = pair_groups(temp_df)

    second_splitter = GroupShuffleSplit(n_splits=1, test_size=0.50, random_state=seed + 10000)
    valid_idx, test_idx = next(second_splitter.split(temp_df, groups=temp_groups))
    valid_df = temp_df.iloc[valid_idx].reset_index(drop=True)
    test_df = temp_df.iloc[test_idx].reset_index(drop=True)
    verify_no_pair_overlap(train_df, valid_df, test_df)
    return train_df, valid_df, test_df


def split_summary(seed: int, train_df: pd.DataFrame, valid_df: pd.DataFrame, test_df: pd.DataFrame) -> dict:
    return {
        "seed": seed,
        "train_samples": int(len(train_df)),
        "valid_samples": int(len(valid_df)),
        "test_samples": int(len(test_df)),
        "train_pairs": int(train_df.groupby(["Solvent_SMILES", "Solute_SMILES"]).ngroups),
        "valid_pairs": int(valid_df.groupby(["Solvent_SMILES", "Solute_SMILES"]).ngroups),
        "test_pairs": int(test_df.groupby(["Solvent_SMILES", "Solute_SMILES"]).ngroups),
    }


def find_auto_seeds(args, df: pd.DataFrame) -> list[int]:
    selected = []
    for seed in range(args.seed_search_start, args.seed_search_end + 1):
        train_df, valid_df, test_df = split_701515(df, seed, args.train_ratio)
        if (
            len(train_df) >= args.min_train_samples
            and len(valid_df) >= args.min_valid_samples
            and len(test_df) >= args.min_test_samples
        ):
            selected.append(seed)
            if len(selected) == args.n_repeats:
                return selected
    raise RuntimeError("Could not find enough seeds satisfying the minimum split sizes.")


def run_training(
    args,
    seed: int,
    run_dir: Path,
    run_cache_dir: Path,
    train_path: Path,
    valid_path: Path,
    test_path: Path,
) -> Path:
    prediction_path = run_dir / f"{train_path.stem}_{args.model_name}_{test_path.stem}_predictions.csv"
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
        str(run_dir),
        "--cache-dir",
        str(run_cache_dir),
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
        str(seed),
    ]
    if args.quiet_progress:
        cmd.append("--quiet-progress")

    env = os.environ.copy()
    env.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    if args.force_cpu:
        env["CUDA_VISIBLE_DEVICES"] = "-1"

    print(f"\n=== Seed {seed}: training water-as-solute PGSSI ===", flush=True)
    print(" ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=PROJECT_ROOT, env=env, check=True)
    return prediction_path


def write_markdown(df: pd.DataFrame, path: Path):
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(df.to_markdown(index=False))
        fh.write("\n")


def main():
    args = parse_args()
    ratio_sum = args.train_ratio + args.valid_ratio + args.test_ratio
    if abs(ratio_sum - 1.0) > 1e-8:
        raise ValueError("--train-ratio + --valid-ratio + --test-ratio must equal 1.0.")
    if abs(args.valid_ratio - args.test_ratio) > 1e-8:
        raise ValueError("This script currently expects equal validation and test ratios.")

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
    flagged_df = add_water_flags(df)
    water_solute_df = flagged_df[flagged_df["is_water_solute"]].reset_index(drop=True)
    if water_solute_df.empty:
        raise ValueError("No water-as-solute samples were found.")

    dataset_summary = {
        "input_path": str(input_path.resolve()),
        "subset": "water_as_solute",
        "n_samples": int(len(water_solute_df)),
        "n_pairs": int(water_solute_df.groupby(["Solvent_SMILES", "Solute_SMILES"]).ngroups),
        "target_mean": float(water_solute_df["log-gamma"].mean()),
        "target_std": float(water_solute_df["log-gamma"].std(ddof=1)),
        "target_min": float(water_solute_df["log-gamma"].min()),
        "target_max": float(water_solute_df["log-gamma"].max()),
        "split_protocol": "five repeated grouped holdout splits with train/valid/test = 70/15/15",
    }
    with open(output_dir / "water_as_solute_dataset_summary.json", "w", encoding="utf-8") as fh:
        json.dump(dataset_summary, fh, indent=2, ensure_ascii=False)
    pd.DataFrame([dataset_summary]).to_csv(output_dir / "water_as_solute_dataset_summary.csv", index=False)
    print(json.dumps(dataset_summary, indent=2, ensure_ascii=False), flush=True)

    requested_seeds = parse_seed_list(args.seeds)
    seeds = requested_seeds if requested_seeds is not None else find_auto_seeds(args, water_solute_df)
    seeds = seeds[: args.n_repeats]
    print(f"Using seeds: {seeds}", flush=True)

    split_rows = []
    metric_rows = []
    for run_index, seed in enumerate(seeds, start=1):
        train_df, valid_df, test_df = split_701515(water_solute_df, seed, args.train_ratio)
        current_split = split_summary(seed, train_df, valid_df, test_df)
        current_split["run"] = run_index
        split_rows.append(current_split)

        run_dir = output_dir / f"seed_{seed}"
        run_cache_dir = cache_dir / f"seed_{seed}"
        run_dir.mkdir(parents=True, exist_ok=True)
        run_cache_dir.mkdir(parents=True, exist_ok=True)

        train_path = split_dir / f"seed_{seed}_water_as_solute_train.csv"
        valid_path = split_dir / f"seed_{seed}_water_as_solute_valid.csv"
        test_path = split_dir / f"seed_{seed}_water_as_solute_test.csv"
        train_df[REQUIRED_COLUMNS].to_csv(train_path, index=False)
        valid_df[REQUIRED_COLUMNS].to_csv(valid_path, index=False)
        test_df[REQUIRED_COLUMNS].to_csv(test_path, index=False)
        with open(run_dir / "split_summary.json", "w", encoding="utf-8") as fh:
            json.dump(current_split, fh, indent=2, ensure_ascii=False)

        if args.prepare_only:
            continue

        prediction_path = run_training(args, seed, run_dir, run_cache_dir, train_path, valid_path, test_path)
        pred_df = pd.read_csv(prediction_path)
        row = metric_row(pred_df, "water_as_solute", seed)
        row["run"] = run_index
        row["seed"] = seed
        metric_rows.append(row)
        pd.DataFrame([row]).to_csv(run_dir / "water_as_solute_metrics.csv", index=False)

    split_df = pd.DataFrame(split_rows)
    split_path = output_dir / "water_as_solute_split_summary.csv"
    split_df.to_csv(split_path, index=False)
    write_markdown(split_df, output_dir / "water_as_solute_split_summary.md")
    print(f"\nSaved split summary to: {split_path}", flush=True)

    if args.prepare_only:
        print("Prepare-only mode: skipped model training and metric aggregation.", flush=True)
        print(split_df.to_string(index=False), flush=True)
        return

    metrics_df = pd.DataFrame(metric_rows)
    summary_df = summarize_metrics(metrics_df.rename(columns={"seed": "seed_value"}))
    metrics_path = output_dir / "water_as_solute_holdout_metrics.csv"
    summary_path = output_dir / "water_as_solute_holdout_summary_metrics.csv"
    metrics_df.to_csv(metrics_path, index=False)
    summary_df.to_csv(summary_path, index=False)
    write_markdown(metrics_df, output_dir / "water_as_solute_holdout_metrics.md")
    write_markdown(summary_df, output_dir / "water_as_solute_holdout_summary_metrics.md")

    print(f"Saved holdout metrics to: {metrics_path}", flush=True)
    print(f"Saved summary metrics to: {summary_path}", flush=True)
    print(summary_df.replace({np.nan: ""}).to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
