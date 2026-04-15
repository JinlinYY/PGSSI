"""Pair-grouped, distribution-aware dataset split.

This splitter keeps every `(Solvent_SMILES, Solute_SMILES)` pair entirely within
one split and uses a greedy assignment to approximately preserve the global
`log-gamma` distribution across train/valid/test.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


REQUIRED_COLUMNS = ["Solvent_SMILES", "Solute_SMILES", "T", "log-gamma"]
DEFAULT_INPUT = Path(__file__).resolve().parents[2] / "dataset" / "all" / "all_merged.csv"


def validate_columns(df: pd.DataFrame):
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def make_bins(target: pd.Series, n_bins: int = 10) -> pd.Series:
    effective_bins = max(2, min(n_bins, target.nunique()))
    try:
        bins = pd.qcut(target, q=effective_bins, labels=False, duplicates="drop")
    except ValueError:
        bins = pd.Series(np.zeros(len(target), dtype=int), index=target.index)
    return bins.astype(int)


def build_pair_groups(df: pd.DataFrame, n_bins: int = 10) -> tuple[pd.DataFrame, np.ndarray]:
    work_df = df.copy()
    work_df["pair_key"] = list(zip(work_df["Solvent_SMILES"], work_df["Solute_SMILES"]))
    work_df["target_bin"] = make_bins(work_df["log-gamma"], n_bins=n_bins)

    global_bin_counts = (
        work_df["target_bin"]
        .value_counts(sort=False)
        .sort_index()
        .reindex(range(work_df["target_bin"].max() + 1), fill_value=0)
        .to_numpy(dtype=float)
    )

    records = []
    for pair_key, group in work_df.groupby("pair_key", sort=False):
        bin_counts = (
            group["target_bin"]
            .value_counts(sort=False)
            .sort_index()
            .reindex(range(len(global_bin_counts)), fill_value=0)
            .to_numpy(dtype=float)
        )
        records.append(
            {
                "pair_key": pair_key,
                "rows": int(len(group)),
                "target_mean": float(group["log-gamma"].mean()),
                "target_std": float(group["log-gamma"].std(ddof=0)) if len(group) > 1 else 0.0,
                "bin_counts": bin_counts,
            }
        )

    pair_df = pd.DataFrame(records)
    pair_df = pair_df.sort_values(
        by=["rows", "target_std", "target_mean"],
        ascending=[False, False, False],
        kind="mergesort",
    ).reset_index(drop=True)
    pair_df["target_bin"] = make_bins(pair_df["target_mean"], n_bins=min(n_bins, len(pair_df)))
    return pair_df, global_bin_counts


def choose_split(
    row_count: int,
    bin_counts: np.ndarray,
    split_counts: dict[str, float],
    split_bin_counts: dict[str, np.ndarray],
    target_rows: dict[str, float],
    target_bins: dict[str, np.ndarray],
) -> str:
    best_name = None
    best_score = None
    total_rows = sum(target_rows.values())
    assigned_rows = sum(split_counts.values())
    completion_target = (assigned_rows + row_count) / max(total_rows, 1.0)

    for split_name in target_rows:
        new_rows = split_counts[split_name] + row_count
        new_bins = split_bin_counts[split_name] + bin_counts

        completion_error = abs((new_rows / max(target_rows[split_name], 1.0)) - completion_target)
        bin_error = np.abs(new_bins - target_bins[split_name]).sum() / max(target_bins[split_name].sum(), 1.0)
        overflow = max(0.0, new_rows - target_rows[split_name]) / max(total_rows, 1.0)
        score = completion_error + 0.35 * bin_error + 12.0 * overflow

        if best_score is None or score < best_score:
            best_name = split_name
            best_score = score

    return best_name


def maybe_stratify(values: pd.Series):
    counts = values.value_counts()
    if len(counts) < 2 or int(counts.min()) < 2:
        return None
    return values


def assign_pairs(pair_df: pd.DataFrame, ratios: dict[str, float], random_state: int = 42) -> dict[str, list[tuple[str, str]]]:
    valid_plus_test_ratio = ratios["valid"] + ratios["test"]
    stratify_all = maybe_stratify(pair_df["target_bin"])
    train_pairs, holdout_pairs = train_test_split(
        pair_df,
        test_size=valid_plus_test_ratio,
        random_state=random_state,
        shuffle=True,
        stratify=stratify_all,
    )

    valid_fraction_of_holdout = ratios["valid"] / valid_plus_test_ratio
    stratify_holdout = maybe_stratify(holdout_pairs["target_bin"])
    valid_pairs, test_pairs = train_test_split(
        holdout_pairs,
        train_size=valid_fraction_of_holdout,
        random_state=random_state,
        shuffle=True,
        stratify=stratify_holdout,
    )

    return {
        "train": train_pairs["pair_key"].tolist(),
        "valid": valid_pairs["pair_key"].tolist(),
        "test": test_pairs["pair_key"].tolist(),
    }


def verify_no_pair_overlap(splits: dict[str, pd.DataFrame]):
    names = list(splits)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            left_pairs = set(zip(splits[names[i]]["Solvent_SMILES"], splits[names[i]]["Solute_SMILES"]))
            right_pairs = set(zip(splits[names[j]]["Solvent_SMILES"], splits[names[j]]["Solute_SMILES"]))
            pair_overlap = left_pairs.intersection(right_pairs)
            if pair_overlap:
                raise ValueError(
                    f"Pair leakage detected between {names[i]} and {names[j]}: {len(pair_overlap)} overlapping pairs."
                )


def print_split_summary(splits: dict[str, pd.DataFrame], total_rows: int):
    print("\n" + "=" * 72, flush=True)
    print("Split summary", flush=True)
    print("=" * 72, flush=True)
    for name, part in splits.items():
        ratio = len(part) / max(total_rows, 1)
        pair_count = part.groupby(["Solvent_SMILES", "Solute_SMILES"]).ngroups if len(part) > 0 else 0
        y = part["log-gamma"]
        print(
            f"{name:<6} rows={len(part):>6} pairs={pair_count:>6} ratio={ratio:>7.4f} "
            f"log-gamma mean={y.mean():>8.4f} std={y.std(ddof=0):>8.4f} "
            f"min={y.min():>8.4f} max={y.max():>8.4f}",
            flush=True,
        )


def split_dataset_by_pair(
    input_csv_path: str | Path,
    output_dir: str | Path | None = None,
    train_ratio: float = 0.8,
    valid_ratio: float = 0.1,
    test_ratio: float = 0.1,
):
    input_csv_path = Path(input_csv_path)
    output_dir = Path(output_dir) if output_dir is not None else input_csv_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Reading dataset: {input_csv_path}", flush=True)
    df = pd.read_csv(input_csv_path)
    validate_columns(df)
    df = df[REQUIRED_COLUMNS].copy()
    print(f"Total rows: {len(df)}", flush=True)

    ratio_sum = train_ratio + valid_ratio + test_ratio
    if not math.isclose(ratio_sum, 1.0, rel_tol=0.0, abs_tol=1e-8):
        raise ValueError(f"Split ratios must sum to 1.0, got {ratio_sum}")
    ratios = {"train": train_ratio, "valid": valid_ratio, "test": test_ratio}

    pair_df, global_bin_counts = build_pair_groups(df)
    print(f"Unique pairs: {len(pair_df)}", flush=True)
    print(f"Largest pair rows: {int(pair_df['rows'].max())}", flush=True)

    assignments = assign_pairs(pair_df, ratios=ratios)
    pair_key_series = pd.Series(list(zip(df["Solvent_SMILES"], df["Solute_SMILES"])), index=df.index)
    splits = {
        split_name: df[pair_key_series.isin(pair_keys)].reset_index(drop=True)
        for split_name, pair_keys in assignments.items()
    }

    empty_splits = [name for name, part in splits.items() if part.empty]
    if empty_splits:
        raise ValueError(f"Split failed; empty splits found: {empty_splits}")

    verify_no_pair_overlap(splits)

    stem = input_csv_path.stem
    train_path = output_dir / f"{stem}_train.csv"
    valid_path = output_dir / f"{stem}_valid.csv"
    test_path = output_dir / f"{stem}_test.csv"
    splits["train"].to_csv(train_path, index=False)
    splits["valid"].to_csv(valid_path, index=False)
    splits["test"].to_csv(test_path, index=False)

    print_split_summary(splits, total_rows=len(df))
    print(f"\nSaved train: {train_path}", flush=True)
    print(f"Saved valid: {valid_path}", flush=True)
    print(f"Saved test : {test_path}", flush=True)
    return splits["train"], splits["valid"], splits["test"]


def parse_args():
    parser = argparse.ArgumentParser(description="Pair-grouped, distribution-aware split")
    parser.add_argument("--input-csv", type=str, default=str(DEFAULT_INPUT))
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--valid-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    split_dataset_by_pair(
        input_csv_path=args.input_csv,
        output_dir=args.output_dir,
        train_ratio=args.train_ratio,
        valid_ratio=args.valid_ratio,
        test_ratio=args.test_ratio,
    )
