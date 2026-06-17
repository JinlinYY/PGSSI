"""Isomerism subset analysis and pair-grouped 5-fold PGSSI experiments.

This script reads ``dataset/all/all_merged.csv``, identifies stereochemical or
isomeric samples with RDKit canonical/isomeric SMILES first and SMILES markers
as a fallback, writes analysis artifacts, creates solute-solvent pair grouped
5-fold splits, optionally trains full PGSSI and the topology-only baseline, and
summarizes log-gamma metrics as CSV, Markdown, and LaTeX tables.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, GroupShuffleSplit


matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    from rdkit import Chem
except ImportError:  # pragma: no cover - handled at runtime in environments without RDKit.
    Chem = None


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT = PROJECT_ROOT / "dataset" / "all" / "all_merged.csv"
DEFAULT_OUTPUT = SCRIPT_DIR
TRAIN_SCRIPT = PROJECT_ROOT / "src" / "models" / "PGSSI" / "PGSSI_train.py"
REQUIRED_COLUMNS = ["Solute_SMILES", "Solvent_SMILES", "T", "log-gamma"]
PAIR_COLUMNS = ["Solute_SMILES", "Solvent_SMILES"]
HEURISTIC_PATTERN = re.compile(r"[@/\\]")


@dataclass(frozen=True)
class MoleculeInfo:
    raw_smiles: str
    rdkit_valid: bool
    canonical_isomeric: str
    canonical_nonisomeric: str
    rdkit_isomeric_flag: bool
    heuristic_flag: bool
    detection_method: str


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(
        description="Analyze isomerism subset and run pair-grouped 5-fold PGSSI experiments."
    )
    parser.add_argument("--input-csv", type=str, default=str(DEFAULT_INPUT))
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT))
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--valid-size", type=float, default=0.125, help="Validation fraction within each outer-train fold.")
    parser.add_argument("--prepare-only", action="store_true", help="Write analysis and split files without training.")
    parser.add_argument("--skip-analysis", action="store_true", help="Reuse existing isomer subset artifacts if present.")
    parser.add_argument("--resume", action="store_true", help="Pass --resume to PGSSI_train.py.")
    parser.add_argument("--force-train", action="store_true", help="Rerun training even if metrics JSON exists.")
    parser.add_argument("--python", type=str, default=sys.executable, help="Python executable used to launch training.")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--early-stopping-patience", type=int, default=20)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--train-num-workers", type=int, default=None)
    parser.add_argument("--valid-num-workers", type=int, default=None)
    parser.add_argument("--checkpoint-interval", type=int, default=10)
    parser.add_argument("--quiet-progress", action="store_true", help="Pass --quiet-progress to PGSSI_train.py.")
    parser.add_argument("--cache-dir", type=str, default=None, help="Shared cache root. Defaults to <output-dir>/cache.")
    parser.add_argument("--pair-cache-dir", type=str, default=None, help="Global pair graph cache. Defaults to <cache-root>/global_pair_graphs.")
    args, extra_train_args = parser.parse_known_args()
    return args, extra_train_args


def validate_columns(df: pd.DataFrame) -> None:
    missing = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def safe_text(value: object) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip()


def heuristic_isomer_flag(smiles: object) -> bool:
    return HEURISTIC_PATTERN.search(safe_text(smiles)) is not None


def canonicalize_smiles(smiles: object) -> MoleculeInfo:
    raw = safe_text(smiles)
    heuristic_flag = heuristic_isomer_flag(raw)
    if Chem is None or not raw:
        method = "heuristic" if heuristic_flag else "none"
        return MoleculeInfo(raw, False, raw, raw, False, heuristic_flag, method)

    mol = Chem.MolFromSmiles(raw)
    if mol is None:
        method = "heuristic" if heuristic_flag else "invalid"
        return MoleculeInfo(raw, False, raw, raw, False, heuristic_flag, method)

    canonical_isomeric = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
    canonical_nonisomeric = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=False)
    rdkit_flag = canonical_isomeric != canonical_nonisomeric
    if rdkit_flag and heuristic_flag:
        method = "rdkit+heuristic"
    elif rdkit_flag:
        method = "rdkit"
    elif heuristic_flag:
        method = "heuristic"
    else:
        method = "none"
    return MoleculeInfo(
        raw,
        True,
        canonical_isomeric,
        canonical_nonisomeric,
        rdkit_flag,
        heuristic_flag,
        method,
    )


def build_molecule_table(df: pd.DataFrame, column: str, role: str) -> pd.DataFrame:
    unique_smiles = pd.Series(df[column].dropna().unique(), name="raw_smiles")
    infos = [canonicalize_smiles(value) for value in unique_smiles]
    table = pd.DataFrame(
        [
            {
                "role": role,
                "raw_smiles": info.raw_smiles,
                "rdkit_valid": info.rdkit_valid,
                "canonical_isomeric": info.canonical_isomeric,
                "canonical_nonisomeric": info.canonical_nonisomeric,
                "rdkit_isomeric_flag": info.rdkit_isomeric_flag,
                "heuristic_flag": info.heuristic_flag,
                "detection_method": info.detection_method,
            }
            for info in infos
        ]
    )
    counts = df[column].map(safe_text).value_counts(dropna=False).rename_axis("raw_smiles").reset_index(name="sample_count")
    return table.merge(counts, on="raw_smiles", how="left")


def add_collision_flags(molecule_table: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    table = molecule_table.copy()
    group_cols = ["role", "canonical_nonisomeric"]
    group_stats = (
        table.groupby(group_cols, dropna=False)
        .agg(
            unique_raw_smiles=("raw_smiles", "nunique"),
            unique_isomeric_smiles=("canonical_isomeric", "nunique"),
            sample_count=("sample_count", "sum"),
            any_rdkit_isomeric=("rdkit_isomeric_flag", "max"),
            any_heuristic=("heuristic_flag", "max"),
        )
        .reset_index()
    )
    group_stats["isomer_collision_group"] = (
        (group_stats["unique_isomeric_smiles"] > 1)
        | ((group_stats["unique_raw_smiles"] > 1) & (group_stats["any_rdkit_isomeric"] | group_stats["any_heuristic"]))
    )
    group_stats["collision_group_id"] = ""

    collision_mask = group_stats["isomer_collision_group"]
    for role in group_stats.loc[collision_mask, "role"].unique():
        role_mask = collision_mask & (group_stats["role"] == role)
        order = group_stats.loc[role_mask].sort_values(["sample_count", "canonical_nonisomeric"], ascending=[False, True]).index
        for idx, row_idx in enumerate(order, start=1):
            group_stats.loc[row_idx, "collision_group_id"] = f"{role}_iso_collision_{idx:04d}"

    table = table.merge(
        group_stats[group_cols + ["isomer_collision_group", "collision_group_id"]],
        on=group_cols,
        how="left",
    )
    collision_details = table[table["isomer_collision_group"]].copy()
    collision_details = collision_details.sort_values(["role", "collision_group_id", "canonical_isomeric", "raw_smiles"])
    return table, collision_details


def annotate_dataset(df: pd.DataFrame, molecule_table: pd.DataFrame) -> pd.DataFrame:
    annotated = df.copy()
    for role, column in [("solute", "Solute_SMILES"), ("solvent", "Solvent_SMILES")]:
        role_table = molecule_table[molecule_table["role"] == role].copy()
        role_table = role_table.rename(
            columns={
                "raw_smiles": column,
                "rdkit_valid": f"{role}_rdkit_valid",
                "canonical_isomeric": f"{role}_canonical_isomeric",
                "canonical_nonisomeric": f"{role}_canonical_nonisomeric",
                "rdkit_isomeric_flag": f"{role}_rdkit_isomeric_flag",
                "heuristic_flag": f"{role}_heuristic_isomer_flag",
                "detection_method": f"{role}_isomer_detection_method",
                "isomer_collision_group": f"{role}_isomer_collision_group",
                "collision_group_id": f"{role}_collision_group_id",
            }
        )
        keep_cols = [
            column,
            f"{role}_rdkit_valid",
            f"{role}_canonical_isomeric",
            f"{role}_canonical_nonisomeric",
            f"{role}_rdkit_isomeric_flag",
            f"{role}_heuristic_isomer_flag",
            f"{role}_isomer_detection_method",
            f"{role}_isomer_collision_group",
            f"{role}_collision_group_id",
        ]
        annotated[column] = annotated[column].map(safe_text)
        annotated = annotated.merge(role_table[keep_cols], on=column, how="left")

    for role in ["solute", "solvent"]:
        for suffix in ["rdkit_isomeric_flag", "heuristic_isomer_flag", "isomer_collision_group"]:
            annotated[f"{role}_{suffix}"] = annotated[f"{role}_{suffix}"].fillna(False).astype(bool)

    annotated["isomer_subset_flag"] = (
        annotated["solute_rdkit_isomeric_flag"]
        | annotated["solvent_rdkit_isomeric_flag"]
        | annotated["solute_heuristic_isomer_flag"]
        | annotated["solvent_heuristic_isomer_flag"]
        | annotated["solute_isomer_collision_group"]
        | annotated["solvent_isomer_collision_group"]
    )
    annotated["isomer_detection_reason"] = annotated.apply(format_detection_reason, axis=1)
    return annotated


def format_detection_reason(row: pd.Series) -> str:
    reasons = []
    for role in ["solute", "solvent"]:
        if bool(row.get(f"{role}_rdkit_isomeric_flag", False)):
            reasons.append(f"{role}:rdkit")
        if bool(row.get(f"{role}_heuristic_isomer_flag", False)):
            reasons.append(f"{role}:heuristic")
        if bool(row.get(f"{role}_isomer_collision_group", False)):
            reasons.append(f"{role}:collision")
    return ";".join(reasons)


def build_pair_collision_groups(annotated: pd.DataFrame) -> pd.DataFrame:
    work = annotated.copy()
    work["pair_nonisomeric_signature"] = (
        work["solute_canonical_nonisomeric"].fillna(work["Solute_SMILES"].map(safe_text))
        + " || "
        + work["solvent_canonical_nonisomeric"].fillna(work["Solvent_SMILES"].map(safe_text))
    )
    work["pair_isomeric_signature"] = (
        work["solute_canonical_isomeric"].fillna(work["Solute_SMILES"].map(safe_text))
        + " || "
        + work["solvent_canonical_isomeric"].fillna(work["Solvent_SMILES"].map(safe_text))
    )
    group_stats = (
        work.groupby("pair_nonisomeric_signature", dropna=False)
        .agg(
            unique_pair_isomeric_forms=("pair_isomeric_signature", "nunique"),
            unique_raw_pairs=("pair_isomeric_signature", "nunique"),
            sample_count=("pair_isomeric_signature", "size"),
            unique_solute=("Solute_SMILES", "nunique"),
            unique_solvent=("Solvent_SMILES", "nunique"),
        )
        .reset_index()
    )
    group_stats = group_stats[group_stats["unique_pair_isomeric_forms"] > 1].copy()
    if group_stats.empty:
        return pd.DataFrame()
    group_stats = group_stats.sort_values(["sample_count", "pair_nonisomeric_signature"], ascending=[False, True]).reset_index(drop=True)
    group_stats["pair_collision_group_id"] = [f"pair_iso_collision_{idx:04d}" for idx in range(1, len(group_stats) + 1)]
    details = work.merge(
        group_stats[["pair_nonisomeric_signature", "pair_collision_group_id"]],
        on="pair_nonisomeric_signature",
        how="inner",
    )
    return details.sort_values(["pair_collision_group_id", "pair_isomeric_signature", "T"])


def pct(count: int, total: int) -> float:
    return float(count / total * 100.0) if total else 0.0


def write_summary(
    df: pd.DataFrame,
    annotated: pd.DataFrame,
    molecule_table: pd.DataFrame,
    collision_details: pd.DataFrame,
    pair_collision_details: pd.DataFrame,
    output_dir: Path,
    input_csv: Path,
) -> pd.DataFrame:
    total = len(df)
    isomer = annotated[annotated["isomer_subset_flag"]]
    rows = [
        {"section": "overall", "metric": "total_samples", "value": total, "percentage": 100.0},
        {"section": "overall", "metric": "unique_solute", "value": int(df["Solute_SMILES"].nunique()), "percentage": np.nan},
        {"section": "overall", "metric": "unique_solvent", "value": int(df["Solvent_SMILES"].nunique()), "percentage": np.nan},
        {
            "section": "isomer_subset",
            "metric": "isomer_samples",
            "value": int(len(isomer)),
            "percentage": pct(len(isomer), total),
        },
        {
            "section": "isomer_subset",
            "metric": "unique_isomer_solute",
            "value": int(isomer["Solute_SMILES"].nunique()),
            "percentage": np.nan,
        },
        {
            "section": "isomer_subset",
            "metric": "unique_isomer_solvent",
            "value": int(isomer["Solvent_SMILES"].nunique()),
            "percentage": np.nan,
        },
        {
            "section": "isomer_subset",
            "metric": "unique_isomer_pairs",
            "value": int(isomer.groupby(PAIR_COLUMNS, dropna=False).ngroups) if len(isomer) else 0,
            "percentage": np.nan,
        },
        {
            "section": "rdkit",
            "metric": "unique_valid_molecules",
            "value": int(molecule_table["rdkit_valid"].sum()),
            "percentage": pct(int(molecule_table["rdkit_valid"].sum()), len(molecule_table)),
        },
        {
            "section": "rdkit",
            "metric": "unique_rdkit_isomeric_molecules",
            "value": int(molecule_table["rdkit_isomeric_flag"].sum()),
            "percentage": pct(int(molecule_table["rdkit_isomeric_flag"].sum()), len(molecule_table)),
        },
        {
            "section": "heuristic",
            "metric": "unique_heuristic_isomeric_molecules",
            "value": int(molecule_table["heuristic_flag"].sum()),
            "percentage": pct(int(molecule_table["heuristic_flag"].sum()), len(molecule_table)),
        },
        {
            "section": "collision",
            "metric": "component_collision_groups",
            "value": int(collision_details["collision_group_id"].nunique()) if not collision_details.empty else 0,
            "percentage": np.nan,
        },
        {
            "section": "collision",
            "metric": "pair_collision_groups",
            "value": int(pair_collision_details["pair_collision_group_id"].nunique()) if not pair_collision_details.empty else 0,
            "percentage": np.nan,
        },
    ]
    summary = pd.DataFrame(rows)
    summary.to_csv(output_dir / "isomerism_summary.csv", index=False)

    md_lines = [
        "# Isomerism subset analysis",
        "",
        f"Input: `{input_csv.relative_to(PROJECT_ROOT) if input_csv.is_relative_to(PROJECT_ROOT) else input_csv}`",
        "",
        "Detection priority: RDKit canonical isomeric vs non-isomeric SMILES; the `@`, `/`, and `\\` marker heuristic is retained as a fallback and supplement.",
        "",
        f"- Total samples: {total}",
        f"- Isomer/stereochemistry subset samples: {len(isomer)} ({pct(len(isomer), total):.2f}%)",
        f"- Unique solutes in subset: {int(isomer['Solute_SMILES'].nunique()) if len(isomer) else 0}",
        f"- Unique solvents in subset: {int(isomer['Solvent_SMILES'].nunique()) if len(isomer) else 0}",
        f"- Component collision groups: {int(collision_details['collision_group_id'].nunique()) if not collision_details.empty else 0}",
        f"- Pair collision groups: {int(pair_collision_details['pair_collision_group_id'].nunique()) if not pair_collision_details.empty else 0}",
        "",
        "Generated files include `isomer_subset.csv`, `molecule_isomerism_table.csv`, `isomer_collision_groups.csv`, `pair_isomer_collision_groups.csv`, and `isomerism_statistics.png`.",
    ]
    (output_dir / "isomerism_summary.md").write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    return summary


def plot_isomerism_statistics(
    annotated: pd.DataFrame,
    molecule_table: pd.DataFrame,
    collision_details: pd.DataFrame,
    output_path: Path,
) -> None:
    isomer_mask = annotated["isomer_subset_flag"]
    reason_counts = (
        annotated.loc[isomer_mask, "isomer_detection_reason"]
        .str.get_dummies(sep=";")
        .sum()
        .sort_values(ascending=False)
    )
    if reason_counts.empty:
        reason_counts = pd.Series({"none": 0})

    role_counts = (
        molecule_table.groupby("role")[["rdkit_isomeric_flag", "heuristic_flag", "isomer_collision_group"]]
        .sum()
        .rename(columns={"rdkit_isomeric_flag": "RDKit", "heuristic_flag": "Heuristic", "isomer_collision_group": "Collision"})
    )
    collision_sizes = collision_details.groupby("collision_group_id")["raw_smiles"].nunique().sort_values(ascending=False)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes[0, 0].bar(["Full data", "Isomer subset"], [len(annotated), int(isomer_mask.sum())], color=["#4C78A8", "#F58518"])
    axes[0, 0].set_ylabel("Samples")
    axes[0, 0].set_title("Dataset scale")

    reason_counts.head(12).plot(kind="bar", ax=axes[0, 1], color="#54A24B")
    axes[0, 1].set_ylabel("Samples")
    axes[0, 1].set_title("Subset detection reasons")
    axes[0, 1].tick_params(axis="x", rotation=45)

    role_counts.plot(kind="bar", ax=axes[1, 0], color=["#4C78A8", "#E45756", "#72B7B2"])
    axes[1, 0].set_ylabel("Unique molecules")
    axes[1, 0].set_title("Unique molecule flags by role")
    axes[1, 0].tick_params(axis="x", rotation=0)

    if collision_sizes.empty:
        axes[1, 1].bar(["No collision"], [0], color="#B279A2")
    else:
        collision_sizes.head(20).plot(kind="bar", ax=axes[1, 1], color="#B279A2")
    axes[1, 1].set_ylabel("Unique isomeric forms")
    axes[1, 1].set_title("Largest component collision groups")
    axes[1, 1].tick_params(axis="x", rotation=75)

    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def analyze_isomerism(input_csv: Path, analysis_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    analysis_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(input_csv)
    validate_columns(df)
    df = df.copy()
    df["Solute_SMILES"] = df["Solute_SMILES"].map(safe_text)
    df["Solvent_SMILES"] = df["Solvent_SMILES"].map(safe_text)

    solute_table = build_molecule_table(df, "Solute_SMILES", "solute")
    solvent_table = build_molecule_table(df, "Solvent_SMILES", "solvent")
    molecule_table = pd.concat([solute_table, solvent_table], ignore_index=True)
    molecule_table, collision_details = add_collision_flags(molecule_table)
    annotated = annotate_dataset(df, molecule_table)
    pair_collision_details = build_pair_collision_groups(annotated)
    isomer_subset = annotated[annotated["isomer_subset_flag"]].copy()

    annotated.to_csv(analysis_dir / "all_merged_with_isomer_flags.csv", index=False)
    isomer_subset.to_csv(analysis_dir / "isomer_subset.csv", index=False)
    molecule_table.to_csv(analysis_dir / "molecule_isomerism_table.csv", index=False)
    collision_details.to_csv(analysis_dir / "isomer_collision_groups.csv", index=False)
    pair_collision_details.to_csv(analysis_dir / "pair_isomer_collision_groups.csv", index=False)
    write_summary(df, annotated, molecule_table, collision_details, pair_collision_details, analysis_dir, input_csv)
    plot_isomerism_statistics(annotated, molecule_table, collision_details, analysis_dir / "isomerism_statistics.png")
    return annotated, isomer_subset


def split_pair_grouped_folds(
    df: pd.DataFrame,
    output_dir: Path,
    dataset_name: str,
    n_folds: int,
    seed: int,
    valid_size: float,
) -> pd.DataFrame:
    if n_folds < 2:
        raise ValueError("--folds must be at least 2")
    output_dir.mkdir(parents=True, exist_ok=True)
    work = df.reset_index(drop=True).copy()
    validate_columns(work)
    work["pair_key"] = work[PAIR_COLUMNS].astype(str).agg("||".join, axis=1)
    unique_pairs = pd.Series(work["pair_key"].unique())
    if len(unique_pairs) < n_folds:
        raise ValueError(f"{dataset_name} has only {len(unique_pairs)} unique pairs; cannot create {n_folds} folds.")

    pair_kfold = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    fold_rows = []
    for fold_idx, (train_pair_idx, test_pair_idx) in enumerate(pair_kfold.split(unique_pairs), start=1):
        train_pairs = set(unique_pairs.iloc[train_pair_idx])
        test_pairs = set(unique_pairs.iloc[test_pair_idx])
        train_valid_df = work[work["pair_key"].isin(train_pairs)].copy()
        test_df = work[work["pair_key"].isin(test_pairs)].copy()

        groups = train_valid_df["pair_key"].to_numpy()
        if train_valid_df["pair_key"].nunique() >= 2 and 0.0 < valid_size < 1.0:
            splitter = GroupShuffleSplit(n_splits=1, test_size=valid_size, random_state=seed + fold_idx)
            train_idx, valid_idx = next(splitter.split(train_valid_df, groups=groups))
            train_df = train_valid_df.iloc[train_idx].copy()
            valid_df = train_valid_df.iloc[valid_idx].copy()
        else:
            train_df = train_valid_df.copy()
            valid_df = train_valid_df.iloc[0:0].copy()

        split_map = {"train": train_df, "valid": valid_df, "test": test_df}
        for split_name, split_df in split_map.items():
            split_df = split_df.drop(columns=["pair_key"], errors="ignore").reset_index(drop=True)
            split_path = output_dir / f"{dataset_name}_fold{fold_idx:02d}_{split_name}.csv"
            split_df.to_csv(split_path, index=False)
            fold_rows.append(
                {
                    "dataset": dataset_name,
                    "fold": fold_idx,
                    "split": split_name,
                    "path": str(split_path),
                    "samples": int(len(split_df)),
                    "unique_pairs": int(split_df.groupby(PAIR_COLUMNS, dropna=False).ngroups) if len(split_df) else 0,
                    "unique_solute": int(split_df["Solute_SMILES"].nunique()) if len(split_df) else 0,
                    "unique_solvent": int(split_df["Solvent_SMILES"].nunique()) if len(split_df) else 0,
                    "log_gamma_mean": float(split_df["log-gamma"].mean()) if len(split_df) else math.nan,
                    "log_gamma_std": float(split_df["log-gamma"].std(ddof=0)) if len(split_df) else math.nan,
                }
            )

    split_summary = pd.DataFrame(fold_rows)
    split_summary.to_csv(output_dir / f"{dataset_name}_fold_summary.csv", index=False)
    return split_summary


def metric_json_exists(run_dir: Path, test_stem: str) -> Path | None:
    matches = sorted(run_dir.glob(f"*_{test_stem}_metrics.json"))
    return matches[0] if matches else None


def build_train_command(
    args: argparse.Namespace,
    extra_train_args: list[str],
    train_path: Path,
    valid_path: Path,
    test_path: Path,
    run_dir: Path,
    cache_dir: Path,
    pair_cache_dir: Path,
    model_name: str,
    cache_model_name: str,
    topology_only: bool,
    seed: int,
) -> list[str]:
    cmd = [
        args.python,
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
        cache_model_name,
        "--n-epochs",
        str(args.epochs),
        "--early-stopping-patience",
        str(args.early_stopping_patience),
        "--hidden-dim",
        str(args.hidden_dim),
        "--batch-size",
        str(args.batch_size),
        "--checkpoint-interval",
        str(args.checkpoint_interval),
        "--seed",
        str(seed),
    ]
    if args.train_num_workers is not None:
        cmd.extend(["--train-num-workers", str(args.train_num_workers)])
    if args.valid_num_workers is not None:
        cmd.extend(["--valid-num-workers", str(args.valid_num_workers)])
    if args.quiet_progress:
        cmd.append("--quiet-progress")
    if args.resume:
        cmd.append("--resume")
    if topology_only:
        cmd.append("--topology-only")
    cmd.extend(extra_train_args)
    return cmd


def run_training_grid(args: argparse.Namespace, extra_train_args: list[str], output_dir: Path) -> pd.DataFrame:
    folds_dir = output_dir / "folds"
    runs_dir = output_dir / "runs"
    cache_root = Path(args.cache_dir) if args.cache_dir else output_dir / "cache"
    if not cache_root.is_absolute():
        cache_root = PROJECT_ROOT / cache_root
    pair_cache_dir = Path(args.pair_cache_dir) if args.pair_cache_dir else cache_root / "global_pair_graphs"
    if not pair_cache_dir.is_absolute():
        pair_cache_dir = PROJECT_ROOT / pair_cache_dir
    pair_cache_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    experiments = [
        ("full_pgssi", "Full PGSSI", "PGSSI_CV_Full", False),
        ("topology_only", "2D/topology baseline", "PGSSI_CV_Topology", True),
    ]
    datasets = ["full", "isomer_subset"]

    for dataset_name in datasets:
        for fold in range(1, args.folds + 1):
            train_path = folds_dir / f"{dataset_name}_fold{fold:02d}_train.csv"
            valid_path = folds_dir / f"{dataset_name}_fold{fold:02d}_valid.csv"
            test_path = folds_dir / f"{dataset_name}_fold{fold:02d}_test.csv"
            if not train_path.exists() or not valid_path.exists() or not test_path.exists():
                raise FileNotFoundError(f"Missing split files for {dataset_name} fold {fold:02d}")

            for model_key, model_label, model_name, topology_only in experiments:
                run_dir = runs_dir / dataset_name / model_key / f"fold{fold:02d}"
                # Keep the cache path/model key compatible with older full-PGSSI
                # runs, then reuse that same PairData cache for topology-only.
                cache_dir = cache_root / dataset_name / "full_pgssi" / f"fold{fold:02d}"
                cache_model_name = "PGSSI_CV_Full"
                run_dir.mkdir(parents=True, exist_ok=True)
                cache_dir.mkdir(parents=True, exist_ok=True)
                metrics_path = metric_json_exists(run_dir, test_path.stem)
                if metrics_path is None or args.force_train:
                    cmd = build_train_command(
                        args=args,
                        extra_train_args=extra_train_args,
                        train_path=train_path,
                        valid_path=valid_path,
                        test_path=test_path,
                        run_dir=run_dir,
                        cache_dir=cache_dir,
                        pair_cache_dir=pair_cache_dir,
                        model_name=model_name,
                        cache_model_name=cache_model_name,
                        topology_only=topology_only,
                        seed=args.seed + fold,
                    )
                    print(f"\n=== Running {dataset_name} | {model_label} | fold {fold:02d} ===", flush=True)
                    print(" ".join(cmd), flush=True)
                    subprocess.run(cmd, check=True, cwd=PROJECT_ROOT)
                    metrics_path = metric_json_exists(run_dir, test_path.stem)
                else:
                    print(f"\n=== Reusing {dataset_name} | {model_label} | fold {fold:02d} ===", flush=True)

                if metrics_path is None:
                    raise FileNotFoundError(f"Metrics JSON not found in {run_dir}")
                with open(metrics_path, "r", encoding="utf-8") as fh:
                    metrics = json.load(fh)
                rows.append(
                    {
                        "dataset": dataset_name,
                        "model": model_label,
                        "model_key": model_key,
                        "fold": fold,
                        "mae": float(metrics.get("mae", math.nan)),
                        "rmse": float(metrics.get("rmse", math.nan)),
                        "r2": float(metrics.get("r2", math.nan)),
                        "num_samples_total": int(metrics.get("num_samples_total", 0)),
                        "num_samples_predicted": int(metrics.get("num_samples_predicted", 0)),
                        "num_samples_skipped": int(metrics.get("num_samples_skipped", 0)),
                        "metrics_file": str(metrics_path),
                        "run_dir": str(run_dir),
                    }
                )
    fold_results = pd.DataFrame(rows)
    fold_results.to_csv(output_dir / "results_by_fold.csv", index=False)
    return fold_results


def format_mean_std(mean: float, std: float, digits: int = 4) -> str:
    if pd.isna(mean):
        return ""
    return f"{mean:.{digits}f} +/- {std:.{digits}f}"


def latex_mean_std(mean: float, std: float, digits: int = 4) -> str:
    if pd.isna(mean):
        return ""
    return f"${mean:.{digits}f} \\pm {std:.{digits}f}$"


def summarize_results(fold_results: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    if fold_results.empty:
        return pd.DataFrame()
    grouped = (
        fold_results.groupby(["dataset", "model"], dropna=False)
        .agg(
            folds=("fold", "nunique"),
            mae_mean=("mae", "mean"),
            mae_std=("mae", "std"),
            rmse_mean=("rmse", "mean"),
            rmse_std=("rmse", "std"),
            r2_mean=("r2", "mean"),
            r2_std=("r2", "std"),
            total_test_samples=("num_samples_total", "sum"),
            total_predicted_samples=("num_samples_predicted", "sum"),
            total_skipped_samples=("num_samples_skipped", "sum"),
        )
        .reset_index()
    )
    grouped["mae_mean_std"] = grouped.apply(lambda r: format_mean_std(r["mae_mean"], r["mae_std"]), axis=1)
    grouped["rmse_mean_std"] = grouped.apply(lambda r: format_mean_std(r["rmse_mean"], r["rmse_std"]), axis=1)
    grouped["r2_mean_std"] = grouped.apply(lambda r: format_mean_std(r["r2_mean"], r["r2_std"]), axis=1)
    grouped.to_csv(output_dir / "results.csv", index=False)

    md_cols = ["dataset", "model", "folds", "mae_mean_std", "rmse_mean_std", "r2_mean_std", "total_test_samples"]
    md_text = grouped[md_cols].to_markdown(index=False)
    (output_dir / "results.md").write_text(
        "# 5-fold cross-validation results\n\n"
        "Metrics are computed on log-gamma. Splits are grouped by solute-solvent pair.\n\n"
        + md_text
        + "\n",
        encoding="utf-8",
    )

    latex_lines = [
        "\\begin{table}[htbp]",
        "\\centering",
        "\\caption{Pair-grouped 5-fold cross-validation on log-gamma.}",
        "\\label{tab:isomerism_cv}",
        "\\begin{tabular}{llccc}",
        "\\toprule",
        "Dataset & Model & MAE & RMSE & $R^2$ \\\\",
        "\\midrule",
    ]
    for _, row in grouped.iterrows():
        latex_lines.append(
            f"{row['dataset']} & {row['model']} & "
            f"{latex_mean_std(row['mae_mean'], row['mae_std'])} & "
            f"{latex_mean_std(row['rmse_mean'], row['rmse_std'])} & "
            f"{latex_mean_std(row['r2_mean'], row['r2_std'])} \\\\"
        )
    latex_lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{table}", ""])
    latex_table = "\n".join(latex_lines)
    (output_dir / "results_table.tex").write_text(latex_table, encoding="utf-8")
    return grouped


def load_existing_analysis(analysis_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    annotated_path = analysis_dir / "all_merged_with_isomer_flags.csv"
    subset_path = analysis_dir / "isomer_subset.csv"
    if not annotated_path.exists() or not subset_path.exists():
        raise FileNotFoundError("--skip-analysis requested, but existing analysis CSV files were not found.")
    return pd.read_csv(annotated_path), pd.read_csv(subset_path)


def main() -> None:
    args, extra_train_args = parse_args()
    input_csv = Path(args.input_csv)
    if not input_csv.is_absolute():
        input_csv = PROJECT_ROOT / input_csv
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    analysis_dir = output_dir / "analysis"
    folds_dir = output_dir / "folds"

    if args.skip_analysis:
        annotated, isomer_subset = load_existing_analysis(analysis_dir)
    else:
        annotated, isomer_subset = analyze_isomerism(input_csv, analysis_dir)

    full_df = pd.read_csv(input_csv)
    validate_columns(full_df)
    full_df["Solute_SMILES"] = full_df["Solute_SMILES"].map(safe_text)
    full_df["Solvent_SMILES"] = full_df["Solvent_SMILES"].map(safe_text)

    split_summaries = [
        split_pair_grouped_folds(full_df, folds_dir, "full", args.folds, args.seed, args.valid_size),
        split_pair_grouped_folds(isomer_subset, folds_dir, "isomer_subset", args.folds, args.seed, args.valid_size),
    ]
    pd.concat(split_summaries, ignore_index=True).to_csv(folds_dir / "fold_summary.csv", index=False)

    if args.prepare_only:
        print(f"Prepared isomerism analysis and fold files under: {output_dir}", flush=True)
        return

    fold_results = run_training_grid(args, extra_train_args, output_dir)
    summary = summarize_results(fold_results, output_dir)
    print("\n=== Cross-validation summary ===", flush=True)
    print(summary.to_string(index=False), flush=True)
    print(f"\nSaved results to: {output_dir / 'results.csv'}", flush=True)


if __name__ == "__main__":
    main()
