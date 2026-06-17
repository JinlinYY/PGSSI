"""Prebuild reusable PGSSI 3D pair graph cache.

The cache key is the unique ``Solvent_SMILES + Solute_SMILES`` pair. The cached
object intentionally excludes row-level targets such as temperature, log-gamma,
and sample_index; those are attached by PGSSI_data.build_pair_dataset when a
specific train/valid/test CSV is loaded.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT = PROJECT_ROOT / "dataset" / "all" / "all_merged.csv"
DEFAULT_PAIR_CACHE = SCRIPT_DIR / "cache" / "global_pair_graphs"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.PGSSI.PGSSI_data import GEOM_CACHE_VERSION, prebuild_pair_graph_cache


REQUIRED_COLUMNS = ["Solvent_SMILES", "Solute_SMILES"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prebuild global PGSSI 3D pair graph cache.")
    parser.add_argument("--input-csv", type=str, default=str(DEFAULT_INPUT))
    parser.add_argument("--pair-cache-dir", type=str, default=str(DEFAULT_PAIR_CACHE))
    parser.add_argument("--summary-csv", type=str, default=None)
    parser.add_argument("--summary-json", type=str, default=None)
    parser.add_argument("--limit", type=int, default=None, help="Optional smoke-test limit on unique pairs.")
    parser.add_argument("--force", action="store_true", help="Rebuild existing pair cache files.")
    return parser.parse_args()


def resolve_path(path_str: str) -> Path:
    path = Path(path_str)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path


def validate_columns(df: pd.DataFrame) -> None:
    missing = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def main() -> None:
    args = parse_args()
    input_csv = resolve_path(args.input_csv)
    pair_cache_dir = resolve_path(args.pair_cache_dir)
    pair_cache_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_csv)
    validate_columns(df)
    unique_pair_count = int(df[REQUIRED_COLUMNS].drop_duplicates().shape[0])
    print(f"Input CSV: {input_csv}", flush=True)
    print(f"Rows: {len(df)} | unique solvent-solute pairs: {unique_pair_count}", flush=True)
    print(f"Pair graph cache: {pair_cache_dir}", flush=True)
    print(f"Geometry cache version: {GEOM_CACHE_VERSION}", flush=True)

    summary = prebuild_pair_graph_cache(
        df,
        pair_cache_dir=pair_cache_dir,
        limit=args.limit,
        force=args.force,
    )
    records = summary.pop("records")

    summary_csv = resolve_path(args.summary_csv) if args.summary_csv else pair_cache_dir / "prebuild_summary.csv"
    summary_json = resolve_path(args.summary_json) if args.summary_json else pair_cache_dir / "prebuild_summary.json"
    summary_csv.parent.mkdir(parents=True, exist_ok=True)
    summary_json.parent.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(records).to_csv(summary_csv, index=False)
    with open(summary_json, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, ensure_ascii=False)

    print(
        "Done | "
        f"total={summary['total_unique_pairs']} | "
        f"built={summary['built']} | "
        f"reused={summary['reused']} | "
        f"skipped={summary['skipped']}",
        flush=True,
    )
    print(f"Summary CSV: {summary_csv}", flush=True)
    print(f"Summary JSON: {summary_json}", flush=True)


if __name__ == "__main__":
    main()
