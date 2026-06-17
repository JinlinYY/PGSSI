"""Analyze PGSSI dataset composition and data quality.

This script reads the merged PGSSI dataset, summarizes composition and value
coverage, checks duplicate/conflicting records, generates figures, and writes a
Chinese markdown report based only on computed values.
"""

from __future__ import annotations

import argparse
import re
import textwrap
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd


matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = PROJECT_ROOT / "dataset" / "all" / "all_merged.csv"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent

REQUIRED_COLUMNS = ["Solute_SMILES", "Solvent_SMILES", "T", "log-gamma", "source_file"]
PAIR_COLUMNS = ["Solute_SMILES", "Solvent_SMILES"]
CONDITION_COLUMNS = ["Solute_SMILES", "Solvent_SMILES", "T"]
WATER_SMILES = {"O", "[OH2]", "[H]O[H]"}
HBOND_RICH_PATTERN = re.compile(r"[ONFon]")

# Final double-column figure size. Keeping the exported figure at its intended
# LaTeX width prevents an oversized canvas from shrinking the text at insertion.
FIGURE_WIDTH_IN = 7.2
FIGURE_HEIGHT_IN = 5.0
BASE_FONT_SIZE_PT = 8.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze PGSSI dataset composition and quality.")
    parser.add_argument("--input-csv", type=str, default=str(DEFAULT_INPUT), help="Input merged dataset CSV.")
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT_DIR), help="Output directory.")
    parser.add_argument("--hist-bins", type=int, default=30, help="Number of histogram bins for numeric columns.")
    parser.add_argument("--top-n", type=int, default=20, help="Number of top source/frequency entries to show in plots.")
    return parser.parse_args()


def validate_columns(df: pd.DataFrame) -> None:
    missing = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def load_dataset(input_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(input_csv)
    validate_columns(df)
    df = df[REQUIRED_COLUMNS].copy()
    df["T"] = pd.to_numeric(df["T"], errors="coerce")
    df["log-gamma"] = pd.to_numeric(df["log-gamma"], errors="coerce")
    return df


def format_number(value: float | int | str | None, digits: int = 6) -> str:
    if value is None or pd.isna(value):
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    return f"{float(value):.{digits}g}"


def pct(count: int, total: int) -> float:
    return float(count / total * 100.0) if total else 0.0


def add_summary_row(
    rows: list[dict[str, object]],
    section: str,
    metric: str,
    value: object,
    percentage: float | None = None,
    notes: str = "",
) -> None:
    rows.append(
        {
            "section": section,
            "metric": metric,
            "value": value,
            "percentage": percentage,
            "notes": notes,
        }
    )


def add_numeric_summary(rows: list[dict[str, object]], df: pd.DataFrame, column: str, section: str) -> None:
    values = df[column].dropna()
    add_summary_row(rows, section, f"{column}_count", int(values.count()))
    add_summary_row(rows, section, f"{column}_missing", int(df[column].isna().sum()))
    add_summary_row(rows, section, f"{column}_min", values.min())
    add_summary_row(rows, section, f"{column}_q25", values.quantile(0.25))
    add_summary_row(rows, section, f"{column}_median", values.median())
    add_summary_row(rows, section, f"{column}_mean", values.mean())
    add_summary_row(rows, section, f"{column}_q75", values.quantile(0.75))
    add_summary_row(rows, section, f"{column}_max", values.max())
    add_summary_row(rows, section, f"{column}_std", values.std(ddof=0))


def add_histogram_rows(
    rows: list[dict[str, object]],
    df: pd.DataFrame,
    column: str,
    section: str,
    bins: int,
) -> None:
    values = df[column].dropna()
    if values.empty:
        return
    counts, edges = np.histogram(values.to_numpy(dtype=float), bins=bins)
    total = int(counts.sum())
    for idx, count in enumerate(counts):
        label = f"[{edges[idx]:.6g}, {edges[idx + 1]:.6g})"
        if idx == len(counts) - 1:
            label = f"[{edges[idx]:.6g}, {edges[idx + 1]:.6g}]"
        add_summary_row(rows, section, label, int(count), pct(int(count), total), notes=f"{column} histogram bin")


def has_isomer_marker(smiles: object) -> bool:
    text = str(smiles)
    return "@" in text or "/" in text or "\\" in text or re.search(r"\[[0-9]+[A-Za-z]", text) is not None


def strip_isomer_markers(smiles: object) -> str:
    text = str(smiles)
    text = text.replace("@", "").replace("/", "").replace("\\", "")
    return re.sub(r"\[([0-9]+)([A-Za-z])", r"[\2", text)


def has_hbond_rich_marker(smiles: object) -> bool:
    return HBOND_RICH_PATTERN.search(str(smiles)) is not None


def compute_extension_statistics(df: pd.DataFrame) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    total = len(df)
    solute_isomer_mask = df["Solute_SMILES"].map(has_isomer_marker)
    solvent_isomer_mask = df["Solvent_SMILES"].map(has_isomer_marker)
    any_isomer_mask = solute_isomer_mask | solvent_isomer_mask
    solute_signature_groups = (
        df[["Solute_SMILES"]]
        .drop_duplicates()
        .assign(nonisomeric_signature=lambda x: x["Solute_SMILES"].map(strip_isomer_markers))
        .groupby("nonisomeric_signature")["Solute_SMILES"]
        .nunique()
    )
    solvent_signature_groups = (
        df[["Solvent_SMILES"]]
        .drop_duplicates()
        .assign(nonisomeric_signature=lambda x: x["Solvent_SMILES"].map(strip_isomer_markers))
        .groupby("nonisomeric_signature")["Solvent_SMILES"]
        .nunique()
    )

    add_summary_row(
        rows,
        "isomer_statistics",
        "isomer_detection_method",
        "SMILES marker heuristic",
        notes="Counts SMILES containing stereochemical bond markers (@, /, \\) or isotope labels; no RDKit canonicalization is required.",
    )
    add_summary_row(rows, "isomer_statistics", "samples_with_solute_isomer_marker", int(solute_isomer_mask.sum()), pct(int(solute_isomer_mask.sum()), total))
    add_summary_row(rows, "isomer_statistics", "samples_with_solvent_isomer_marker", int(solvent_isomer_mask.sum()), pct(int(solvent_isomer_mask.sum()), total))
    add_summary_row(rows, "isomer_statistics", "samples_with_any_isomer_marker", int(any_isomer_mask.sum()), pct(int(any_isomer_mask.sum()), total))
    add_summary_row(rows, "isomer_statistics", "unique_solutes_with_isomer_marker", int(df.loc[solute_isomer_mask, "Solute_SMILES"].nunique()))
    add_summary_row(rows, "isomer_statistics", "unique_solvents_with_isomer_marker", int(df.loc[solvent_isomer_mask, "Solvent_SMILES"].nunique()))
    add_summary_row(rows, "isomer_statistics", "solute_nonisomeric_signature_collision_groups", int((solute_signature_groups > 1).sum()))
    add_summary_row(rows, "isomer_statistics", "solvent_nonisomeric_signature_collision_groups", int((solvent_signature_groups > 1).sum()))

    solute_as_text = df["Solute_SMILES"].astype(str)
    solvent_as_text = df["Solvent_SMILES"].astype(str)
    water_solute_mask = solute_as_text.isin(WATER_SMILES)
    water_solvent_mask = solvent_as_text.isin(WATER_SMILES)
    water_component_mask = water_solute_mask | water_solvent_mask
    nonwater_system_mask = ~water_component_mask
    add_summary_row(
        rows,
        "water_nonwater_statistics",
        "water_definition",
        "; ".join(sorted(WATER_SMILES)),
        notes="Exact solute/solvent SMILES matches used for the current water/non-water composition statistics.",
    )
    add_summary_row(rows, "water_nonwater_statistics", "water_as_solvent_samples", int(water_solvent_mask.sum()), pct(int(water_solvent_mask.sum()), total))
    add_summary_row(rows, "water_nonwater_statistics", "water_as_solute_samples", int(water_solute_mask.sum()), pct(int(water_solute_mask.sum()), total))
    add_summary_row(rows, "water_nonwater_statistics", "samples_with_water_component", int(water_component_mask.sum()), pct(int(water_component_mask.sum()), total))
    add_summary_row(
        rows,
        "water_nonwater_statistics",
        "samples_without_water_component",
        int(nonwater_system_mask.sum()),
        pct(int(nonwater_system_mask.sum()), total),
    )
    add_summary_row(rows, "water_nonwater_statistics", "unique_water_solvents", int(solvent_as_text[water_solvent_mask].nunique()))
    add_summary_row(rows, "water_nonwater_statistics", "unique_nonwater_solvents", int(solvent_as_text[~water_solvent_mask].nunique()))
    add_summary_row(rows, "water_nonwater_statistics", "unique_pairs_with_water_component", int(df.loc[water_component_mask].groupby(PAIR_COLUMNS, dropna=False).ngroups))
    add_summary_row(rows, "water_nonwater_statistics", "unique_pairs_without_water_component", int(df.loc[nonwater_system_mask].groupby(PAIR_COLUMNS, dropna=False).ngroups))

    solute_hbond_mask = solute_as_text.map(has_hbond_rich_marker)
    solvent_hbond_mask = solvent_as_text.map(has_hbond_rich_marker)
    hbond_rich_mask = solute_hbond_mask & solvent_hbond_mask
    add_summary_row(
        rows,
        "hbond_rich_statistics",
        "hbond_rich_definition",
        "Both solute and solvent contain O/N/F or aromatic o/n in SMILES",
    )
    add_summary_row(rows, "hbond_rich_statistics", "hbond_rich_samples", int(hbond_rich_mask.sum()), pct(int(hbond_rich_mask.sum()), total))
    add_summary_row(rows, "hbond_rich_statistics", "non_hbond_rich_samples", int((~hbond_rich_mask).sum()), pct(int((~hbond_rich_mask).sum()), total))
    add_summary_row(rows, "hbond_rich_statistics", "unique_hbond_rich_pairs", int(df.loc[hbond_rich_mask].groupby(PAIR_COLUMNS, dropna=False).ngroups))
    return rows


def add_quality_summary(rows: list[dict[str, object]], issues: pd.DataFrame, total: int) -> None:
    issue_labels = {
        "complete_duplicate": "完全重复记录",
        "same_condition_different_log_gamma": "同条件 log-gamma 冲突",
        "cross_source_same_condition_different_log_gamma": "跨来源同条件 log-gamma 冲突",
    }
    for issue_type, label in issue_labels.items():
        row_count = int((issues["issue_type"] == issue_type).sum()) if not issues.empty else 0
        group_count = int(issues.loc[issues["issue_type"] == issue_type, "issue_group_id"].nunique()) if row_count else 0
        add_summary_row(rows, "data_quality", f"{issue_type}_rows", row_count, pct(row_count, total), notes=label)
        add_summary_row(rows, "data_quality", f"{issue_type}_groups", group_count, notes=label)


def build_composition_summary(df: pd.DataFrame, issues: pd.DataFrame, bins: int) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    total = len(df)
    pair_count = df.groupby(PAIR_COLUMNS, dropna=False).ngroups

    add_summary_row(rows, "overall", "total_samples", total)
    add_summary_row(rows, "overall", "unique_solute", int(df["Solute_SMILES"].nunique(dropna=False)))
    add_summary_row(rows, "overall", "unique_solvent", int(df["Solvent_SMILES"].nunique(dropna=False)))
    add_summary_row(rows, "overall", "unique_solute_solvent_pair", int(pair_count))

    add_numeric_summary(rows, df, "T", "temperature")
    add_numeric_summary(rows, df, "log-gamma", "log_gamma")
    add_histogram_rows(rows, df, "T", "temperature_distribution", bins)
    add_histogram_rows(rows, df, "log-gamma", "log_gamma_distribution", bins)

    source_counts = df["source_file"].fillna("<missing>").value_counts(dropna=False)
    for source, count in source_counts.items():
        add_summary_row(rows, "source_file_distribution", str(source), int(count), pct(int(count), total))

    solute_counts = df["Solute_SMILES"].fillna("<missing>").value_counts(dropna=False)
    solvent_counts = df["Solvent_SMILES"].fillna("<missing>").value_counts(dropna=False)
    for rank, (solute, count) in enumerate(solute_counts.head(20).items(), start=1):
        add_summary_row(rows, "top_solute_frequency", str(solute), int(count), pct(int(count), total), notes=f"rank={rank}")
    for rank, (solvent, count) in enumerate(solvent_counts.head(20).items(), start=1):
        add_summary_row(rows, "top_solvent_frequency", str(solvent), int(count), pct(int(count), total), notes=f"rank={rank}")

    add_quality_summary(rows, issues, total)
    rows.extend(compute_extension_statistics(df))
    summary = pd.DataFrame(rows)
    return summary


def assign_group_ids(df: pd.DataFrame, columns: list[str], prefix: str) -> pd.Series:
    keys = df[columns].astype(str).agg("||".join, axis=1)
    codes, _ = pd.factorize(keys, sort=True)
    return pd.Series([f"{prefix}_{code + 1:06d}" for code in codes], index=df.index)


def build_duplicate_conflicts(df: pd.DataFrame) -> pd.DataFrame:
    records: list[pd.DataFrame] = []

    complete_mask = df.duplicated(keep=False)
    if complete_mask.any():
        complete_df = df.loc[complete_mask].copy()
        complete_df.insert(0, "issue_type", "complete_duplicate")
        complete_df.insert(1, "issue_group_id", assign_group_ids(complete_df, REQUIRED_COLUMNS, "dup"))
        complete_df["group_size"] = complete_df.groupby("issue_group_id")["issue_group_id"].transform("size")
        complete_df["unique_log_gamma_values"] = 1
        complete_df["source_file_count"] = complete_df.groupby("issue_group_id")["source_file"].transform("nunique")
        complete_df["log_gamma_range"] = 0.0
        records.append(complete_df)

    condition_groups = df.groupby(CONDITION_COLUMNS, dropna=False)
    condition_stats = condition_groups.agg(
        group_size=("log-gamma", "size"),
        unique_log_gamma_values=("log-gamma", "nunique"),
        source_file_count=("source_file", "nunique"),
        log_gamma_min=("log-gamma", "min"),
        log_gamma_max=("log-gamma", "max"),
    )
    condition_stats["log_gamma_range"] = condition_stats["log_gamma_max"] - condition_stats["log_gamma_min"]

    value_conflict_keys = condition_stats[condition_stats["unique_log_gamma_values"] > 1].index
    if len(value_conflict_keys) > 0:
        conflict_df = df.set_index(CONDITION_COLUMNS).loc[value_conflict_keys].reset_index()
        conflict_df.insert(0, "issue_type", "same_condition_different_log_gamma")
        conflict_df.insert(1, "issue_group_id", assign_group_ids(conflict_df, CONDITION_COLUMNS, "conflict"))
        enriched = conflict_df.join(condition_stats, on=CONDITION_COLUMNS, rsuffix="_stats")
        records.append(enriched.drop(columns=["log_gamma_min", "log_gamma_max"]))

    cross_source_conflict_keys = condition_stats[
        (condition_stats["source_file_count"] > 1) & (condition_stats["unique_log_gamma_values"] > 1)
    ].index
    if len(cross_source_conflict_keys) > 0:
        cross_df = df.set_index(CONDITION_COLUMNS).loc[cross_source_conflict_keys].reset_index()
        cross_df.insert(0, "issue_type", "cross_source_same_condition_different_log_gamma")
        cross_df.insert(1, "issue_group_id", assign_group_ids(cross_df, CONDITION_COLUMNS, "cross_source_conflict"))
        enriched = cross_df.join(condition_stats, on=CONDITION_COLUMNS, rsuffix="_stats")
        records.append(enriched.drop(columns=["log_gamma_min", "log_gamma_max"]))

    if not records:
        return pd.DataFrame(
            columns=[
                "issue_type",
                "issue_group_id",
                "group_size",
                "unique_log_gamma_values",
                "source_file_count",
                "log_gamma_range",
                *REQUIRED_COLUMNS,
            ]
        )

    result = pd.concat(records, ignore_index=True, sort=False)
    metadata = ["issue_type", "issue_group_id", "group_size", "unique_log_gamma_values", "source_file_count", "log_gamma_range"]
    return result[metadata + REQUIRED_COLUMNS]


def configure_plot_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": BASE_FONT_SIZE_PT,
            "axes.titlesize": 9,
            "axes.titleweight": "semibold",
            "axes.labelsize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "axes.edgecolor": "#222222",
            "axes.linewidth": 0.8,
            "axes.grid": True,
            "grid.color": "#dddddd",
            "grid.linestyle": "--",
            "grid.linewidth": 0.5,
            "legend.frameon": False,
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
            "savefig.dpi": 600,
        }
    )


def format_source_label(source: object) -> str:
    label = Path(str(source)).stem.replace("_", " ")
    return "\n".join(textwrap.wrap(label, width=10))


def plot_dataset_composition(df: pd.DataFrame, issues: pd.DataFrame, output_dir: Path, top_n: int, bins: int) -> Path:
    configure_plot_style()
    figure_path = output_dir / "dataset_composition_figures.png"
    pdf_path = output_dir / "dataset_composition_figures.pdf"
    svg_path = output_dir / "dataset_composition_figures.svg"
    source_counts = df["source_file"].fillna("<missing>").value_counts().head(top_n)

    fig, axes = plt.subplots(
        2,
        3,
        figsize=(FIGURE_WIDTH_IN, FIGURE_HEIGHT_IN),
        layout="constrained",
    )

    source_labels = [format_source_label(source) for source in source_counts.index]
    axes[0, 0].bar(source_labels, source_counts.values, color="#3b6f8f")
    axes[0, 0].set_title("Samples by source_file")
    axes[0, 0].set_ylabel("Samples")
    axes[0, 0].tick_params(axis="x", labelsize=6.5)
    axes[0, 0].margins(y=0.12)
    axes[0, 0].bar_label(axes[0, 0].containers[0], fontsize=7, padding=2)

    axes[0, 1].hist(df["log-gamma"].dropna(), bins=bins, color="#bf6f59", edgecolor="white")
    axes[0, 1].set_title("log-gamma distribution")
    axes[0, 1].set_xlabel("log-gamma")
    axes[0, 1].set_ylabel("Samples")

    axes[0, 2].hist(df["T"].dropna(), bins=bins, color="#4d8e94", edgecolor="white")
    axes[0, 2].set_title("Temperature distribution")
    axes[0, 2].set_xlabel("T (Celsius)")
    axes[0, 2].set_ylabel("Samples")

    solute_isomer_mask = df["Solute_SMILES"].map(has_isomer_marker)
    solvent_isomer_mask = df["Solvent_SMILES"].map(has_isomer_marker)
    isomer_values = [
        int(solute_isomer_mask.sum()),
        int(solvent_isomer_mask.sum()),
        int((solute_isomer_mask | solvent_isomer_mask).sum()),
    ]
    axes[1, 0].bar(["Solute", "Solvent", "Any"], isomer_values, color=["#2f5c7a", "#c5894b", "#7e9a46"])
    axes[1, 0].set_title("Samples with isomer markers")
    axes[1, 0].set_ylabel("Samples")
    axes[1, 0].margins(y=0.12)
    axes[1, 0].bar_label(axes[1, 0].containers[0], fontsize=7, padding=2)

    solute_hbond_mask = df["Solute_SMILES"].astype(str).map(has_hbond_rich_marker)
    solvent_hbond_mask = df["Solvent_SMILES"].astype(str).map(has_hbond_rich_marker)
    hbond_rich_mask = solute_hbond_mask & solvent_hbond_mask
    hbond_values = [int(hbond_rich_mask.sum()), int((~hbond_rich_mask).sum())]
    axes[1, 1].bar(["H-bond\nrich", "Other"], hbond_values, color=["#6a9f58", "#b88a44"])
    axes[1, 1].set_title("H-bond-rich systems")
    axes[1, 1].set_ylabel("Samples")
    axes[1, 1].margins(y=0.12)
    axes[1, 1].bar_label(axes[1, 1].containers[0], fontsize=7, padding=2)

    solute_as_text = df["Solute_SMILES"].astype(str)
    solvent_as_text = df["Solvent_SMILES"].astype(str)
    water_solute_mask = solute_as_text.isin(WATER_SMILES)
    water_solvent_mask = solvent_as_text.isin(WATER_SMILES)
    water_component_mask = water_solute_mask | water_solvent_mask
    water_values = [
        int(water_solvent_mask.sum()),
        int(water_solute_mask.sum()),
        int(water_component_mask.sum()),
        int((~water_component_mask).sum()),
    ]
    water_labels = ["Water\nsolvent", "Water\nsolute", "Any\nwater", "No\nwater"]
    axes[1, 2].bar(water_labels, water_values, color=["#4d8e94", "#7e9a46", "#3b6f8f", "#c5894b"])
    axes[1, 2].set_title("Water/non-water systems")
    axes[1, 2].set_ylabel("Samples")
    axes[1, 2].margins(y=0.12)
    axes[1, 2].bar_label(axes[1, 2].containers[0], fontsize=7, padding=2)

    for panel_label, ax in zip("abcdef", axes.flat):
        ax.text(
            -0.12,
            1.04,
            panel_label,
            transform=ax.transAxes,
            fontsize=10,
            fontweight="bold",
            ha="left",
            va="bottom",
        )

    fig.savefig(figure_path, dpi=600)
    fig.savefig(pdf_path)
    fig.savefig(svg_path)
    plt.close(fig)
    return figure_path


def summary_value(summary: pd.DataFrame, section: str, metric: str) -> object:
    match = summary[(summary["section"] == section) & (summary["metric"] == metric)]
    if match.empty:
        return ""
    return match.iloc[0]["value"]


def issue_count(issues: pd.DataFrame, issue_type: str) -> int:
    return int((issues["issue_type"] == issue_type).sum()) if not issues.empty else 0


def issue_group_count(issues: pd.DataFrame, issue_type: str) -> int:
    if issues.empty:
        return 0
    return int(issues.loc[issues["issue_type"] == issue_type, "issue_group_id"].nunique())


def source_markdown_table(df: pd.DataFrame) -> str:
    total = len(df)
    rows = ["| source_file | 样本数 | 占比 |", "| --- | ---: | ---: |"]
    for source, count in df["source_file"].fillna("<missing>").value_counts(dropna=False).items():
        rows.append(f"| `{source}` | {int(count)} | {pct(int(count), total):.2f}% |")
    return "\n".join(rows)


def publication_markdown_table(publication_table: pd.DataFrame) -> str:
    rows = [
        "| Category | Metric | Value | Percentage |",
        "| --- | --- | ---: | ---: |",
    ]
    for _, row in publication_table.iterrows():
        values = [str(row[column]).replace("|", r"\|") for column in ["category", "metric", "value", "percentage"]]
        rows.append(f"| {' | '.join(values)} |")
    return "\n".join(rows)


def latex_escape(text: object) -> str:
    value = str(text)
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    for old, new in replacements.items():
        value = value.replace(old, new)
    return value


def build_publication_table(df: pd.DataFrame, summary: pd.DataFrame, issues: pd.DataFrame) -> pd.DataFrame:
    total = int(summary_value(summary, "overall", "total_samples"))
    source_counts = df["source_file"].fillna("<missing>").value_counts(dropna=False)
    rows = [
        {"category": "Dataset scale", "metric": "Total samples", "value": total, "percentage": ""},
        {"category": "Dataset scale", "metric": "Unique solutes", "value": summary_value(summary, "overall", "unique_solute"), "percentage": ""},
        {"category": "Dataset scale", "metric": "Unique solvents", "value": summary_value(summary, "overall", "unique_solvent"), "percentage": ""},
        {"category": "Dataset scale", "metric": "Unique solute-solvent pairs", "value": summary_value(summary, "overall", "unique_solute_solvent_pair"), "percentage": ""},
        {
            "category": "Coverage",
            "metric": "Temperature range (Celsius)",
            "value": f"{format_number(summary_value(summary, 'temperature', 'T_min'))} to {format_number(summary_value(summary, 'temperature', 'T_max'))}",
            "percentage": "",
        },
        {
            "category": "Coverage",
            "metric": "log-gamma range",
            "value": f"{format_number(summary_value(summary, 'log_gamma', 'log-gamma_min'))} to {format_number(summary_value(summary, 'log_gamma', 'log-gamma_max'))}",
            "percentage": "",
        },
        {
            "category": "Data quality",
            "metric": "Complete duplicate rows / groups",
            "value": f"{issue_count(issues, 'complete_duplicate')} / {issue_group_count(issues, 'complete_duplicate')}",
            "percentage": f"{pct(issue_count(issues, 'complete_duplicate'), total):.2f}%",
        },
        {
            "category": "Data quality",
            "metric": "Same-condition conflict rows / groups",
            "value": f"{issue_count(issues, 'same_condition_different_log_gamma')} / {issue_group_count(issues, 'same_condition_different_log_gamma')}",
            "percentage": f"{pct(issue_count(issues, 'same_condition_different_log_gamma'), total):.2f}%",
        },
        {
            "category": "Data quality",
            "metric": "Cross-source conflict rows / groups",
            "value": f"{issue_count(issues, 'cross_source_same_condition_different_log_gamma')} / {issue_group_count(issues, 'cross_source_same_condition_different_log_gamma')}",
            "percentage": f"{pct(issue_count(issues, 'cross_source_same_condition_different_log_gamma'), total):.2f}%",
        },
        {
            "category": "Isomer statistics",
            "metric": "Rows with any isomer marker",
            "value": summary_value(summary, "isomer_statistics", "samples_with_any_isomer_marker"),
            "percentage": f"{float(summary_value(summary, 'isomer_statistics', 'samples_with_any_isomer_marker') or 0) / total * 100:.2f}%",
        },
        {
            "category": "Isomer statistics",
            "metric": "Unique solutes / solvents with markers",
            "value": f"{summary_value(summary, 'isomer_statistics', 'unique_solutes_with_isomer_marker')} / {summary_value(summary, 'isomer_statistics', 'unique_solvents_with_isomer_marker')}",
            "percentage": "",
        },
        {
            "category": "Water statistics",
            "metric": "Rows with water component",
            "value": summary_value(summary, "water_nonwater_statistics", "samples_with_water_component"),
            "percentage": f"{float(summary_value(summary, 'water_nonwater_statistics', 'samples_with_water_component') or 0) / total * 100:.2f}%",
        },
        {
            "category": "Water statistics",
            "metric": "Rows without water component",
            "value": summary_value(summary, "water_nonwater_statistics", "samples_without_water_component"),
            "percentage": f"{float(summary_value(summary, 'water_nonwater_statistics', 'samples_without_water_component') or 0) / total * 100:.2f}%",
        },
        {
            "category": "H-bond-rich statistics",
            "metric": "Rows in H-bond-rich subset",
            "value": summary_value(summary, "hbond_rich_statistics", "hbond_rich_samples"),
            "percentage": f"{float(summary_value(summary, 'hbond_rich_statistics', 'hbond_rich_samples') or 0) / total * 100:.2f}%",
        },
        {
            "category": "H-bond-rich statistics",
            "metric": "Rows outside H-bond-rich subset",
            "value": summary_value(summary, "hbond_rich_statistics", "non_hbond_rich_samples"),
            "percentage": f"{float(summary_value(summary, 'hbond_rich_statistics', 'non_hbond_rich_samples') or 0) / total * 100:.2f}%",
        },
    ]
    for source, count in source_counts.items():
        rows.append(
            {
                "category": "Source distribution",
                "metric": str(source),
                "value": int(count),
                "percentage": f"{pct(int(count), total):.2f}%",
            }
        )
    return pd.DataFrame(rows)


def write_latex_summary_table(publication_table: pd.DataFrame, output_dir: Path) -> tuple[Path, Path]:
    csv_path = output_dir / "dataset_summary_table.csv"
    tex_path = output_dir / "dataset_summary_table.tex"
    publication_table.to_csv(csv_path, index=False, encoding="utf-8-sig")

    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Dataset composition and quality summary.}",
        r"\label{tab:dataset_composition_quality}",
        r"\begin{tabular}{llll}",
        r"\hline",
        r"Category & Metric & Value & Percentage \\",
        r"\hline",
    ]
    for _, row in publication_table.iterrows():
        lines.append(
            f"{latex_escape(row['category'])} & {latex_escape(row['metric'])} & "
            f"{latex_escape(row['value'])} & {latex_escape(row['percentage'])} \\\\"
        )
    lines.extend([r"\hline", r"\end{tabular}", r"\end{table}", ""])
    tex_path.write_text("\n".join(lines), encoding="utf-8")
    return csv_path, tex_path


def write_markdown_report(
    df: pd.DataFrame,
    summary: pd.DataFrame,
    issues: pd.DataFrame,
    input_csv: Path,
    output_dir: Path,
    figure_path: Path,
    latex_table_path: Path,
    publication_table: pd.DataFrame,
) -> Path:
    report_path = output_dir / "dataset_composition.md"
    total = int(summary_value(summary, "overall", "total_samples"))
    unique_solute = int(summary_value(summary, "overall", "unique_solute"))
    unique_solvent = int(summary_value(summary, "overall", "unique_solvent"))
    unique_pair = int(summary_value(summary, "overall", "unique_solute_solvent_pair"))
    t_min = float(summary_value(summary, "temperature", "T_min"))
    t_max = float(summary_value(summary, "temperature", "T_max"))
    t_median = float(summary_value(summary, "temperature", "T_median"))
    lg_min = float(summary_value(summary, "log_gamma", "log-gamma_min"))
    lg_max = float(summary_value(summary, "log_gamma", "log-gamma_max"))
    lg_median = float(summary_value(summary, "log_gamma", "log-gamma_median"))

    source_counts = df["source_file"].fillna("<missing>").value_counts(dropna=False)
    largest_source = str(source_counts.index[0]) if not source_counts.empty else ""
    largest_count = int(source_counts.iloc[0]) if not source_counts.empty else 0
    largest_pct = pct(largest_count, total)

    complete_duplicate_rows = issue_count(issues, "complete_duplicate")
    complete_duplicate_groups = issue_group_count(issues, "complete_duplicate")
    condition_conflict_rows = issue_count(issues, "same_condition_different_log_gamma")
    condition_conflict_groups = issue_group_count(issues, "same_condition_different_log_gamma")
    cross_source_rows = issue_count(issues, "cross_source_same_condition_different_log_gamma")
    cross_source_groups = issue_group_count(issues, "cross_source_same_condition_different_log_gamma")

    water_solvent_samples = int(summary_value(summary, "water_nonwater_statistics", "water_as_solvent_samples"))
    water_solute_samples = int(summary_value(summary, "water_nonwater_statistics", "water_as_solute_samples"))
    water_component_samples = int(summary_value(summary, "water_nonwater_statistics", "samples_with_water_component"))
    nonwater_system_samples = int(summary_value(summary, "water_nonwater_statistics", "samples_without_water_component"))
    isomer_any_samples = int(summary_value(summary, "isomer_statistics", "samples_with_any_isomer_marker"))
    isomer_solute_samples = int(summary_value(summary, "isomer_statistics", "samples_with_solute_isomer_marker"))
    isomer_solvent_samples = int(summary_value(summary, "isomer_statistics", "samples_with_solvent_isomer_marker"))
    unique_isomer_solutes = int(summary_value(summary, "isomer_statistics", "unique_solutes_with_isomer_marker"))
    unique_isomer_solvents = int(summary_value(summary, "isomer_statistics", "unique_solvents_with_isomer_marker"))

    report = f"""# 数据集组成与数据质量分析

## 实验怎么做

本分析脚本读取 `{input_csv.relative_to(PROJECT_ROOT)}`，并要求数据表包含 `Solute_SMILES`、`Solvent_SMILES`、`T`、`log-gamma` 和 `source_file` 五个字段。脚本首先统计总样本数、unique solute、unique solvent、unique solute-solvent pair、`source_file` 分布、温度 `T` 的范围与直方分布、`log-gamma` 的范围与直方分布，以及来源样本量。

重复与冲突的定义如下：完全重复记录指五个必需字段全部相同的记录；同条件冲突指 `Solute_SMILES + Solvent_SMILES + T` 相同但 `log-gamma` 存在多个取值的记录；跨来源冲突是在同条件冲突基础上进一步要求同一条件出现在多个 `source_file` 中。脚本将所有命中的明细写入 `duplicate_conflicts.csv`，并在其中标注 `issue_type`、`issue_group_id`、组内样本数、不同 `log-gamma` 取值数、来源文件数和 `log-gamma` 极差。

图表由脚本直接从数据表生成，输出为 `{figure_path.name}`，采用 `2x3` 布局，包括 `source_file` 样本数柱状图、`log-gamma` 分布、温度分布、重复/冲突检查结果、异构体标记统计，以及水/非水体系组成。`dataset_composition.csv` 使用长格式 summary 表保存总体统计、来源分布、数值直方分布、top 溶质/溶剂频次、数据质量、异构体标记和水/非水统计；`dataset_summary_table.csv` 与 `{latex_table_path.name}` 则保存适合论文使用的汇总表。

异构体统计采用不依赖 RDKit 的 SMILES 标记启发式：若 SMILES 中包含 `@`、`/`、`\\` 或同位素标签，则计为带异构体/立体化学标记的记录。水/非水体系统计按溶质或溶剂 SMILES 是否精确匹配 `{'; '.join(sorted(WATER_SMILES))}` 计算，因此既给出 water-as-solvent，也给出 water-as-solute 和任一组分含水的样本数。

## 数据集组成图

![数据集组成、数值分布、数据质量及水/非水体系统计]({figure_path.name})

**Figure 1. Composition and chemical coverage characteristics of the PGSSI dataset. (a) Number of samples from each source file. (b) Distribution of log-gamma values. (c) Distribution of temperatures in the dataset. (d) Number of samples containing isomeric or stereochemical markers in the solute, solvent, or either component. (e) Numbers of samples in the H-bond-rich subset and outside this subset. (f) Numbers of samples corresponding to water-as-solvent, water-as-solute, any-water, and non-water systems.**

## 数据集汇总表

下表与论文 LaTeX 表格 [`{latex_table_path.name}`]({latex_table_path.name}) 使用相同的统计数据。

**Table 1. Dataset composition and quality summary.**

{publication_markdown_table(publication_table)}

## Fingerhut 等人数据处理方式对照

审稿意见中提到的 Fingerhut 等人（2018）在公开检索结果中对应到常被引用的 Fingerhut 等人关于 COSMO-SAC 模型大规模评估的文章，公开索引为 Fingerhut et al., Industrial & Engineering Chemistry Research, 2017, DOI: `10.1021/acs.iecr.7b01360`。该文使用 Dortmund Data Bank (DDB) 的 VLE 和无限稀释活度系数数据，并且只保留所有比较模型都能计算的严格子集；文中报告了 2,295 个组分、10,897 个无限稀释活度系数混合物和 29,173 个无限稀释活度系数数据点。原文还明确说明应用了 quality filter 以移除不可靠实验数据；对 VLE 数据，还移除了两相区边界纯物质点以及压力高于 1000 kPa 的点，因为所比较模型将气相按理想气体处理。

可以核实到的是，Fingerhut 等人的做法是基于 DDB 质量标记和模型可计算性进行筛选，并将筛选后的数据用于模型误差评估；公开文本中没有看到其对同一 `solute-solvent-temperature` 条件下多个相互冲突的实验 `log-gamma` 值进行平均、删除或仲裁的具体规则。因此，本文当前脚本不对冲突值做自动合并或人为删改，而是保留原始记录、显式标注完全重复、同条件冲突和跨来源同条件冲突，并输出 `duplicate_conflicts.csv` 供复核。这样处理更适合回应可复现性问题：任何潜在冲突都可由输入数据、定义和脚本重新得到。

## 结果分析

合并数据集共有 {total} 条样本，包含 {unique_solute} 个 unique solute、{unique_solvent} 个 unique solvent，以及 {unique_pair} 个 unique solute-solvent pair。这说明当前主数据文件不仅给出了样本总量，也能追溯到具体的溶质、溶剂和二元体系覆盖范围。

`source_file` 分布如下：

{source_markdown_table(df)}

按样本量看，最大来源为 `{largest_source}`，包含 {largest_count} 条样本，占全数据集 {largest_pct:.2f}%。该分布直接给出了各来源对合并数据集的贡献比例，可用于回应审稿人关于数据来源构成是否透明的问题。

温度 `T` 的覆盖范围为 {t_min:.6g} 到 {t_max:.6g} 摄氏度，中位数为 {t_median:.6g} 摄氏度。`log-gamma` 的覆盖范围为 {lg_min:.6g} 到 {lg_max:.6g}，中位数为 {lg_median:.6g}。这些范围和直方分布已经写入 `dataset_composition.csv`，图中也给出了温度和目标值的整体分布形态，因此模型训练数据的温度和目标覆盖范围可以由脚本复现。

数据质量检查发现：完全重复记录 {complete_duplicate_rows} 条，涉及 {complete_duplicate_groups} 个重复组；`Solute_SMILES + Solvent_SMILES + T` 相同但 `log-gamma` 不同的冲突记录 {condition_conflict_rows} 条，涉及 {condition_conflict_groups} 个条件组；其中跨 `source_file` 的同条件冲突记录 {cross_source_rows} 条，涉及 {cross_source_groups} 个条件组。当前处理方式是标记并输出这些记录，不在分析脚本中自动平均或删除，因此审稿人可以根据 `duplicate_conflicts.csv` 复核每一组重复/冲突的来源和数值差异。

异构体标记统计显示：含 solute 异构体/立体化学标记的样本为 {isomer_solute_samples} 条，含 solvent 异构体/立体化学标记的样本为 {isomer_solvent_samples} 条，任一组分含标记的样本为 {isomer_any_samples} 条；带标记的 unique solute 为 {unique_isomer_solutes} 个，带标记的 unique solvent 为 {unique_isomer_solvents} 个。水/非水统计显示：water-as-solvent 样本 {water_solvent_samples} 条，water-as-solute 样本 {water_solute_samples} 条，任一组分含水的样本 {water_component_samples} 条，完全非水体系样本 {nonwater_system_samples} 条。

综上，`dataset_composition.csv`、`duplicate_conflicts.csv`、`dataset_summary_table.csv`、`{latex_table_path.name}`、`dataset_composition.md` 和组成图均由同一个可复跑脚本从 `dataset/all/all_merged.csv` 自动生成。结论均来自实际统计值，能够明确说明数据来源比例、温度与 `log-gamma` 覆盖范围、异构体和水/非水组成，以及重复/冲突记录的检查标准和结果，从而增强数据组成说明和审稿复现性。
"""
    report_path.write_text(report, encoding="utf-8")
    return report_path


def run_analysis(input_csv: Path, output_dir: Path, bins: int, top_n: int) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    df = load_dataset(input_csv)
    issues = build_duplicate_conflicts(df)
    summary = build_composition_summary(df, issues=issues, bins=bins)
    figure_path = plot_dataset_composition(df, issues=issues, output_dir=output_dir, top_n=top_n, bins=bins)
    publication_table = build_publication_table(df, summary, issues)
    table_csv_path, latex_table_path = write_latex_summary_table(publication_table, output_dir)

    summary_path = output_dir / "dataset_composition.csv"
    issues_path = output_dir / "duplicate_conflicts.csv"
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
    issues.to_csv(issues_path, index=False, encoding="utf-8-sig")
    report_path = write_markdown_report(
        df,
        summary,
        issues,
        input_csv,
        output_dir,
        figure_path,
        latex_table_path,
        publication_table,
    )

    print(f"Saved summary: {summary_path}", flush=True)
    print(f"Saved duplicate/conflict records: {issues_path}", flush=True)
    print(f"Saved table CSV: {table_csv_path}", flush=True)
    print(f"Saved LaTeX table: {latex_table_path}", flush=True)
    print(f"Saved report: {report_path}", flush=True)
    print(f"Saved figure: {figure_path}", flush=True)


def main() -> None:
    args = parse_args()
    run_analysis(
        input_csv=Path(args.input_csv),
        output_dir=Path(args.output_dir),
        bins=args.hist_bins,
        top_n=args.top_n,
    )


if __name__ == "__main__":
    main()
