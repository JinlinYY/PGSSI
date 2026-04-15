from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.lines import Line2D
from sklearn.metrics import mean_absolute_error, r2_score

try:
    from rdkit import Chem
except ImportError:
    Chem = None


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"

TRUE_COLUMN = "log-gamma"
PRED_COLUMN = "pred_log-gamma"
SOLVENT_COLUMN = "Solvent_SMILES"
SOLUTE_COLUMN = "Solute_SMILES"

FAMILY_PRIORITY = [
    ("Acids", "[CX3](=O)[OX2H1]"),
    ("Esters", "[CX3](=O)[OX2][#6]"),
    ("Amides", "[NX3][CX3](=O)[#6]"),
    ("Nitriles", "[CX2]#N"),
    ("Amines", "[NX3;H2,H1,H0;!$(NC=O)]"),
    ("Alcohols", "[OX2H][#6]"),
    ("Ethers", "[OD2]([#6])[#6]"),
    ("Ketones", "[#6][CX3](=O)[#6]"),
    ("Aromatics", "a"),
    ("Halogenated", "[F,Cl,Br,I]"),
    ("Alkenes", "C=C"),
    ("Alkynes", "C#C"),
]

FAMILY_ORDER = [
    "Others",
    "Alcohols",
    "Alkanes",
    "Esters",
    "Aromatics",
    "Amides",
    "Ketones",
    "Nitriles",
    "Halogenated",
    "Acids",
    "Amines",
    "Ethers",
    "Alkenes",
    "Alkynes",
]

NATURE_BG = "#ffffff"
NATURE_AXES_BG = "#ffffff"
NATURE_TEXT = "#22313f"
NATURE_GRID = "#d9d2c4"
NATURE_SPINE = "#000000"
NATURE_LOWER_BETTER_CMAP = LinearSegmentedColormap.from_list(
    "nature_lower_better",
    ["#1d5a72", "#4e8aa2", "#a8c7cb", "#e9d6b0", "#bf6f59"],
)
NATURE_HIGHER_BETTER_CMAP = LinearSegmentedColormap.from_list(
    "nature_higher_better",
    ["#cb7a66", "#ebc9a4", "#c8d9d1", "#6da6a3", "#1f5a63"],
)
NATURE_ERROR_CMAP = LinearSegmentedColormap.from_list(
    "nature_error",
    ["#244b67", "#5a86a3", "#b7d4d9", "#edd7ac", "#d57a61"],
)
NATURE_GROUP_PALETTE = ["#2f5c7a", "#4d8e94", "#7e9a46", "#c5894b", "#8b6278", "#71808c", "#bb6d6b"]


def configure_publication_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "Liberation Sans", "DejaVu Sans"],
            "font.size": 20,
            "font.weight": "normal",
            "text.color": NATURE_TEXT,
            "axes.titlesize": 24,
            "axes.titleweight": "normal",
            "axes.titlepad": 12,
            "axes.labelsize": 21,
            "axes.labelweight": "normal",
            "axes.labelcolor": NATURE_TEXT,
            "axes.labelpad": 10,
            "axes.edgecolor": NATURE_SPINE,
            "axes.linewidth": 1.25,
            "axes.facecolor": NATURE_AXES_BG,
            "figure.facecolor": NATURE_BG,
            "savefig.facecolor": NATURE_BG,
            "xtick.color": NATURE_TEXT,
            "ytick.color": NATURE_TEXT,
            "xtick.labelsize": 18,
            "ytick.labelsize": 18,
            "xtick.major.width": 1.0,
            "ytick.major.width": 1.0,
            "xtick.major.size": 5,
            "ytick.major.size": 5,
            "grid.color": NATURE_GRID,
            "grid.linestyle": "--",
            "grid.linewidth": 0.65,
            "legend.fontsize": 18,
            "legend.title_fontsize": 18,
            "legend.facecolor": "#ffffff",
            "legend.edgecolor": "#000000",
            "legend.framealpha": 0.96,
            "figure.titlesize": 24,
            "figure.titleweight": "normal",
            "savefig.dpi": 600,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot PGSSI prediction results with bubble metrics and parity scatter.",
    )
    parser.add_argument(
        "--input",
        type=str,
        default=str(
            PROJECT_ROOT
            / "runs"
            / "pgssi_train_all_merged_smoke"
            / "all_merged_train_PGSSI_all_merged_test_predictions.csv"
        ),
        help="Prediction CSV file. Must contain true and predicted columns.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory for exported figures.",
    )
    parser.add_argument(
        "--true-column",
        type=str,
        default=TRUE_COLUMN,
        help="Ground-truth column name.",
    )
    parser.add_argument(
        "--pred-column",
        type=str,
        default=PRED_COLUMN,
        help="Prediction column name.",
    )
    parser.add_argument(
        "--solvent-column",
        type=str,
        default=SOLVENT_COLUMN,
        help="Solvent identifier column name.",
    )
    parser.add_argument(
        "--solute-column",
        type=str,
        default=SOLUTE_COLUMN,
        help="Solute identifier column name.",
    )
    parser.add_argument(
        "--solvent-family-column",
        type=str,
        default=None,
        help="Optional solvent family column. If absent, infer from SMILES.",
    )
    parser.add_argument(
        "--solute-family-column",
        type=str,
        default=None,
        help="Optional solute family column. If absent, infer from SMILES.",
    )
    parser.add_argument(
        "--group-column",
        type=str,
        default=None,
        help="Optional category column for parity scatter legend.",
    )
    parser.add_argument(
        "--top-groups",
        type=int,
        default=6,
        help="Max number of groups shown in parity scatter legend.",
    )
    parser.add_argument(
        "--min-count",
        type=int,
        default=5,
        help="Hide family-pair combinations with fewer than this many samples.",
    )
    return parser.parse_args()


def load_prediction_dataframe(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def build_family_patterns():
    if Chem is None:
        return []
    return [(name, Chem.MolFromSmarts(smarts)) for name, smarts in FAMILY_PRIORITY]


def classify_smiles_family_fallback(smiles: str) -> str:
    text = str(smiles)
    upper = text.upper()

    if "#N" in text:
        return "Nitriles"
    if "C#C" in text:
        return "Alkynes"
    if "C=C" in text:
        return "Alkenes"
    if any(token in upper for token in ("CL", "BR", "I")) or "F" in text:
        return "Halogenated"
    if "C(=O)O" in text and "N" not in text:
        return "Esters"
    if "C(=O)N" in text or "NC(=O)" in text:
        return "Amides"
    if "C(=O)O" in text and "O" in text:
        return "Acids"
    if "N" in text:
        return "Amines"
    if "C(=O)" in text:
        return "Ketones"
    if "O" in text and "C(=O)" not in text:
        if text.count("O") >= 2 or "OCC" in text or "COC" in text:
            return "Ethers"
        return "Alcohols"
    if any(ch.islower() for ch in text):
        return "Aromatics"

    stripped = (
        upper.replace("C", "")
        .replace("(", "")
        .replace(")", "")
        .replace("=", "")
        .replace("#", "")
        .replace("[", "")
        .replace("]", "")
        .replace("@", "")
        .replace("+", "")
        .replace("-", "")
        .replace("1", "")
        .replace("2", "")
        .replace("3", "")
        .replace("4", "")
        .replace("5", "")
        .replace("6", "")
        .replace("7", "")
        .replace("8", "")
        .replace("9", "")
        .replace("0", "")
    )
    if not stripped:
        return "Alkanes"
    return "Others"


def classify_smiles_family(smiles: str, patterns) -> str:
    if pd.isna(smiles) or not str(smiles).strip():
        return "Others"

    if Chem is None:
        return classify_smiles_family_fallback(str(smiles))

    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return classify_smiles_family_fallback(str(smiles))

    if all(atom.GetAtomicNum() in (1, 6) for atom in mol.GetAtoms()):
        if mol.HasSubstructMatch(Chem.MolFromSmarts("C#C")):
            return "Alkynes"
        if mol.HasSubstructMatch(Chem.MolFromSmarts("C=C")):
            return "Alkenes"
        if mol.GetRingInfo().NumRings() == 0:
            return "Alkanes"

    for family_name, pattern in patterns:
        if pattern is not None and mol.HasSubstructMatch(pattern):
            return family_name

    return "Others"


def attach_family_columns(
    df: pd.DataFrame,
    solvent_column: str,
    solute_column: str,
    solvent_family_column: str | None,
    solute_family_column: str | None,
) -> pd.DataFrame:
    df = df.copy()

    if solvent_family_column and solvent_family_column in df.columns:
        df["plot_solvent_family"] = df[solvent_family_column].fillna("Others").astype(str)
    else:
        patterns = build_family_patterns()
        df["plot_solvent_family"] = df[solvent_column].map(lambda x: classify_smiles_family(x, patterns))

    if solute_family_column and solute_family_column in df.columns:
        df["plot_solute_family"] = df[solute_family_column].fillna("Others").astype(str)
    else:
        patterns = build_family_patterns()
        df["plot_solute_family"] = df[solute_column].map(lambda x: classify_smiles_family(x, patterns))

    return df


def compute_group_metrics(df: pd.DataFrame, true_column: str, pred_column: str) -> pd.DataFrame:
    rows = []
    for (solute_family, solvent_family), part in df.groupby(
        ["plot_solute_family", "plot_solvent_family"],
        dropna=False,
    ):
        y_true = part[true_column].to_numpy(dtype=float)
        y_pred = part[pred_column].to_numpy(dtype=float)
        rows.append(
            {
                "plot_solute_family": solute_family,
                "plot_solvent_family": solvent_family,
                "count": len(part),
                "MAE": float(mean_absolute_error(y_true, y_pred)),
                "RMSE": float(np.sqrt(np.mean(np.square(y_true - y_pred)))),
                "R2": float(r2_score(y_true, y_pred)) if len(part) > 1 else np.nan,
            }
        )
    return pd.DataFrame(rows)


def size_to_area(counts: np.ndarray) -> np.ndarray:
    counts = np.asarray(counts, dtype=float)
    if counts.size == 0:
        return counts

    min_count = counts.min()
    max_count = counts.max()
    if min_count == max_count:
        return np.full_like(counts, 260.0)
    return 120.0 + (counts - min_count) / (max_count - min_count) * 500.0


def style_axes(ax) -> None:
    ax.set_facecolor(NATURE_AXES_BG)
    ax.grid(True, linestyle="--", linewidth=0.65, alpha=0.6, color=NATURE_GRID)
    ax.set_axisbelow(True)
    ax.tick_params(axis="both", pad=6, colors=NATURE_TEXT, width=1.0, length=5)
    for spine in ax.spines.values():
        spine.set_linewidth(1.25)
        spine.set_color(NATURE_SPINE)


def style_colorbar(
    colorbar,
    label: str,
    labelsize: int = 17,
    ticksize: int = 13,
    labelpad: int = 10,
) -> None:
    colorbar.set_label(label, labelpad=labelpad)
    colorbar.ax.yaxis.label.set_size(labelsize)
    colorbar.ax.tick_params(colors=NATURE_TEXT, width=0.9, length=4, labelsize=ticksize)
    colorbar.outline.set_linewidth(1.0)
    colorbar.outline.set_edgecolor(NATURE_SPINE)
    colorbar.ax.yaxis.label.set_color(NATURE_TEXT)
    colorbar.ax.yaxis.set_label_position("right")
    colorbar.ax.yaxis.set_ticks_position("right")


def style_legend(legend) -> None:
    if legend is None:
        return
    frame = legend.get_frame()
    frame.set_facecolor("#ffffff")
    frame.set_edgecolor("#000000")
    frame.set_linewidth(1.0)
    frame.set_alpha(0.96)


def metric_color_settings(metric_name: str, values: pd.Series):
    finite_values = values.replace([np.inf, -np.inf], np.nan).dropna().to_numpy(dtype=float)
    if finite_values.size == 0:
        finite_values = np.array([0.0, 1.0], dtype=float)

    if metric_name == "R2":
        cmap = NATURE_HIGHER_BETTER_CMAP
        vmax = max(1.0, float(np.nanmax(finite_values)))
        vmin = min(float(np.nanmin(finite_values)), vmax - 1e-6)
    else:
        cmap = NATURE_LOWER_BETTER_CMAP
        vmin = min(0.0, float(np.nanmin(finite_values)))
        vmax = max(float(np.nanmax(finite_values)), vmin + 1e-6)
    return cmap, Normalize(vmin=vmin, vmax=vmax)


def prepare_metric_bubble_plot(metrics_df: pd.DataFrame, min_count: int):
    plot_df = metrics_df.loc[metrics_df["count"] >= min_count].copy()
    if plot_df.empty:
        raise ValueError(f"No family combinations meet min_count={min_count}.")

    solvent_levels = [name for name in FAMILY_ORDER if name in set(plot_df["plot_solvent_family"])]
    solute_levels = [name for name in reversed(FAMILY_ORDER) if name in set(plot_df["plot_solute_family"])]
    solvent_index = {name: idx for idx, name in enumerate(solvent_levels)}
    solute_index = {name: idx for idx, name in enumerate(solute_levels)}

    plot_df["x"] = plot_df["plot_solvent_family"].map(solvent_index)
    plot_df["y"] = plot_df["plot_solute_family"].map(solute_index)
    plot_df["area"] = size_to_area(plot_df["count"].to_numpy())
    return plot_df, solvent_levels, solute_levels


def draw_metric_bubble_axis(
    ax,
    cax,
    plot_df: pd.DataFrame,
    solvent_levels: list[str],
    solute_levels: list[str],
    metric_name: str,
    *,
    show_ylabel: bool,
    show_yticklabels: bool,
    colorbar_labelsize: int = 17,
    colorbar_ticksize: int = 13,
    colorbar_labelpad: int = 10,
) -> None:
    cmap, norm = metric_color_settings(metric_name, plot_df[metric_name])
    sc = ax.scatter(
        plot_df["x"],
        plot_df["y"],
        s=plot_df["area"],
        c=plot_df[metric_name],
        cmap=cmap,
        norm=norm,
        alpha=0.96,
        edgecolors=NATURE_AXES_BG,
        linewidths=0.8,
    )

    ax.set_xticks(range(len(solvent_levels)))
    ax.set_xticklabels(solvent_levels, rotation=55, ha="right")
    ax.set_yticks(range(len(solute_levels)))
    ax.set_yticklabels(solute_levels if show_yticklabels else [])
    ax.set_xlim(-0.45, len(solvent_levels) - 0.55)
    ax.set_ylim(-0.45, len(solute_levels) - 0.55)
    style_axes(ax)
    ax.tick_params(axis="both", labelsize=15)
    ax.set_xlabel("Solvent Family", fontsize=18)
    if show_ylabel:
        ax.set_ylabel("Solute Family", fontsize=18)
    else:
        ax.set_ylabel("")
    ax.set_title(metric_name, fontsize=22, pad=8)

    cbar = ax.figure.colorbar(sc, cax=cax)
    style_colorbar(
        cbar,
        "Higher is better" if metric_name == "R2" else "Lower is better",
        labelsize=colorbar_labelsize,
        ticksize=colorbar_ticksize,
        labelpad=colorbar_labelpad,
    )


def draw_metric_bubble_panels(
    metrics_df: pd.DataFrame,
    output_path: Path,
    min_count: int,
) -> None:
    fig, _ = draw_metric_bubble_figure(metrics_df=metrics_df, min_count=min_count)
    fig.savefig(output_path, dpi=600, bbox_inches="tight")
    plt.close(fig)


def draw_metric_bubble_figure(
    metrics_df: pd.DataFrame,
    min_count: int,
):
    plot_df, solvent_levels, solute_levels = prepare_metric_bubble_plot(metrics_df, min_count)

    fig = plt.figure(figsize=(17.0, 5.95), constrained_layout=False)
    outer = fig.add_gridspec(1, 3, wspace=0.16)
    axes = []
    caxes = []
    for idx in range(3):
        pair = outer[0, idx].subgridspec(1, 2, width_ratios=[1.0, 0.028], wspace=0.04)
        axes.append(fig.add_subplot(pair[0, 0]))
        caxes.append(fig.add_subplot(pair[0, 1]))
    fig.subplots_adjust(left=0.055, right=0.992, bottom=0.24, top=0.87)
    metrics = ["MAE", "RMSE", "R2"]

    for ax, cax, metric_name, panel_idx in zip(axes, caxes, metrics, range(3)):
        draw_metric_bubble_axis(
            ax=ax,
            cax=cax,
            plot_df=plot_df,
            solvent_levels=solvent_levels,
            solute_levels=solute_levels,
            metric_name=metric_name,
            show_ylabel=False,
            show_yticklabels=panel_idx == 0,
            colorbar_labelsize=15,
            colorbar_ticksize=13,
            colorbar_labelpad=5,
        )

    return fig, plot_df


def compute_global_metrics(df: pd.DataFrame, true_column: str, pred_column: str) -> dict[str, float]:
    y_true = df[true_column].to_numpy(dtype=float)
    y_pred = df[pred_column].to_numpy(dtype=float)
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(np.sqrt(np.mean(np.square(y_true - y_pred)))),
        "R2": float(r2_score(y_true, y_pred)),
    }


def draw_parity_plot(
    df: pd.DataFrame,
    output_path: Path,
    true_column: str,
    pred_column: str,
    group_column: str | None,
    top_groups: int,
) -> None:
    fig, ax = plt.subplots(figsize=(6.9, 5.95), constrained_layout=False)
    fig.subplots_adjust(left=0.11, right=0.94, bottom=0.13, top=0.965)
    draw_parity_axis(
        ax=ax,
        fig=fig,
        df=df,
        true_column=true_column,
        pred_column=pred_column,
        group_column=group_column,
        top_groups=top_groups,
    )
    fig.savefig(output_path, dpi=600, bbox_inches="tight")
    plt.close(fig)


def draw_parity_axis(
    ax,
    fig,
    df: pd.DataFrame,
    true_column: str,
    pred_column: str,
    group_column: str | None,
    top_groups: int,
    colorbar_ax=None,
    metric_box_fontsize: int = 18,
    metric_box_pad: float = 0.38,
    metric_box_x: float = 0.045,
    metric_box_y: float = 0.955,
) -> None:
    metrics = compute_global_metrics(df, true_column, pred_column)

    if group_column and group_column in df.columns:
        top_values = df[group_column].astype(str).value_counts().head(top_groups).index.tolist()
        plot_df = df.copy()
        plot_df[group_column] = plot_df[group_column].astype(str)
        plot_df.loc[~plot_df[group_column].isin(top_values), group_column] = "Others"
        markers = ["o", "s", "^", "D", "P", "X", "v"]
        for idx, value in enumerate(plot_df[group_column].unique()):
            part = plot_df.loc[plot_df[group_column] == value]
            ax.scatter(
                part[true_column],
                part[pred_column],
                s=62,
                alpha=0.84,
                label=value,
                color=NATURE_GROUP_PALETTE[idx % len(NATURE_GROUP_PALETTE)],
                marker=markers[idx % len(markers)],
                edgecolors=NATURE_AXES_BG,
                linewidths=0.45,
            )
    else:
        abs_err = np.abs(df[true_column] - df[pred_column])
        scatter = ax.scatter(
            df[true_column],
            df[pred_column],
            c=abs_err,
            cmap=NATURE_ERROR_CMAP,
            s=52,
            alpha=0.88,
            edgecolors=NATURE_AXES_BG,
            linewidths=0.35,
        )
        if colorbar_ax is not None:
            cbar = fig.colorbar(scatter, cax=colorbar_ax)
        else:
            cbar = fig.colorbar(scatter, ax=ax, fraction=0.045, pad=0.018, shrink=0.985)
        style_colorbar(cbar, "|Error|", labelsize=16, ticksize=12)

    low = float(min(df[true_column].min(), df[pred_column].min()))
    high = float(max(df[true_column].max(), df[pred_column].max()))
    padding = 0.03 * (high - low if high > low else 1.0)
    low -= padding
    high += padding

    ax.plot([low, high], [low, high], linestyle="--", linewidth=1.8, color=NATURE_SPINE, label="y = x")
    ax.set_xlim(low, high)
    ax.set_ylim(low, high)
    ax.set_xlabel(r"Experimental $\ln(\gamma^\infty)$")
    ax.set_ylabel(r"Predicted $\ln(\gamma^\infty)$")
    style_axes(ax)
    ax.set_aspect("equal", adjustable="box")

    metric_text = f"MAE  {metrics['MAE']:.4f}\nRMSE {metrics['RMSE']:.4f}\nR2   {metrics['R2']:.4f}"
    ax.text(
        metric_box_x,
        metric_box_y,
        metric_text,
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=metric_box_fontsize,
        linespacing=1.25,
        bbox={
            "boxstyle": f"round,pad={metric_box_pad}",
            "facecolor": "#ffffff",
            "edgecolor": "#000000",
            "alpha": 0.97,
        },
    )

    if group_column and group_column in df.columns:
        legend = ax.legend(loc="lower right", frameon=True)
    else:
        line_handle = Line2D([0], [0], linestyle="--", color=NATURE_SPINE, linewidth=1.8, label="y = x")
        legend = ax.legend(handles=[line_handle], loc="lower right", frameon=True)
    style_legend(legend)


def draw_combined_figure(
    metrics_df: pd.DataFrame,
    df: pd.DataFrame,
    output_path: Path,
    true_column: str,
    pred_column: str,
    group_column: str | None,
    top_groups: int,
    min_count: int,
) -> None:
    plot_df, solvent_levels, solute_levels = prepare_metric_bubble_plot(metrics_df, min_count)

    fig = plt.figure(figsize=(21.0, 6.3), constrained_layout=False)
    grid = fig.add_gridspec(
        1,
        8,
        width_ratios=[0.92, 0.035, 1.0, 0.035, 1.0, 0.035, 1.0, 0.035],
        wspace=0.12,
    )
    axes = [
        fig.add_subplot(grid[0, 0]),
        fig.add_subplot(grid[0, 2]),
        fig.add_subplot(grid[0, 4]),
        fig.add_subplot(grid[0, 6]),
    ]
    caxes = [
        fig.add_subplot(grid[0, 1]),
        fig.add_subplot(grid[0, 3]),
        fig.add_subplot(grid[0, 5]),
        fig.add_subplot(grid[0, 7]),
    ]
    fig.subplots_adjust(left=0.04, right=0.992, bottom=0.23, top=0.88)

    draw_parity_axis(
        ax=axes[0],
        fig=fig,
        df=df,
        true_column=true_column,
        pred_column=pred_column,
        group_column=group_column,
        top_groups=top_groups,
        colorbar_ax=caxes[0],
        metric_box_fontsize=16,
        metric_box_pad=0.36,
        metric_box_x=0.04,
        metric_box_y=0.94,
    )
    axes[0].set_title("Parity", pad=8)

    metrics = ["MAE", "RMSE", "R2"]
    for ax, cax, metric_name, panel_idx in zip(axes[1:], caxes[1:], metrics, range(3)):
        draw_metric_bubble_axis(
            ax=ax,
            cax=cax,
            plot_df=plot_df,
            solvent_levels=solvent_levels,
            solute_levels=solute_levels,
            metric_name=metric_name,
            show_ylabel=False,
            show_yticklabels=panel_idx == 0,
        )

    fig.savefig(output_path, dpi=600, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    configure_publication_style()
    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = load_prediction_dataframe(input_path)
    df = df.dropna(subset=[args.true_column, args.pred_column]).copy()
    if df.empty:
        raise ValueError("No valid rows found after dropping missing true/prediction values.")

    if args.true_column != TRUE_COLUMN or args.pred_column != PRED_COLUMN:
        df = df.rename(columns={args.true_column: TRUE_COLUMN, args.pred_column: PRED_COLUMN})
        args.true_column = TRUE_COLUMN
        args.pred_column = PRED_COLUMN

    if args.solvent_column not in df.columns or args.solute_column not in df.columns:
        raise KeyError(
            f"Input file must contain `{args.solvent_column}` and `{args.solute_column}` columns for bubble plot."
        )

    family_df = attach_family_columns(
        df=df,
        solvent_column=args.solvent_column,
        solute_column=args.solute_column,
        solvent_family_column=args.solvent_family_column,
        solute_family_column=args.solute_family_column,
    )
    family_metrics = compute_group_metrics(family_df, args.true_column, args.pred_column)

    stem = input_path.stem
    draw_metric_bubble_panels(
        metrics_df=family_metrics,
        output_path=output_dir / f"{stem}_family_metrics.png",
        min_count=args.min_count,
    )
    draw_parity_plot(
        df=df,
        output_path=output_dir / f"{stem}_parity.png",
        true_column=args.true_column,
        pred_column=args.pred_column,
        group_column=args.group_column,
        top_groups=args.top_groups,
    )
    draw_combined_figure(
        metrics_df=family_metrics,
        df=df,
        output_path=output_dir / f"{stem}_combined_1x4.png",
        true_column=args.true_column,
        pred_column=args.pred_column,
        group_column=args.group_column,
        top_groups=args.top_groups,
        min_count=args.min_count,
    )

    family_metrics.sort_values(["plot_solute_family", "plot_solvent_family"]).to_csv(
        output_dir / f"{stem}_family_metrics.csv",
        index=False,
    )

    global_metrics = compute_global_metrics(df, args.true_column, args.pred_column)
    pd.DataFrame([global_metrics]).to_csv(output_dir / f"{stem}_overall_metrics.csv", index=False)

    print(f"Saved bubble plot to: {output_dir / f'{stem}_family_metrics.png'}")
    print(f"Saved parity plot to: {output_dir / f'{stem}_parity.png'}")
    print(f"Saved combined figure to: {output_dir / f'{stem}_combined_1x4.png'}")
    print(f"Saved family metrics to: {output_dir / f'{stem}_family_metrics.csv'}")
    print(f"Saved overall metrics to: {output_dir / f'{stem}_overall_metrics.csv'}")


if __name__ == "__main__":
    main()
