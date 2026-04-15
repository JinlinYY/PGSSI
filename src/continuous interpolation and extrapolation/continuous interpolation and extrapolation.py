from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, Normalize
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TRAIN = PROJECT_ROOT / "dataset" / "wu_et_al" / "molecule_trainvalid.csv"
DEFAULT_TEST_PRED = (
    PROJECT_ROOT
    / "runs"
    / "pgssi_wu_benchmark_refit_trainvalid"
    / "molecule_trainvalid_PGSSI_WuBenchmarkRefit_molecule_test_predictions.csv"
)
DEFAULT_OUTPUT = Path(__file__).resolve().parent / "outputs"

FIG_BG = "#ffffff"
AX_BG = "#ffffff"
TEXT_COLOR = "#243746"
GRID_COLOR = "#e4e8ec"
SPINE_COLOR = "#000000"
BOX_EDGE = "#000000"
DISTANCE_CMAP = LinearSegmentedColormap.from_list(
    "temperature_distance",
    ["#1d5a72", "#4e8aa2", "#a8c7cb", "#e9d6b0", "#bf6f59"],
)
ERROR_CMAP = LinearSegmentedColormap.from_list(
    "temperature_error",
    ["#244b67", "#5a86a3", "#b7d4d9", "#edd7ac", "#d57a61"],
)
REGIME_COLORS = {
    "interpolation": "#3b7f7a",
    "lower_extrapolation": "#5f88ad",
    "upper_extrapolation": "#bc7851",
}
REGIME_TITLES = {
    "interpolation": "Interpolation",
    "lower_extrapolation": "Lower Extrapolation",
    "upper_extrapolation": "Upper Extrapolation",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Visualize interpolation and extrapolation performance for the final PGSSI model."
    )
    parser.add_argument("--train-path", type=str, default=str(DEFAULT_TRAIN))
    parser.add_argument("--test-pred-path", type=str, default=str(DEFAULT_TEST_PRED))
    parser.add_argument("--pred-column", type=str, default="pred_log-gamma")
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT))
    parser.add_argument("--dpi", type=int, default=600)
    return parser.parse_args()


def configure_plot_style():
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.weight": "normal",
            "font.size": 20,
            "text.color": TEXT_COLOR,
            "axes.titlesize": 27,
            "axes.titleweight": "normal",
            "axes.titlepad": 14,
            "axes.labelsize": 24,
            "axes.labelweight": "normal",
            "axes.labelcolor": TEXT_COLOR,
            "axes.labelpad": 11,
            "axes.linewidth": 1.2,
            "axes.edgecolor": SPINE_COLOR,
            "xtick.labelsize": 19,
            "ytick.labelsize": 19,
            "xtick.color": TEXT_COLOR,
            "ytick.color": TEXT_COLOR,
            "xtick.major.width": 1.1,
            "ytick.major.width": 1.1,
            "xtick.major.size": 5.5,
            "ytick.major.size": 5.5,
            "legend.fontsize": 18,
            "legend.facecolor": FIG_BG,
            "legend.edgecolor": "#cad2d9",
            "legend.framealpha": 0.96,
            "figure.titlesize": 27,
            "figure.titleweight": "normal",
            "figure.facecolor": FIG_BG,
            "axes.facecolor": AX_BG,
            "savefig.facecolor": FIG_BG,
            "savefig.bbox": "tight",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def nature_distance_cmap():
    return DISTANCE_CMAP


def style_axes(ax, equal: bool = False):
    ax.set_facecolor(AX_BG)
    ax.grid(True, which="major", color=GRID_COLOR, lw=0.7, alpha=0.8, linestyle="--")
    ax.tick_params(axis="both", pad=6, colors=TEXT_COLOR, width=1.0, length=5)
    for spine in ax.spines.values():
        spine.set_color(SPINE_COLOR)
        spine.set_linewidth(1.0)
    if equal:
        ax.set_aspect("equal", adjustable="box")


def style_colorbar(cbar, label: str, labelsize: int = 19, ticksize: int = 17):
    cbar.set_label(label, color=TEXT_COLOR, labelpad=10)
    cbar.ax.yaxis.label.set_size(labelsize)
    cbar.ax.tick_params(colors=TEXT_COLOR, width=0.9, length=4, labelsize=ticksize)
    cbar.outline.set_edgecolor(SPINE_COLOR)
    cbar.outline.set_linewidth(0.9)


def get_system_temperatures(df: pd.DataFrame) -> dict[str, list[float]]:
    systems: dict[str, set[float]] = {}
    for row in df.itertuples(index=False):
        key = f"{row.Solvent_SMILES}__{row.Solute_SMILES}"
        systems.setdefault(key, set()).add(float(row.T))
    return {key: sorted(values) for key, values in systems.items()}


def classify_regime(train_map: dict[str, list[float]], row: pd.Series) -> tuple[str, float]:
    key = f"{row['Solvent_SMILES']}__{row['Solute_SMILES']}"
    test_temp = float(row["T"])
    if key not in train_map:
        return "novel_system", np.nan

    train_temps = np.asarray(train_map[key], dtype=float)
    if test_temp < float(train_temps.min()):
        return "lower_extrapolation", float(train_temps.min() - test_temp)
    if test_temp > float(train_temps.max()):
        return "upper_extrapolation", float(test_temp - train_temps.max())
    return "interpolation", float(np.min(np.abs(train_temps - test_temp)))


def prepare_analysis_dataframe(train_df: pd.DataFrame, test_df: pd.DataFrame, pred_column: str) -> pd.DataFrame:
    if pred_column not in test_df.columns:
        raise KeyError(f"Prediction column not found: {pred_column}")
    if "log-gamma" not in test_df.columns:
        raise KeyError("Expected column 'log-gamma' in test prediction file.")

    train_map = get_system_temperatures(train_df)
    classified = test_df.apply(lambda row: classify_regime(train_map, row), axis=1, result_type="expand")
    out = test_df.copy()
    out["regime"] = classified[0]
    out["distance"] = classified[1]
    out["abs_error"] = (out["log-gamma"] - out[pred_column]).abs()
    out["squared_error"] = np.square(out["log-gamma"] - out[pred_column])
    return out


def metric_summary(df: pd.DataFrame, pred_column: str) -> pd.DataFrame:
    rows = []
    for regime in ["interpolation", "lower_extrapolation", "upper_extrapolation", "novel_system"]:
        subset = df[df["regime"] == regime].dropna(subset=[pred_column, "log-gamma"])
        if subset.empty:
            continue
        y_true = subset["log-gamma"].to_numpy(dtype=float)
        y_pred = subset[pred_column].to_numpy(dtype=float)
        rows.append(
            {
                "regime": regime,
                "n_samples": int(len(subset)),
                "mae": float(mean_absolute_error(y_true, y_pred)),
                "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
                "r2": float(r2_score(y_true, y_pred)),
                "mean_distance_C": float(subset["distance"].mean()) if subset["distance"].notna().any() else np.nan,
                "max_distance_C": float(subset["distance"].max()) if subset["distance"].notna().any() else np.nan,
            }
        )
    return pd.DataFrame(rows)


def add_metric_box(
    ax,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_samples: int,
    *,
    fontsize: int = 20,
    x: float = 0.035,
    y: float = 0.968,
    box_pad: float = 0.62,
):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    text = f"n = {n_samples}\nMAE {mae:.4f}\nRMSE {rmse:.4f}\nR2   {r2:.4f}"
    ax.text(
        x,
        y,
        text,
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=fontsize,
        color=TEXT_COLOR,
        linespacing=1.24,
        bbox={"boxstyle": f"round,pad={box_pad}", "fc": FIG_BG, "ec": BOX_EDGE, "lw": 1.05, "alpha": 0.985},
        zorder=10,
    )


def plot_single_regime(
    df: pd.DataFrame,
    regime: str,
    title: str,
    pred_column: str,
    output_dir: Path,
    dpi: int,
    cmap,
    color_column: str = "distance",
    color_label: str = "Temperature distance (°C)",
):
    subset = df[df["regime"] == regime].dropna(subset=[pred_column, "log-gamma"]).copy()
    if subset.empty:
        return

    sort_col = color_column if color_column in subset.columns else pred_column
    subset = subset.sort_values(sort_col)
    y_true = subset["log-gamma"].to_numpy(dtype=float)
    y_pred = subset[pred_column].to_numpy(dtype=float)
    color_values = subset[color_column].to_numpy(dtype=float)

    lim_min = min(y_true.min(), y_pred.min())
    lim_max = max(y_true.max(), y_pred.max())
    pad = 0.04 * max(lim_max - lim_min, 1.0)
    vmax = float(np.nanmax(color_values)) if np.nanmax(color_values) > np.nanmin(color_values) else float(np.nanmin(color_values) + 1.0)
    norm = Normalize(vmin=float(np.nanmin(color_values)), vmax=vmax)

    fig, ax = plt.subplots(figsize=(8.8, 7.6), constrained_layout=True)
    ax.plot([lim_min - pad, lim_max + pad], [lim_min - pad, lim_max + pad], ls="--", lw=1.5, c=SPINE_COLOR, zorder=1)
    sc = ax.scatter(
        y_true,
        y_pred,
        c=color_values,
        cmap=cmap,
        norm=norm,
        s=82,
        alpha=0.92,
        edgecolors=FIG_BG,
        linewidths=0.55,
        zorder=3,
    )
    ax.set_xlim(lim_min - pad, lim_max + pad)
    ax.set_ylim(lim_min - pad, lim_max + pad)
    ax.set_xlabel(r"Experimental $\ln(\gamma^\infty)$")
    ax.set_ylabel(r"Predicted $\ln(\gamma^\infty)$")
    ax.set_title(title)
    style_axes(ax, equal=True)
    add_metric_box(ax, y_true, y_pred, len(subset))
    cbar = fig.colorbar(sc, ax=ax, fraction=0.05, pad=0.03, shrink=0.95)
    style_colorbar(cbar, color_label)
    fig.savefig(output_dir / f"{regime}_parity.png", dpi=dpi)
    fig.savefig(output_dir / f"{regime}_parity.pdf")
    plt.close(fig)


def plot_temperature_triptych(df: pd.DataFrame, pred_column: str, output_dir: Path, dpi: int, cmap):
    display = [(k, REGIME_TITLES[k]) for k in ["interpolation", "lower_extrapolation", "upper_extrapolation"]]
    available = [(k, t) for k, t in display if not df[df["regime"] == k].dropna(subset=[pred_column, "log-gamma"]).empty]
    if not available:
        return

    full = df[df["regime"].isin([k for k, _ in available])].dropna(subset=[pred_column, "log-gamma", "distance"]).copy()
    dist = full["distance"].to_numpy(dtype=float)
    vmax = float(np.nanmax(dist)) if np.nanmax(dist) > np.nanmin(dist) else float(np.nanmin(dist) + 1.0)
    norm = Normalize(vmin=float(np.nanmin(dist)), vmax=vmax)

    fig = plt.figure(figsize=(12.9, 5.05), constrained_layout=False)
    grid = fig.add_gridspec(
        1,
        len(available) + 1,
        width_ratios=[1.0] * len(available) + [0.018],
        wspace=0.002,
    )
    axes = []
    for idx in range(len(available)):
        axes.append(fig.add_subplot(grid[0, idx]))
    cax = fig.add_subplot(grid[0, -1])
    fig.subplots_adjust(left=0.048, right=0.992, bottom=0.17, top=0.865)

    scatter = None
    for idx, (ax, (regime, title)) in enumerate(zip(axes, available)):
        subset = df[df["regime"] == regime].dropna(subset=[pred_column, "log-gamma"]).sort_values("distance")
        y_true = subset["log-gamma"].to_numpy(dtype=float)
        y_pred = subset[pred_column].to_numpy(dtype=float)
        color_values = subset["distance"].to_numpy(dtype=float)
        lim_min = min(y_true.min(), y_pred.min())
        lim_max = max(y_true.max(), y_pred.max())
        pad = 0.05 * max(lim_max - lim_min, 1.0)
        ax.plot([lim_min - pad, lim_max + pad], [lim_min - pad, lim_max + pad], ls="--", lw=1.45, c=SPINE_COLOR, zorder=1)
        scatter = ax.scatter(
            y_true,
            y_pred,
            c=color_values,
            cmap=cmap,
            norm=norm,
            s=56,
            alpha=0.9,
            edgecolors=FIG_BG,
            linewidths=0.42,
            zorder=3,
        )
        ax.set_title(title, fontsize=20, pad=6)
        ax.set_xlabel(r"Experimental $\ln(\gamma^\infty)$", fontsize=17, labelpad=5)
        ax.set_xlim(lim_min - pad, lim_max + pad)
        ax.set_ylim(lim_min - pad, lim_max + pad)
        style_axes(ax, equal=True)
        ax.tick_params(axis="both", labelsize=14, pad=3)
        add_metric_box(ax, y_true, y_pred, len(subset), fontsize=15, x=0.03, y=0.94, box_pad=0.42)

    axes[0].set_ylabel(r"Predicted $\ln(\gamma^\infty)$", fontsize=17, labelpad=6)
    cbar = fig.colorbar(scatter, cax=cax)
    style_colorbar(cbar, "Temperature distance (°C)", labelsize=16, ticksize=13)
    fig.savefig(output_dir / "interpolation_extrapolation_overview.png", dpi=dpi)
    fig.savefig(output_dir / "interpolation_extrapolation_overview.pdf")
    fig.savefig(output_dir / "temperature_regimes_triptych.png", dpi=dpi)
    fig.savefig(output_dir / "temperature_regimes_triptych.pdf")
    plt.close(fig)



def plot_novel_system(df: pd.DataFrame, pred_column: str, output_dir: Path, dpi: int):
    subset = df[df["regime"] == "novel_system"].dropna(subset=[pred_column, "log-gamma", "abs_error"]).copy()
    if subset.empty:
        return
    plot_single_regime(
        subset,
        "novel_system",
        "Novel Binary Systems",
        pred_column,
        output_dir,
        dpi,
        ERROR_CMAP,
        color_column="abs_error",
        color_label=r"Absolute error of $\ln(\gamma^\infty)$",
    )


def plot_error_trend(df: pd.DataFrame, output_dir: Path, dpi: int):
    subset = df[df["regime"].isin(["interpolation", "lower_extrapolation", "upper_extrapolation"])].dropna(subset=["distance", "abs_error"]).copy()
    if subset.empty:
        return

    fig, ax = plt.subplots(figsize=(9.0, 7.2), constrained_layout=True)
    for regime in ["interpolation", "lower_extrapolation", "upper_extrapolation"]:
        part = subset[subset["regime"] == regime].sort_values("distance")
        if part.empty:
            continue
        ax.scatter(
            part["distance"],
            part["abs_error"],
            s=54,
            alpha=0.68,
            color=REGIME_COLORS[regime],
            edgecolors=FIG_BG,
            linewidths=0.4,
            label=REGIME_TITLES[regime],
        )

    ax.set_xlabel("Temperature distance from training range (°C)")
    ax.set_ylabel(r"Absolute error of $\ln(\gamma^\infty)$")
    style_axes(ax, equal=False)
    legend = ax.legend(loc="upper left")
    legend.get_frame().set_facecolor(FIG_BG)
    legend.get_frame().set_edgecolor("#cad2d9")
    legend.get_frame().set_linewidth(0.9)
    fig.savefig(output_dir / "distance_error_trend.png", dpi=dpi)
    fig.savefig(output_dir / "distance_error_trend.pdf")
    plt.close(fig)


def main():
    args = parse_args()
    configure_plot_style()

    train_path = Path(args.train_path)
    test_pred_path = Path(args.test_pred_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_df = pd.read_csv(train_path)
    test_pred_df = pd.read_csv(test_pred_path)
    analysis_df = prepare_analysis_dataframe(train_df, test_pred_df, args.pred_column)
    metrics_df = metric_summary(analysis_df, args.pred_column)

    analysis_df.to_csv(output_dir / "interpolation_extrapolation_predictions.csv", index=False)
    metrics_df.to_csv(output_dir / "interpolation_extrapolation_metrics.csv", index=False)

    cmap = nature_distance_cmap()
    plot_temperature_triptych(analysis_df, args.pred_column, output_dir, args.dpi, cmap)
    plot_error_trend(analysis_df, output_dir, args.dpi)
    plot_single_regime(analysis_df, "interpolation", REGIME_TITLES["interpolation"], args.pred_column, output_dir, args.dpi, cmap)
    plot_single_regime(analysis_df, "lower_extrapolation", REGIME_TITLES["lower_extrapolation"], args.pred_column, output_dir, args.dpi, cmap)
    plot_single_regime(analysis_df, "upper_extrapolation", REGIME_TITLES["upper_extrapolation"], args.pred_column, output_dir, args.dpi, cmap)
    plot_novel_system(analysis_df, args.pred_column, output_dir, args.dpi)

    print(f"Saved analysis table to: {output_dir / 'interpolation_extrapolation_predictions.csv'}")
    print(f"Saved metrics table to: {output_dir / 'interpolation_extrapolation_metrics.csv'}")
    print(f"Saved figures to: {output_dir}")


if __name__ == "__main__":
    main()
