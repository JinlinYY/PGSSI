"""PGSSI 特征重要性（遮挡 / saliency / grad*input）。

默认与 ``runs/pgssi_train`` 下 molecule_train 系列权重及 summary 对齐；``--train-summary-json``
用于自动恢复训练时架构开关。Occlusion 对离散特征置 0 表示「某一取值」而非因果移除，解读时宜保守。
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
import warnings
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from matplotlib.colors import LinearSegmentedColormap
from rdkit.Chem import Descriptors
from torch_geometric.data import Batch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.interpretability.pgssi_interpretability import (
    build_selected_data,
    ensure_dir,
    heavy_atom_count,
    load_model,
    predict_sample,
)


matplotlib.rcParams.update(
    {
        "font.family": "Arial",
        "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans"],
        "font.size": 20,
        "axes.labelsize": 20,
        "axes.titlesize": 22,
        "xtick.labelsize": 18,
        "ytick.labelsize": 18,
        "figure.dpi": 220,
        "savefig.dpi": 600,
        "axes.linewidth": 1.0,
    }
)


def parse_args():
    parser = argparse.ArgumentParser(description="Feature importance experiment for PGSSI.")
    parser.add_argument(
        "--model-path",
        type=str,
        default="runs/pgssi_train/molecule_train_PGSSI_best.pth",
    )
    parser.add_argument("--data-path", type=str, default="dataset/all/all_merged_test.csv")
    parser.add_argument(
        "--predictions-path",
        type=str,
        default="runs/pgssi_train/molecule_train_PGSSI_all_merged_test_predictions.csv",
    )
    parser.add_argument("--output-dir", type=str, default="src/interpretability/outputs/final_model/feature_importance_experiment")
    parser.add_argument("--cache-dir", type=str, default="cache/interpretability_feature_experiment")
    parser.add_argument("--num-samples", type=int, default=32)
    parser.add_argument("--selection", type=str, default="lowest_error", choices=["lowest_error", "top_error", "first"])
    parser.add_argument("--exclude-small-solvents", action="store_true", default=True)
    parser.add_argument("--allow-small-solvents", dest="exclude_small_solvents", action="store_false")
    parser.add_argument("--top-atom", type=int, default=12)
    parser.add_argument("--top-bond", type=int, default=4)
    parser.add_argument("--top-global", type=int, default=13)
    parser.add_argument(
        "--components",
        type=int,
        default=2,
        help="Deprecated legacy argument; retained only for CLI compatibility.",
    )
    parser.add_argument("--method", type=str, default="occlusion", choices=["saliency", "grad_input", "occlusion"])
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument(
        "--num-intra-layers",
        type=int,
        default=None,
        help="分子内 EGNN 层数；默认从 --train-summary-json 读取，否则为 2。",
    )
    parser.add_argument("--use-interaction-types", action="store_true", default=True)
    parser.add_argument("--disable-interaction-types", dest="use_interaction_types", action="store_false")
    parser.add_argument("--use-moe", action="store_true", default=True)
    parser.add_argument("--disable-moe", dest="use_moe", action="store_false")
    parser.add_argument(
        "--train-summary-json",
        type=str,
        default=str(REPO_ROOT / "runs" / "pgssi_train" / "molecule_train_PGSSI_summary.json"),
        help="若存在则读取其中的架构开关（与训练时一致）；CLI 的 --disable-* 仍优先生效。",
    )
    parser.add_argument("--disable-cross-interaction", action="store_true")
    parser.add_argument("--disable-physics-prior", action="store_true")
    parser.add_argument("--disable-cross-refine", action="store_true")
    parser.add_argument("--topology-only", action="store_true")
    parser.add_argument("--direct-loggamma-head", action="store_true")
    parser.add_argument("--disable-formula-layer", action="store_true")
    return parser.parse_args()


def _resolve_repo_path(path_str: str | Path) -> Path:
    p = Path(path_str)
    return p if p.is_absolute() else (REPO_ROOT / p)


def apply_pgssi_model_flags(args: argparse.Namespace) -> None:
    """将路径解析为绝对路径，并从训练 summary 合并与 PGSSI_train 一致的架构开关。"""
    args.model_path = str(_resolve_repo_path(args.model_path))
    args.data_path = str(_resolve_repo_path(args.data_path))
    args.predictions_path = str(_resolve_repo_path(args.predictions_path))

    summary_path = _resolve_repo_path(args.train_summary_json)
    flags: dict = {}
    if summary_path.exists():
        try:
            with open(summary_path, encoding="utf-8") as fh:
                flags = json.load(fh)
        except (OSError, json.JSONDecodeError):
            flags = {}

    saved_use_interaction_types = bool(args.use_interaction_types)
    saved_use_moe = bool(args.use_moe)

    if args.num_intra_layers is None:
        args.num_intra_layers = int(flags.get("num_intra_layers", 2))
    else:
        args.num_intra_layers = int(args.num_intra_layers)

    args.enable_cross_interaction = bool(flags.get("enable_cross_interaction", True))
    if getattr(args, "disable_cross_interaction", False):
        args.enable_cross_interaction = False

    if "use_interaction_types" in flags:
        args.use_interaction_types = bool(flags["use_interaction_types"])
    if not saved_use_interaction_types:
        args.use_interaction_types = False

    if "use_moe" in flags:
        args.use_moe = bool(flags["use_moe"])
    if not saved_use_moe:
        args.use_moe = False

    args.use_physics_prior = not bool(getattr(args, "disable_physics_prior", False))
    args.disable_cross_refine = bool(getattr(args, "disable_cross_refine", False))
    args.topology_only = bool(getattr(args, "topology_only", False))
    args.direct_loggamma_head = bool(getattr(args, "direct_loggamma_head", False))
    args.disable_formula_layer = bool(getattr(args, "disable_formula_layer", False))


def atom_feature_names_38() -> list[str]:
    symbols = ["C", "N", "O", "Cl", "S", "F", "Br", "I", "P", "H", "Si", "Sn", "Pb", "Ge", "Hg", "Te"]
    names = [f"Atom {symbol}" for symbol in symbols]
    names += ["In Ring", "Aromatic", "Degree", "Formal Charge", "Total H", "SP", "SP2", "SP3", "Chiral"]
    while len(names) < 38:
        names.append(f"Atom Padding {len(names) - 24}")
    return names[:38]


def atom_feature_names_model() -> list[str]:
    names = atom_feature_names_38()
    names += [
        "Partial Charge",
        "H-bond Donor",
        "H-bond Acceptor",
        "Aromatic Role",
        "Hydrophobic Role",
        "Polar Role",
        "Aromatic Normal X",
        "Aromatic Normal Y",
        "Aromatic Normal Z",
        "Mol Dipole X",
        "Mol Dipole Y",
        "Mol Dipole Z",
    ]
    return names


def bond_feature_names_4() -> list[str]:
    return ["Bond Angle", "Dihedral", "Has Angle", "Has Dihedral"]


def global_feature_names() -> list[str]:
    return [name for name, _ in Descriptors._descList]


def interaction_feature_names() -> list[str]:
    return [
        "Distance",
        "LJ Energy",
        "Coulomb Energy",
        "Charge Product",
        "Charge Difference",
        "H-bond Tendency",
        "Aromatic Pair",
        "Pi-stacking Align",
        "Dipole Align",
        "Dipole Opposition",
        "Hydrophobic Pair",
        "Polar Pair",
        "Hydrophobic-Polar",
    ]


def interaction_feature_key_groups() -> list[tuple[str, list[str]]]:
    base = [
        ("Distance", ["cross_dist", "cross_refine_dist"]),
        ("LJ Energy", ["cross_lj_energy", "cross_refine_lj_energy"]),
        ("Coulomb Energy", ["cross_coulomb_energy", "cross_refine_coulomb_energy"]),
        ("Charge Product", ["cross_charge_product", "cross_refine_charge_product"]),
        ("Charge Difference", ["cross_charge_difference", "cross_refine_charge_difference"]),
        ("H-bond Tendency", ["cross_hbond_tendency", "cross_refine_hbond_tendency"]),
        ("Aromatic Pair", ["cross_aromatic_pair", "cross_refine_aromatic_pair"]),
        ("Pi-stacking Align", ["cross_pi_stacking_align", "cross_refine_pi_stacking_align"]),
        ("Dipole Align", ["cross_dipole_align", "cross_refine_dipole_align"]),
        ("Dipole Opposition", ["cross_dipole_opposition", "cross_refine_dipole_opposition"]),
        ("Hydrophobic Pair", ["cross_hydrophobic_pair", "cross_refine_hydrophobic_pair"]),
        ("Polar Pair", ["cross_polar_pair", "cross_refine_polar_pair"]),
        ("Hydrophobic-Polar", ["cross_hydrophobic_polar", "cross_refine_hydrophobic_polar"]),
    ]
    return base


def interaction_feature_display_groups() -> list[tuple[str, list[str]]]:
    return [
        ("Distance-related", ["Distance", "LJ Energy"]),
        ("Charge-related", ["Coulomb Energy", "Charge Product", "Charge Difference"]),
        (
            "Type-related",
            [
                "H-bond Tendency",
                "Aromatic Pair",
                "Pi-stacking Align",
                "Dipole Align",
                "Dipole Opposition",
                "Hydrophobic Pair",
                "Polar Pair",
                "Hydrophobic-Polar",
            ],
        ),
    ]


def interaction_activation_scores(physics_aux: dict[str, torch.Tensor]) -> np.ndarray:
    scores = []
    for _, aux_keys in interaction_feature_key_groups():
        values = []
        for aux_key in aux_keys:
            tensor = physics_aux.get(aux_key)
            if tensor is None or not torch.is_tensor(tensor) or tensor.numel() == 0:
                continue
            values.append(float(tensor.detach().abs().mean().cpu()))
        scores.append(float(np.mean(values)) if values else 0.0)
    return np.asarray(scores, dtype=float)


def occlusion_feature_importance(model, data, device, baseline_pred):
    def _mask_attr_dim(attr: str, dim_idx: int):
        edited = copy.deepcopy(data)
        tensor = getattr(edited, attr, None)
        if tensor is not None and tensor.numel() > 0 and tensor.dim() == 2 and dim_idx < tensor.size(1):
            tensor[:, dim_idx] = 0.0
        return edited

    def _attribute_dim(attr: str) -> int:
        tensor = getattr(data, attr, None)
        if tensor is None or tensor.numel() == 0 or tensor.dim() != 2:
            return 0
        return int(tensor.size(1))

    atom_attrs = {
        "solvent": ["x_s", "charge_s", "atom_role_s", "aromatic_normal_s", "dipole_vec_s"],
        "solute": ["x_v", "charge_v", "atom_role_v", "aromatic_normal_v", "dipole_vec_v"],
    }

    atom_scores_s, atom_scores_v = [], []
    for attr in atom_attrs["solvent"]:
        for dim_idx in range(_attribute_dim(attr)):
            pred = predict_sample(model, _mask_attr_dim(attr, dim_idx), device)["prediction"]
            atom_scores_s.append(baseline_pred - pred)
    for attr in atom_attrs["solute"]:
        for dim_idx in range(_attribute_dim(attr)):
            pred = predict_sample(model, _mask_attr_dim(attr, dim_idx), device)["prediction"]
            atom_scores_v.append(baseline_pred - pred)

    bond_scores_s, bond_scores_v = [], []
    bond_dim_s = _attribute_dim("edge_attr_s")
    bond_dim_v = _attribute_dim("edge_attr_v")
    for dim_idx in range(bond_dim_s):
        pred = predict_sample(model, _mask_attr_dim("edge_attr_s", dim_idx), device)["prediction"]
        bond_scores_s.append(baseline_pred - pred)
    for dim_idx in range(bond_dim_v):
        pred = predict_sample(model, _mask_attr_dim("edge_attr_v", dim_idx), device)["prediction"]
        bond_scores_v.append(baseline_pred - pred)

    global_scores_s, global_scores_v = [], []
    for dim_idx in range(_attribute_dim("rdkit_s")):
        pred = predict_sample(model, _mask_attr_dim("rdkit_s", dim_idx), device)["prediction"]
        global_scores_s.append(baseline_pred - pred)
    for dim_idx in range(_attribute_dim("rdkit_v")):
        pred = predict_sample(model, _mask_attr_dim("rdkit_v", dim_idx), device)["prediction"]
        global_scores_v.append(baseline_pred - pred)

    interaction_scores = []
    num_interaction_features = len(interaction_feature_names())
    for dim_idx in range(num_interaction_features):
        edited = copy.deepcopy(data)
        edited.interaction_feature_mask = torch.ones((num_interaction_features,), dtype=torch.float32)
        edited.interaction_feature_mask[dim_idx] = 0.0
        pred = predict_sample(model, edited, device)["prediction"]
        interaction_scores.append(baseline_pred - pred)

    return {
        "atom_solvent": np.asarray(atom_scores_s, dtype=float),
        "atom_solute": np.asarray(atom_scores_v, dtype=float),
        "bond_solvent": np.asarray(bond_scores_s, dtype=float),
        "bond_solute": np.asarray(bond_scores_v, dtype=float),
        "global_solvent": np.asarray(global_scores_s, dtype=float),
        "global_solute": np.asarray(global_scores_v, dtype=float),
        "interaction": np.asarray(interaction_scores, dtype=float),
    }


def saliency_feature_importance(model, data, device, use_grad_times_input: bool):
    batch = Batch.from_data_list([copy.deepcopy(data)]).to(device)
    watched = [
        "x_s",
        "x_v",
        "charge_s",
        "charge_v",
        "atom_role_s",
        "atom_role_v",
        "aromatic_normal_s",
        "aromatic_normal_v",
        "dipole_vec_s",
        "dipole_vec_v",
        "edge_attr_s",
        "edge_attr_v",
        "rdkit_s",
        "rdkit_v",
    ]
    leaf_tensors = {}
    for name in watched:
        tensor = getattr(batch, name, None)
        if tensor is None or tensor.numel() == 0:
            continue
        detached = tensor.detach().clone().requires_grad_(True)
        setattr(batch, name, detached)
        leaf_tensors[name] = detached

    model.zero_grad(set_to_none=True)
    out = model(batch, return_dict=True)
    pred = out["log_gamma"].view(-1)[0]
    physics_aux = out.get("physics_aux", {})
    retained_aux = {}
    for key, tensor in physics_aux.items():
        if tensor is None or not torch.is_tensor(tensor) or tensor.numel() == 0 or not tensor.requires_grad:
            continue
        tensor.retain_grad()
        retained_aux[key] = tensor
    pred.backward()

    def _aggregate(name: str) -> np.ndarray:
        tensor = leaf_tensors.get(name)
        if tensor is None or tensor.grad is None or tensor.numel() == 0:
            return np.zeros((0,), dtype=float)
        contrib = (tensor.grad * tensor).abs() if use_grad_times_input else tensor.grad.abs()
        return contrib.sum(dim=0).detach().cpu().numpy().astype(float)

    atom_scores_solvent = np.concatenate(
        [
            _aggregate("x_s"),
            _aggregate("charge_s"),
            _aggregate("atom_role_s"),
            _aggregate("aromatic_normal_s"),
            _aggregate("dipole_vec_s"),
        ]
    ).astype(float)
    atom_scores_solute = np.concatenate(
        [
            _aggregate("x_v"),
            _aggregate("charge_v"),
            _aggregate("atom_role_v"),
            _aggregate("aromatic_normal_v"),
            _aggregate("dipole_vec_v"),
        ]
    ).astype(float)

    interaction_scores = []
    activation_scores = interaction_activation_scores(physics_aux)
    used_activation_fallback = False
    for idx, (_, aux_keys) in enumerate(interaction_feature_key_groups()):
        grad_values = []
        for aux_key in aux_keys:
            retained = retained_aux.get(aux_key)
            if retained is None or retained.grad is None:
                continue
            contrib = (retained.grad * retained).abs() if use_grad_times_input else retained.grad.abs()
            grad_values.append(float(contrib.mean().detach().cpu()))
        if grad_values and any(value > 0.0 for value in grad_values):
            interaction_scores.append(float(np.mean(grad_values)))
        else:
            used_activation_fallback = True
            interaction_scores.append(float(activation_scores[idx]) if idx < len(activation_scores) else 0.0)

    if used_activation_fallback:
        warnings.warn(
            "Saliency/grad_input：部分跨分子通道未收到 physics_aux 的梯度，已退化为用激活强度代替；"
            "交互项排名仅作启发式参考，建议与 occlusion 对照。",
            UserWarning,
            stacklevel=2,
        )

    return {
        "atom_solvent": atom_scores_solvent,
        "atom_solute": atom_scores_solute,
        "bond_solvent": _aggregate("edge_attr_s"),
        "bond_solute": _aggregate("edge_attr_v"),
        "global_solvent": _aggregate("rdkit_s"),
        "global_solute": _aggregate("rdkit_v"),
        "interaction": np.asarray(interaction_scores, dtype=float),
    }


def load_analysis_frame(data_path: str, predictions_path: str) -> pd.DataFrame:
    data_df = pd.read_csv(data_path)
    pred_df = pd.read_csv(predictions_path)
    if len(data_df) != len(pred_df):
        raise ValueError("Prediction file length does not match data file length.")
    key_cols = [c for c in ("Solvent_SMILES", "Solute_SMILES", "T") if c in data_df.columns and c in pred_df.columns]
    if len(key_cols) >= 2:
        left = data_df[key_cols].reset_index(drop=True).copy()
        right = pred_df[key_cols].reset_index(drop=True).copy()
        if "T" in key_cols:
            left["T"] = pd.to_numeric(left["T"], errors="coerce")
            right["T"] = pd.to_numeric(right["T"], errors="coerce")
        for col in key_cols:
            if col != "T" and str(left[col].dtype) == "object":
                left[col] = left[col].astype(str)
                right[col] = right[col].astype(str)
        aligned = True
        for col in key_cols:
            if col == "T":
                la = left[col].to_numpy(dtype=float)
                ra = right[col].to_numpy(dtype=float)
                if la.shape != ra.shape or not np.allclose(la, ra, rtol=0.0, atol=1e-5, equal_nan=True):
                    aligned = False
                    break
            elif not left[col].equals(right[col]):
                aligned = False
                break
        if not aligned:
            raise ValueError(
                "data-path 与 predictions-path 按行不对齐（已比对列 "
                f"{key_cols}）。请保证两表行序一致，或先在相同键上 join 后再导出预测。"
            )
    merged = pred_df.copy()
    if "abs_error" not in merged.columns and {"log-gamma", "pred_log-gamma"}.issubset(merged.columns):
        merged["abs_error"] = np.abs(merged["log-gamma"] - merged["pred_log-gamma"])
    merged["row_index"] = np.arange(len(merged))
    return merged


def select_sample_indices(frame: pd.DataFrame, num_samples: int, selection: str, exclude_small_solvents: bool) -> list[int]:
    work = frame.copy()
    if exclude_small_solvents:
        work = work[work["Solvent_SMILES"].map(heavy_atom_count) >= 3].copy()
    if "abs_error" not in work.columns:
        raise ValueError("Prediction file must contain abs_error or both log-gamma and pred_log-gamma.")

    if selection == "lowest_error":
        work = work.sort_values(["abs_error", "row_index"], ascending=[True, True])
    elif selection == "top_error":
        work = work.sort_values(["abs_error", "row_index"], ascending=[False, True])
    else:
        work = work.sort_values("row_index", ascending=True)
    return work["row_index"].head(num_samples).astype(int).tolist()


def role_rank_table(role_matrices: dict[str, np.ndarray], feature_names: list[str], top_k: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not role_matrices:
        return pd.DataFrame(), pd.DataFrame()

    rank_df = pd.DataFrame({"Feature": feature_names})
    role_rank_cols = []
    role_importance_cols = []
    for role_name, matrix in role_matrices.items():
        values = np.asarray(matrix, dtype=float)
        if values.ndim != 2 or values.shape[1] != len(feature_names):
            values = np.zeros((0, len(feature_names)), dtype=float)
        abs_values = np.abs(values)
        mean_abs = abs_values.mean(axis=0) if abs_values.size else np.zeros((len(feature_names),), dtype=float)
        std_abs = abs_values.std(axis=0) if abs_values.size else np.zeros((len(feature_names),), dtype=float)
        order = np.argsort(-mean_abs)
        rank_array = np.empty(len(feature_names), dtype=int)
        rank_array[order] = np.arange(1, len(feature_names) + 1)
        rank_df[f"{role_name} MeanAbsImportance"] = mean_abs
        rank_df[f"{role_name} StdAbsImportance"] = std_abs
        rank_df[f"{role_name} Rank"] = rank_array
        role_rank_cols.append(f"{role_name} Rank")
        role_importance_cols.append(f"{role_name} MeanAbsImportance")

    rank_df["MeanRank"] = rank_df[role_rank_cols].mean(axis=1)
    rank_df["MeanAbsImportance"] = rank_df[role_importance_cols].mean(axis=1)
    rank_df = rank_df.sort_values(["MeanRank", "MeanAbsImportance"], ascending=[True, False]).reset_index(drop=True)
    display_df = rank_df.head(min(top_k, len(rank_df))).copy()
    return rank_df, display_df


def save_raw_matrix(matrix: np.ndarray, feature_names: list[str], sample_indices: list[int], path: Path) -> None:
    df = pd.DataFrame(matrix, columns=feature_names)
    df.insert(0, "sample_index", sample_indices[: len(df)])
    df.to_csv(path, index=False)


def ensure_width(matrix: np.ndarray, width: int) -> np.ndarray:
    values = np.asarray(matrix, dtype=float)
    if values.ndim != 2:
        values = np.zeros((0, width), dtype=float)
    if values.shape[1] == width:
        return values
    if values.shape[1] > width:
        return values[:, :width]
    pad = np.zeros((values.shape[0], width - values.shape[1]), dtype=float)
    return np.hstack([values, pad])


def _style_axis_box(ax) -> None:
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.1)
        spine.set_edgecolor("#50636b")
    ax.set_facecolor("#fbfbf8")


def _draw_rank_panel(fig, ax, title: str, frame: pd.DataFrame, cmap, show_ylabel: bool, scale: float = 1.0) -> None:
    title_fs = 18 * scale
    label_fs = 14 * scale
    tick_fs = 12 * scale
    annot_fs = 10 * scale
    cbar_fs = 11 * scale
    cbar_tick_fs = 9 * scale
    if frame.empty:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", fontsize=tick_fs, color="#4a5568")
        _style_axis_box(ax)
        return

    rank_cols = [col for col in frame.columns if col.endswith(" Rank")]
    plot_frame = frame.copy().iloc[::-1]
    matrix = plot_frame[rank_cols].to_numpy(dtype=float)
    im = ax.imshow(matrix, cmap=cmap, aspect="auto", interpolation="nearest")
    ax.set_title(title, pad=10, fontweight="normal", fontsize=title_fs)
    ax.set_xticks(range(len(rank_cols)))
    ax.set_xticklabels([col.replace(" Rank", "") for col in rank_cols], fontweight="normal", fontsize=tick_fs)
    ax.set_xlabel("Component", labelpad=6, fontweight="normal", fontsize=label_fs)
    ax.set_yticks(range(len(plot_frame)))
    ax.set_yticklabels(plot_frame["Feature"], fontweight="normal", fontsize=tick_fs)
    if show_ylabel:
        ax.set_ylabel("Feature", labelpad=8, fontweight="normal", fontsize=label_fs)
    ax.tick_params(axis="y", labelsize=tick_fs)
    ax.tick_params(axis="x", labelsize=tick_fs)
    threshold = float(np.nanmax(matrix)) * 0.55 if matrix.size else 0.0
    for row_idx in range(matrix.shape[0]):
        for col_idx in range(matrix.shape[1]):
            value = matrix[row_idx, col_idx]
            text_color = "#102a43" if value >= threshold else "white"
            ax.text(
                col_idx,
                row_idx,
                f"{int(value)}",
                ha="center",
                va="center",
                color=text_color,
                fontsize=annot_fs,
                fontweight="normal",
            )
    cbar = fig.colorbar(im, ax=ax, fraction=0.042, pad=0.03)
    cbar.set_label("Rank within component (smaller means more important)", rotation=90, fontsize=cbar_fs, labelpad=9, fontweight="normal")
    cbar.ax.tick_params(labelsize=cbar_tick_fs)
    _style_axis_box(ax)


def _draw_mean_panel(ax, title: str, frame: pd.DataFrame, color: str, scale: float = 1.0) -> None:
    title_fs = 18 * scale
    label_fs = 14 * scale
    tick_fs = 12 * scale
    note_fs = 13 * scale
    subnote_fs = 10 * scale
    legend_fs = 10 * scale
    if frame.empty:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", fontsize=tick_fs, color="#4a5568")
        _style_axis_box(ax)
        return

    importance_cols = [col for col in frame.columns if col.endswith(" MeanAbsImportance") and col != "MeanAbsImportance"]
    plot_frame = frame.sort_values("MeanAbsImportance", ascending=True)
    if not importance_cols or float(plot_frame["MeanAbsImportance"].abs().max()) <= 1e-12:
        ax.axvline(0.0, color="#c8d1d3", linewidth=1.2)
        ax.text(
            0.5,
            0.56,
            "All importances are 0",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=note_fs,
            color="#4a5568",
        )
        ax.text(
            0.5,
            0.44,
            "under current model/checkpoint",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=subnote_fs,
            color="#718096",
        )
        ax.set_title(title, pad=10, fontsize=title_fs, fontweight="normal")
        ax.set_xlabel("Mean absolute importance", fontsize=label_fs)
        ax.set_yticks([])
        ax.tick_params(axis="x", labelsize=tick_fs)
        _style_axis_box(ax)
        return

    y = np.arange(len(plot_frame))
    if len(importance_cols) == 1:
        ax.barh(y, plot_frame[importance_cols[0]], color=color, edgecolor="none", height=0.72)
    else:
        offsets = np.linspace(-0.18, 0.18, len(importance_cols))
        panel_colors = ["#1f6f78", "#b8c99d", "#c8553d"]
        for idx, col in enumerate(importance_cols):
            ax.barh(
                y + offsets[idx],
                plot_frame[col],
                color=panel_colors[idx % len(panel_colors)],
                edgecolor="none",
                height=0.26,
                label=col.replace(" MeanAbsImportance", ""),
            )
        ax.legend(frameon=False, loc="lower right", fontsize=legend_fs)
    ax.set_yticks(y)
    ax.set_yticklabels(plot_frame["Feature"], fontsize=tick_fs)
    ax.set_title(title, pad=10, fontsize=title_fs, fontweight="normal")
    ax.set_xlabel("Mean absolute importance", fontsize=label_fs)
    ax.tick_params(axis="x", labelsize=tick_fs)
    ax.tick_params(axis="y", labelsize=tick_fs)
    ax.grid(axis="x", color="#d9d9d9", linewidth=0.7, alpha=0.7)
    ax.margins(y=0.03)
    _style_axis_box(ax)


def plot_rank_heatmaps(atom_df: pd.DataFrame, bond_df: pd.DataFrame, global_df: pd.DataFrame, interaction_df: pd.DataFrame, out_path: Path) -> None:
    frames = [
        ("Atom Features", atom_df),
        ("Bond Features", bond_df),
        ("Global Features", global_df),
        ("Intermolecular Interaction Features", interaction_df),
    ]
    cmap = LinearSegmentedColormap.from_list(
        "nature_rank",
        ["#0f4c5c", "#2c7a7b", "#86bcb6", "#f3efe0", "#e6b89c", "#c8553d"],
        N=256,
    )

    fig, axes = plt.subplots(1, 4, figsize=(30, 10.5), constrained_layout=True)
    fig.patch.set_facecolor("white")
    for ax, (title, frame) in zip(axes, frames):
        _draw_rank_panel(fig, ax, title, frame, cmap, show_ylabel=(ax is axes[0]))

    fig.savefig(out_path, dpi=600, bbox_inches="tight")
    plt.close(fig)


def plot_mean_importance(atom_df: pd.DataFrame, bond_df: pd.DataFrame, global_df: pd.DataFrame, interaction_df: pd.DataFrame, out_path: Path) -> None:
    frames = [
        ("Atom Features", atom_df),
        ("Bond Features", bond_df),
        ("Global Features", global_df),
        ("Intermolecular Interaction Features", interaction_df),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(22, 8.4), constrained_layout=True)
    fig.patch.set_facecolor("white")
    colors = ["#1f6f78", "#287271", "#4d908e", "#577590"]
    for ax, (title, frame), color in zip(axes, frames, colors):
        _draw_mean_panel(ax, title, frame, color)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_combined_figure(atom_df: pd.DataFrame, bond_df: pd.DataFrame, global_df: pd.DataFrame, interaction_df: pd.DataFrame, out_path: Path) -> None:
    frames = [
        ("Atom Features", atom_df),
        ("Bond Features", bond_df),
        ("Global Features", global_df),
        ("Intermolecular Interaction Features", interaction_df),
    ]
    cmap = LinearSegmentedColormap.from_list(
        "nature_rank",
        ["#0f4c5c", "#2c7a7b", "#86bcb6", "#f3efe0", "#e6b89c", "#c8553d"],
        N=256,
    )
    colors = ["#1f6f78", "#287271", "#4d908e", "#577590"]

    fig = plt.figure(figsize=(27, 16.5), constrained_layout=True)
    fig.patch.set_facecolor("white")
    grid = fig.add_gridspec(2, 4, height_ratios=[1.12, 1.0], hspace=0.08, wspace=0.08)
    top_axes = [fig.add_subplot(grid[0, idx]) for idx in range(4)]
    bottom_axes = [fig.add_subplot(grid[1, idx]) for idx in range(4)]

    for idx, (ax, (title, frame)) in enumerate(zip(top_axes, frames)):
        _draw_rank_panel(fig, ax, title, frame, cmap, show_ylabel=(idx == 0), scale=1.22)
    for ax, (title, frame), color in zip(bottom_axes, frames, colors):
        _draw_mean_panel(ax, title, frame, color, scale=1.22)

    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_interaction_grouped_importance(interaction_df: pd.DataFrame, out_path: Path) -> None:
    groups = interaction_feature_display_groups()
    colors = {
        "Distance-related": "#c8553d",
        "Charge-related": "#4d908e",
        "Type-related": "#577590",
    }
    width_ratios = [1.0, 1.2, 2.6]
    fig, axes = plt.subplots(
        1,
        len(groups),
        figsize=(20, 8.8),
        constrained_layout=True,
        gridspec_kw={"width_ratios": width_ratios},
    )
    fig.patch.set_facecolor("white")

    title_fs = 20
    label_fs = 15
    tick_fs = 12
    annot_fs = 11

    if interaction_df.empty:
        for ax in np.atleast_1d(axes):
            ax.text(0.5, 0.5, "No data", ha="center", va="center", fontsize=tick_fs, color="#4a5568")
            _style_axis_box(ax)
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        return

    importance_col = "Interaction MeanAbsImportance"
    axes = np.atleast_1d(axes)
    for ax, (group_name, feature_names) in zip(axes, groups):
        frame = interaction_df[interaction_df["Feature"].isin(feature_names)].copy()
        frame = frame.sort_values(importance_col, ascending=True)
        if frame.empty:
            ax.text(0.5, 0.5, "No data", ha="center", va="center", fontsize=tick_fs, color="#4a5568")
            ax.set_title(group_name, pad=10, fontsize=title_fs, fontweight="normal")
            _style_axis_box(ax)
            continue

        y = np.arange(len(frame))
        values = frame[importance_col].to_numpy(dtype=float)
        ax.barh(y, values, color=colors[group_name], edgecolor="none", height=0.66)
        ax.set_yticks(y)
        ax.set_yticklabels(frame["Feature"], fontsize=tick_fs)
        ax.set_title(group_name, pad=10, fontsize=title_fs, fontweight="normal")
        ax.set_xlabel("Mean absolute importance", fontsize=label_fs)
        ax.tick_params(axis="x", labelsize=tick_fs)
        ax.tick_params(axis="y", labelsize=tick_fs)
        ax.grid(axis="x", color="#d9d9d9", linewidth=0.7, alpha=0.7)
        ax.margins(x=0.08, y=0.04)

        max_value = float(values.max()) if values.size else 0.0
        x_pad = max(max_value * 0.03, 1e-6)
        for yi, value in enumerate(values):
            ax.text(
                value + x_pad,
                yi,
                f"{value:.3g}",
                va="center",
                ha="left",
                fontsize=annot_fs,
                color="#334e68",
            )
        _style_axis_box(ax)

    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    args = parse_args()
    apply_pgssi_model_flags(args)
    ensure_dir(args.output_dir)
    ensure_dir(args.cache_dir)

    frame = load_analysis_frame(args.data_path, args.predictions_path)
    sample_indices = select_sample_indices(frame, args.num_samples, args.selection, args.exclude_small_solvents)
    selected_df = pd.read_csv(args.data_path).loc[sample_indices].copy()
    selected_data = build_selected_data(selected_df, sample_indices, args.cache_dir)

    model, device = load_model(args)

    atom_names = atom_feature_names_model()
    bond_names = bond_feature_names_4()
    global_names = global_feature_names()
    interaction_names = interaction_feature_names()

    atom_rows_s = []
    atom_rows_v = []
    bond_rows_s = []
    bond_rows_v = []
    global_rows_s = []
    global_rows_v = []
    interaction_rows = []
    summary_rows = []
    for row_index, (_, row), data in zip(sample_indices, selected_df.iterrows(), selected_data):
        baseline = predict_sample(model, data, device)
        if args.method == "occlusion":
            scores = occlusion_feature_importance(model, data, device, baseline["prediction"])
        else:
            scores = saliency_feature_importance(
                model,
                data,
                device,
                use_grad_times_input=(args.method == "grad_input"),
            )
        atom_rows_s.append(np.asarray(scores["atom_solvent"], dtype=float))
        atom_rows_v.append(np.asarray(scores["atom_solute"], dtype=float))
        bond_rows_s.append(np.asarray(scores["bond_solvent"], dtype=float))
        bond_rows_v.append(np.asarray(scores["bond_solute"], dtype=float))
        global_rows_s.append(np.asarray(scores["global_solvent"], dtype=float))
        global_rows_v.append(np.asarray(scores["global_solute"], dtype=float))
        interaction_rows.append(np.asarray(scores["interaction"], dtype=float))
        summary_rows.append(
            {
                "sample_index": int(row_index),
                "solute_smiles": row["Solute_SMILES"],
                "solvent_smiles": row["Solvent_SMILES"],
                "temperature_c": float(row["T"]),
                "true_log_gamma": float(row["log-gamma"]),
                "pred_log_gamma": float(baseline["prediction"]),
            }
        )

    atom_matrix_s = np.vstack(atom_rows_s) if atom_rows_s else np.zeros((0, len(atom_names)), dtype=float)
    atom_matrix_v = np.vstack(atom_rows_v) if atom_rows_v else np.zeros((0, len(atom_names)), dtype=float)
    bond_matrix_s = np.vstack(bond_rows_s) if bond_rows_s else np.zeros((0, len(bond_names)), dtype=float)
    bond_matrix_v = np.vstack(bond_rows_v) if bond_rows_v else np.zeros((0, len(bond_names)), dtype=float)
    global_matrix_s = np.vstack(global_rows_s) if global_rows_s else np.zeros((0, len(global_names)), dtype=float)
    global_matrix_v = np.vstack(global_rows_v) if global_rows_v else np.zeros((0, len(global_names)), dtype=float)
    interaction_matrix = np.vstack(interaction_rows) if interaction_rows else np.zeros((0, len(interaction_names)), dtype=float)
    atom_matrix_s = ensure_width(atom_matrix_s, len(atom_names))
    atom_matrix_v = ensure_width(atom_matrix_v, len(atom_names))
    bond_matrix_s = ensure_width(bond_matrix_s, len(bond_names))
    bond_matrix_v = ensure_width(bond_matrix_v, len(bond_names))
    global_matrix_s = ensure_width(global_matrix_s, len(global_names))
    global_matrix_v = ensure_width(global_matrix_v, len(global_names))
    interaction_matrix = ensure_width(interaction_matrix, len(interaction_names))

    atom_rank_df, atom_display = role_rank_table({"Solvent": atom_matrix_s, "Solute": atom_matrix_v}, atom_names, args.top_atom)
    bond_rank_df, bond_display = role_rank_table({"Solvent": bond_matrix_s, "Solute": bond_matrix_v}, bond_names, args.top_bond)
    global_rank_df, global_display = role_rank_table({"Solvent": global_matrix_s, "Solute": global_matrix_v}, global_names, args.top_global)
    interaction_rank_df, interaction_display = role_rank_table({"Interaction": interaction_matrix}, interaction_names, len(interaction_names))

    out_dir = Path(args.output_dir)
    pd.DataFrame(summary_rows).to_csv(out_dir / "selected_samples.csv", index=False)
    save_raw_matrix(atom_matrix_s, atom_names, sample_indices, out_dir / "atom_importance_matrix_solvent.csv")
    save_raw_matrix(atom_matrix_v, atom_names, sample_indices, out_dir / "atom_importance_matrix_solute.csv")
    save_raw_matrix(bond_matrix_s, bond_names, sample_indices, out_dir / "bond_importance_matrix_solvent.csv")
    save_raw_matrix(bond_matrix_v, bond_names, sample_indices, out_dir / "bond_importance_matrix_solute.csv")
    save_raw_matrix(global_matrix_s, global_names, sample_indices, out_dir / "global_importance_matrix_solvent.csv")
    save_raw_matrix(global_matrix_v, global_names, sample_indices, out_dir / "global_importance_matrix_solute.csv")
    save_raw_matrix(interaction_matrix, interaction_names, sample_indices, out_dir / "interaction_importance_matrix.csv")
    atom_rank_df.to_csv(out_dir / "atom_feature_ranks.csv", index=False)
    bond_rank_df.to_csv(out_dir / "bond_feature_ranks.csv", index=False)
    global_rank_df.to_csv(out_dir / "global_feature_ranks.csv", index=False)
    interaction_rank_df.to_csv(out_dir / "interaction_feature_ranks.csv", index=False)
    interaction_group_rows = []
    for group_name, feature_names in interaction_feature_display_groups():
        group_frame = interaction_rank_df[interaction_rank_df["Feature"].isin(feature_names)].copy()
        imp = group_frame["Interaction MeanAbsImportance"] if not group_frame.empty else pd.Series(dtype=float)
        group_mean_abs = float(imp.mean()) if len(imp) else 0.0
        group_sum_abs = float(imp.sum()) if len(imp) else 0.0
        interaction_group_rows.append(
            {
                "Group": group_name,
                "NumFeatures": int(len(group_frame)),
                "GroupMeanAbsImportance": group_mean_abs,
                "GroupSumAbsImportance": group_sum_abs,
                "Features": ", ".join(group_frame.sort_values("Interaction Rank")["Feature"].tolist()),
            }
        )
    pd.DataFrame(interaction_group_rows).to_csv(out_dir / "interaction_grouped_importance.csv", index=False)

    plot_rank_heatmaps(atom_display, bond_display, global_display, interaction_display, out_dir / "feature_solvent_solute_ranks.png")
    plot_mean_importance(atom_display, bond_display, global_display, interaction_display, out_dir / "feature_mean_importance.png")
    plot_combined_figure(atom_display, bond_display, global_display, interaction_display, out_dir / "feature_importance_combined.png")
    plot_interaction_grouped_importance(interaction_rank_df, out_dir / "interaction_grouped_importance.png")

    summary = {
        "num_samples": len(sample_indices),
        "selection": args.selection,
        "method": args.method,
        "model_path": args.model_path,
        "train_summary_json": str(_resolve_repo_path(args.train_summary_json)),
        "model_flags": {
            "hidden_dim": args.hidden_dim,
            "num_intra_layers": args.num_intra_layers,
            "enable_cross_interaction": args.enable_cross_interaction,
            "use_interaction_types": args.use_interaction_types,
            "use_moe": args.use_moe,
            "use_physics_prior": args.use_physics_prior,
            "disable_cross_refine": args.disable_cross_refine,
            "topology_only": args.topology_only,
            "direct_loggamma_head": args.direct_loggamma_head,
            "disable_formula_layer": args.disable_formula_layer,
        },
        "sample_indices": [int(idx) for idx in sample_indices],
        "atom_features_shown": atom_display["Feature"].tolist() if not atom_display.empty else [],
        "bond_features_shown": bond_display["Feature"].tolist() if not bond_display.empty else [],
        "global_features_shown": global_display["Feature"].tolist() if not global_display.empty else [],
        "interaction_features_shown": interaction_display["Feature"].tolist() if not interaction_display.empty else [],
        "interaction_feature_groups": {group_name: feature_names for group_name, feature_names in interaction_feature_display_groups()},
        "notes": (
            "Saliency/grad_input 下若 physics_aux 无梯度，交互通道会退化为激活强度（见运行时的 UserWarning）。"
            "interaction_grouped_importance.csv 中 GroupMeanAbsImportance 为组内特征重要性的算术平均，"
            "GroupSumAbsImportance 为组内求和。"
        ),
    }
    with open(out_dir / "feature_importance_summary.json", "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, ensure_ascii=False)

    print(f"Saved feature importance experiment to: {out_dir}")


if __name__ == "__main__":
    main()
