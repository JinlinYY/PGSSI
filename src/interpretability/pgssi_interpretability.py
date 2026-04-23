"""PGSSI 可解释性主脚本：选样、边/节点消融、汇总图与 3D 面板。

默认与 ``runs/pgssi_train`` 下 molecule_train 权重及预测对齐；可通过 ``--train-summary-json``
恢复训练时架构。预测 CSV 与数据 CSV 须行对齐（按 SMILES+T 校验）。"""
from __future__ import annotations

import argparse
import copy
import json
import math
import os
import sys
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
import numpy as np
import pandas as pd
import torch
from rdkit import Chem
from rdkit.Chem import Descriptors, rdDepictor, rdMolDescriptors
from torch_geometric.data import Batch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.models.PGSSI.PGSSI_3D_architecture import PGSSIModel
from src.models.PGSSI.PGSSI_data import RDKIT_DESC_CALC, build_pair_dataset


matplotlib.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans"],
        "font.size": 13,
        "axes.labelsize": 13,
        "axes.titlesize": 13,
        "xtick.labelsize": 13,
        "ytick.labelsize": 13,
        "legend.fontsize": 13,
        "figure.dpi": 220,
        "savefig.dpi": 600,
        "axes.linewidth": 1.0,
    }
)


FUNCTIONAL_GROUP_SMARTS = {
    "Hydroxyl": "[OX2H]",
    "Carbonyl": "[CX3]=[OX1]",
    "Carboxyl": "C(=O)[OX2H1]",
    "Ester": "C(=O)O[#6]",
    "Ether": "[OD2]([#6])[#6]",
    "Amine": "[NX3;H2,H1;!$(NC=O)]",
    "Amide": "C(=O)N",
    "Nitrile": "C#N",
    "Halogen": "[F,Cl,Br,I]",
    "Aromatic": "a1aaaaa1",
    "Alkene": "C=C",
    "Alkyne": "C#C",
    "Sulfur": "[SX2,SX4,SX6]",
    "Nitro": "[NX3](=O)=O",
}


def parse_args():
    parser = argparse.ArgumentParser(description="Interpretability analysis for the final PGSSI model.")
    parser.add_argument(
        "--model-path",
        type=str,
        default="runs/pgssi_train/molecule_train_PGSSI_best.pth",
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default="dataset/all/all_merged_test.csv",
    )
    parser.add_argument(
        "--predictions-path",
        type=str,
        default="runs/pgssi_train/molecule_train_PGSSI_all_merged_test_predictions.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="src/interpretability/outputs/final_model",
    )
    parser.add_argument("--cache-dir", type=str, default="cache/interpretability")
    parser.add_argument("--num-samples", type=int, default=3)
    parser.add_argument(
        "--selection",
        type=str,
        default="top_error",
        choices=["top_error", "lowest_error", "first"],
    )
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument(
        "--num-intra-layers",
        type=int,
        default=None,
        help="分子内层数；默认从 --train-summary-json 读取，否则为 2。",
    )
    parser.add_argument("--use-interaction-types", action="store_true", default=True)
    parser.add_argument("--disable-interaction-types", dest="use_interaction_types", action="store_false")
    parser.add_argument("--use-moe", action="store_true", default=True)
    parser.add_argument("--disable-moe", dest="use_moe", action="store_false")
    parser.add_argument(
        "--train-summary-json",
        type=str,
        default=str(REPO_ROOT / "runs" / "pgssi_train" / "molecule_train_PGSSI_summary.json"),
        help="若存在则读取其中的架构开关；CLI 的 --disable-* 优先生效。",
    )
    parser.add_argument("--disable-cross-interaction", action="store_true")
    parser.add_argument("--disable-physics-prior", action="store_true")
    parser.add_argument("--disable-cross-refine", action="store_true")
    parser.add_argument("--topology-only", action="store_true")
    parser.add_argument("--direct-loggamma-head", action="store_true")
    parser.add_argument("--disable-formula-layer", action="store_true")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--exclude-small-solvents", action="store_true", default=True)
    parser.add_argument("--allow-small-solvents", dest="exclude_small_solvents", action="store_false")
    parser.add_argument("--grid-rows", type=int, default=8)
    parser.add_argument("--grid-cols", type=int, default=6)
    return parser.parse_args()


def _resolve_repo_path(path_str: str | Path) -> Path:
    p = Path(path_str)
    return p if p.is_absolute() else (REPO_ROOT / p)


def apply_pgssi_model_flags(args: argparse.Namespace) -> None:
    """解析路径为绝对路径，并从训练 summary 合并架构开关（与 pgssi_feature_importance / PGSSI_eval 一致）。"""
    args.model_path = str(_resolve_repo_path(args.model_path))
    args.data_path = str(_resolve_repo_path(args.data_path))
    args.predictions_path = str(_resolve_repo_path(args.predictions_path))
    args.output_dir = str(_resolve_repo_path(args.output_dir))
    args.cache_dir = str(_resolve_repo_path(args.cache_dir))

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


def _assert_prediction_rows_aligned(data_df: pd.DataFrame, pred_df: pd.DataFrame) -> None:
    """保证预测表与数据表按行一一对应（与 pgssi_feature_importance.load_analysis_frame 一致）。"""
    if len(data_df) != len(pred_df):
        raise ValueError(f"预测表行数 ({len(pred_df)}) 与数据表 ({len(data_df)}) 不一致。")
    key_cols = [c for c in ("Solvent_SMILES", "Solute_SMILES", "T") if c in data_df.columns and c in pred_df.columns]
    if len(key_cols) < 2:
        return
    left = data_df[key_cols].reset_index(drop=True).copy()
    right = pred_df[key_cols].reset_index(drop=True).copy()
    if "T" in key_cols:
        left["T"] = pd.to_numeric(left["T"], errors="coerce")
        right["T"] = pd.to_numeric(right["T"], errors="coerce")
    for col in key_cols:
        if col != "T" and str(left[col].dtype) == "object":
            left[col] = left[col].astype(str)
            right[col] = right[col].astype(str)
    for col in key_cols:
        if col == "T":
            la = left[col].to_numpy(dtype=float)
            ra = right[col].to_numpy(dtype=float)
            if la.shape != ra.shape or not np.allclose(la, ra, rtol=0.0, atol=1e-5, equal_nan=True):
                raise ValueError(
                    "data-path 与 predictions-path 按行不对齐（列 "
                    f"{key_cols}）。请保证两表行序一致或在相同键上 join 后再导出预测。"
                )
        elif not left[col].equals(right[col]):
            raise ValueError(
                "data-path 与 predictions-path 按行不对齐（列 "
                f"{key_cols}）。请保证两表行序一致或在相同键上 join 后再导出预测。"
            )


def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def load_model(args):
    """构建 PGSSIModel 并加载权重。

    与训练脚本对齐的可选属性（均可用 ``argparse.Namespace`` 或任意带属性的对象传入）：
    ``enable_cross_interaction``, ``use_physics_prior``, ``disable_cross_refine``,
    ``topology_only``, ``direct_loggamma_head``, ``disable_formula_layer``。
    未设置时采用与 ``PGSSI_train`` 默认一致的行为。
    """
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    enable_cross = getattr(args, "enable_cross_interaction", True)
    use_physics_prior = getattr(args, "use_physics_prior", True)
    disable_cross_refine = getattr(args, "disable_cross_refine", False)
    topology_only = getattr(args, "topology_only", False)
    direct_loggamma_head = getattr(args, "direct_loggamma_head", False)
    disable_formula_layer = getattr(args, "disable_formula_layer", False)
    model = PGSSIModel(
        hidden_dim=args.hidden_dim,
        enable_cross_interaction=enable_cross,
        num_intra_layers=args.num_intra_layers,
        use_interaction_types=args.use_interaction_types,
        use_moe=args.use_moe,
        use_physics_prior=use_physics_prior,
        disable_cross_refine=disable_cross_refine,
        topology_only=topology_only,
        direct_loggamma_head=direct_loggamma_head,
        disable_formula_layer=disable_formula_layer,
    ).to(device)
    weight_path = Path(args.model_path)
    if not weight_path.is_file():
        raise FileNotFoundError(f"Model weights not found: {weight_path}")
    checkpoint = torch.load(args.model_path, map_location=device)
    state_dict = checkpoint["model_state_dict"] if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint else checkpoint
    remapped_state_dict = {}
    for key, value in state_dict.items():
        new_key = key
        new_key = new_key.replace("shared_layer.intra_conv1.", "shared_layer.intra_convs.0.")
        new_key = new_key.replace("shared_layer.intra_conv2.", "shared_layer.intra_convs.1.")
        remapped_state_dict[new_key] = value
    missing, unexpected = model.load_state_dict(remapped_state_dict, strict=False)
    if unexpected:
        print(f"Unexpected checkpoint keys ignored: {unexpected}")
    if missing:
        print(f"Missing checkpoint keys left at current init: {missing}")
        if len(missing) > 30:
            print(
                "WARNING: many missing keys — model architecture likely does not match the checkpoint "
                "(check --hidden-dim, --train-summary-json, and ablation flags)."
            )
    model.eval()
    return model, device


ATOM_FEATURE_NAMES = [
    "C",
    "N",
    "O",
    "Cl",
    "S",
    "F",
    "Br",
    "I",
    "P",
    "H",
    "Si",
    "Sn",
    "Pb",
    "Ge",
    "Hg",
    "Te",
    "In Ring",
    "Aromatic",
    "Degree",
    "Formal Charge",
    "Num H",
    "SP",
    "SP2",
    "SP3",
    "Chiral",
]

BOND_FEATURE_NAMES = [
    "Angle",
    "Dihedral",
    "Has Angle",
    "Has Dihedral",
]

GLOBAL_FEATURE_NAMES = [name for name, _ in Descriptors._descList]


def build_selected_data(df, sample_indices, cache_dir):
    subset = df.loc[sample_indices].copy()
    sample_tag = "_".join(str(int(idx)) for idx in sample_indices[:20]) if sample_indices else "empty"
    cache_path = Path(cache_dir) / f"interpretability_subset_{len(sample_indices)}_{sample_tag}.pt"
    return build_pair_dataset(
        subset,
        str(cache_path),
        include_k_targets=False,
        cache_desc="build interpretability subset",
    )


def to_batch(data, device):
    batch = Batch.from_data_list([copy.deepcopy(data)])
    return batch.to(device)


def predict_sample(model, data, device):
    batch = to_batch(data, device)
    with torch.no_grad():
        out = model(batch, return_dict=True)
    return {
        "prediction": float(out["log_gamma"].view(-1)[0].detach().cpu()),
        "k1": float(out["k1"].view(-1)[0].detach().cpu()),
        "k2": float(out["k2"].view(-1)[0].detach().cpu()),
    }


def get_atom_symbols(smiles):
    raw = Chem.MolFromSmiles(smiles)
    if raw is None:
        return []
    mol = Chem.AddHs(raw)
    if mol is None:
        return []
    return [atom.GetSymbol() for atom in mol.GetAtoms()]


def heavy_atom_count(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return 0
    return int(mol.GetNumHeavyAtoms())


def functional_group_importance(smiles, atom_scores):
    raw = Chem.MolFromSmiles(smiles)
    if raw is None:
        return []
    mol = Chem.AddHs(raw)
    if mol is None:
        return []
    abs_scores = np.abs(np.asarray(atom_scores[: mol.GetNumAtoms()], dtype=float))
    if abs_scores.size < mol.GetNumAtoms():
        abs_scores = np.pad(abs_scores, (0, mol.GetNumAtoms() - abs_scores.size))
    rows = []
    for name, smarts in FUNCTIONAL_GROUP_SMARTS.items():
        patt = Chem.MolFromSmarts(smarts)
        if patt is None:
            continue
        matches = mol.GetSubstructMatches(patt)
        if not matches:
            continue
        group_score = 0.0
        for match in matches:
            group_score += float(np.mean(abs_scores[list(match)]))
        rows.append(
            {
                "group": name,
                "importance": group_score / max(len(matches), 1),
                "match_count": len(matches),
            }
        )
    rows.sort(key=lambda item: item["importance"], reverse=True)
    return rows


def build_2d_depiction(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None, np.zeros((0, 2), dtype=float), []
    mol = Chem.Mol(mol)
    # Very small molecules like water look like a single dot without explicit hydrogens.
    # For these cases, add hydrogens so the panel still reads as a molecule.
    if mol.GetNumHeavyAtoms() <= 2:
        mol = Chem.AddHs(mol)
    rdDepictor.Compute2DCoords(mol)
    conf = mol.GetConformer()
    pos = []
    for atom_idx in range(mol.GetNumAtoms()):
        xyz = conf.GetAtomPosition(atom_idx)
        pos.append([float(xyz.x), float(xyz.y)])
    return mol, np.asarray(pos, dtype=float), [atom.GetSymbol() for atom in mol.GetAtoms()]


def normalize_scores(values):
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return values
    vmax = np.max(np.abs(values))
    if vmax < 1e-12:
        return np.zeros_like(values)
    return values / vmax


def unique_undirected_edges(edge_index):
    if edge_index is None or edge_index.numel() == 0:
        return []
    edges = []
    seen = set()
    edge_np = edge_index.detach().cpu().numpy()
    for src, dst in edge_np.T:
        key = tuple(sorted((int(src), int(dst))))
        if src == dst or key in seen:
            continue
        seen.add(key)
        edges.append(key)
    return edges


def remove_undirected_edge(edge_index, edge_attr, pair):
    if edge_index is None or edge_index.numel() == 0:
        return edge_index, edge_attr
    src, dst = pair
    mask = ~(
        ((edge_index[0] == src) & (edge_index[1] == dst))
        | ((edge_index[0] == dst) & (edge_index[1] == src))
    )
    new_edge_index = edge_index[:, mask]
    new_edge_attr = edge_attr[mask] if edge_attr is not None and edge_attr.size(0) == mask.numel() else edge_attr
    return new_edge_index, new_edge_attr


def clone_data(data):
    return copy.deepcopy(data)


def ablate_node(data, molecule_key, node_idx):
    edited = clone_data(data)
    if molecule_key == "solvent":
        edited.x_s[node_idx] = 0.0
        edited.charge_s[node_idx] = 0.0
        edited.atom_role_s[node_idx] = 0.0
        edited.aromatic_normal_s[node_idx] = 0.0
        mask = (edited.edge_index_s[0] != node_idx) & (edited.edge_index_s[1] != node_idx)
        edited.edge_index_s = edited.edge_index_s[:, mask]
        if edited.edge_attr_s.size(0) == mask.numel():
            edited.edge_attr_s = edited.edge_attr_s[mask]
        cross_mask = edited.cross_edge_index[0] != node_idx
        edited.cross_edge_index = edited.cross_edge_index[:, cross_mask]
    else:
        edited.x_v[node_idx] = 0.0
        edited.charge_v[node_idx] = 0.0
        edited.atom_role_v[node_idx] = 0.0
        edited.aromatic_normal_v[node_idx] = 0.0
        mask = (edited.edge_index_v[0] != node_idx) & (edited.edge_index_v[1] != node_idx)
        edited.edge_index_v = edited.edge_index_v[:, mask]
        if edited.edge_attr_v.size(0) == mask.numel():
            edited.edge_attr_v = edited.edge_attr_v[mask]
        cross_mask = edited.cross_edge_index[1] != node_idx
        edited.cross_edge_index = edited.cross_edge_index[:, cross_mask]
    return edited


def ablate_intra_edge(data, molecule_key, pair):
    edited = clone_data(data)
    if molecule_key == "solvent":
        edited.edge_index_s, edited.edge_attr_s = remove_undirected_edge(edited.edge_index_s, edited.edge_attr_s, pair)
    else:
        edited.edge_index_v, edited.edge_attr_v = remove_undirected_edge(edited.edge_index_v, edited.edge_attr_v, pair)
    return edited


def ablate_cross_edge(data, edge_idx):
    edited = clone_data(data)
    if edited.cross_edge_index is None or edited.cross_edge_index.numel() == 0:
        return edited
    mask = torch.ones(edited.cross_edge_index.size(1), dtype=torch.bool)
    mask[edge_idx] = False
    edited.cross_edge_index = edited.cross_edge_index[:, mask]
    return edited


def ablate_component(data, component):
    edited = clone_data(data)
    if component == "solvent_graph":
        edited.x_s.zero_()
        edited.charge_s.zero_()
        edited.atom_role_s.zero_()
        edited.aromatic_normal_s.zero_()
        edited.edge_index_s = edited.edge_index_s[:, :0]
        edited.edge_attr_s = edited.edge_attr_s[:0]
        edited.cross_edge_index = edited.cross_edge_index[:, edited.cross_edge_index[0] < 0]
    elif component == "solute_graph":
        edited.x_v.zero_()
        edited.charge_v.zero_()
        edited.atom_role_v.zero_()
        edited.aromatic_normal_v.zero_()
        edited.edge_index_v = edited.edge_index_v[:, :0]
        edited.edge_attr_v = edited.edge_attr_v[:0]
        edited.cross_edge_index = edited.cross_edge_index[:, edited.cross_edge_index[1] < 0]
    elif component == "cross_interaction":
        edited.cross_edge_index = edited.cross_edge_index[:, :0]
    elif component == "solvent_rdkit":
        edited.rdkit_s.zero_()
    elif component == "solute_rdkit":
        edited.rdkit_v.zero_()
    else:
        raise ValueError(f"Unknown component: {component}")
    return edited


def atom_importance_analysis(model, data, device, baseline_pred):
    solvent_scores = []
    for idx in range(data.x_s.size(0)):
        pred = predict_sample(model, ablate_node(data, "solvent", idx), device)["prediction"]
        solvent_scores.append(baseline_pred - pred)
    solute_scores = []
    for idx in range(data.x_v.size(0)):
        pred = predict_sample(model, ablate_node(data, "solute", idx), device)["prediction"]
        solute_scores.append(baseline_pred - pred)
    return np.asarray(solvent_scores), np.asarray(solute_scores)


def edge_importance_analysis(model, data, device, baseline_pred):
    solvent_pairs = unique_undirected_edges(data.edge_index_s)
    solvent_scores = []
    for pair in solvent_pairs:
        pred = predict_sample(model, ablate_intra_edge(data, "solvent", pair), device)["prediction"]
        solvent_scores.append(baseline_pred - pred)

    solute_pairs = unique_undirected_edges(data.edge_index_v)
    solute_scores = []
    for pair in solute_pairs:
        pred = predict_sample(model, ablate_intra_edge(data, "solute", pair), device)["prediction"]
        solute_scores.append(baseline_pred - pred)

    cross_edges = []
    cross_scores = []
    if data.cross_edge_index is not None and data.cross_edge_index.numel() > 0:
        cross_np = data.cross_edge_index.detach().cpu().numpy().T
        for edge_idx, (src, dst) in enumerate(cross_np):
            pred = predict_sample(model, ablate_cross_edge(data, edge_idx), device)["prediction"]
            cross_edges.append((int(src), int(dst)))
            cross_scores.append(baseline_pred - pred)

    return {
        "solvent_pairs": solvent_pairs,
        "solvent_scores": np.asarray(solvent_scores),
        "solute_pairs": solute_pairs,
        "solute_scores": np.asarray(solute_scores),
        "cross_pairs": cross_edges,
        "cross_scores": np.asarray(cross_scores),
    }


def graph_importance_analysis(model, data, device, baseline_pred):
    components = ["solvent_graph", "solute_graph", "cross_interaction", "solvent_rdkit", "solute_rdkit"]
    scores = {}
    for component in components:
        pred = predict_sample(model, ablate_component(data, component), device)["prediction"]
        scores[component] = baseline_pred - pred
    return scores


def classify_cross_edge(data, src, dst):
    role_s = data.atom_role_s[src].detach().cpu().numpy()
    role_v = data.atom_role_v[dst].detach().cpu().numpy()
    normal_s = data.aromatic_normal_s[src].detach().cpu().numpy()
    normal_v = data.aromatic_normal_v[dst].detach().cpu().numpy()
    donor_acceptor = (role_s[0] * role_v[1]) + (role_s[1] * role_v[0])
    aromatic_pair = role_s[2] * role_v[2]
    normal_align = aromatic_pair * abs(float(np.dot(normal_s, normal_v)))
    hydrophobic_pair = role_s[3] * role_v[3]
    polar_pair = role_s[4] * role_v[4]
    if donor_acceptor > 0.5:
        return "H-bond"
    if aromatic_pair > 0.5 and normal_align > 0.5:
        return "Pi-stacking"
    if hydrophobic_pair > 0.5:
        return "Hydrophobic"
    if polar_pair > 0.5:
        return "Polar"
    return "Mixed"


def ranked_records(scores, labels, molecule_label, top_k=10):
    order = np.argsort(-np.abs(scores))
    rows = []
    for rank, idx in enumerate(order[:top_k], start=1):
        rows.append(
            {
                "molecule": molecule_label,
                "rank": rank,
                "index": int(idx),
                "label": labels[idx] if idx < len(labels) else f"{molecule_label[0].upper()}{idx}",
                "importance": float(scores[idx]),
                "abs_importance": float(abs(scores[idx])),
            }
        )
    return rows


def mask_atom_feature_dim(data, dim_idx):
    edited = clone_data(data)
    if dim_idx < edited.x_s.size(1):
        edited.x_s[:, dim_idx] = 0.0
    if dim_idx < edited.x_v.size(1):
        edited.x_v[:, dim_idx] = 0.0
    return edited


def mask_bond_feature_dim(data, dim_idx):
    edited = clone_data(data)
    if edited.edge_attr_s.numel() > 0 and dim_idx < edited.edge_attr_s.size(1):
        edited.edge_attr_s[:, dim_idx] = 0.0
    if edited.edge_attr_v.numel() > 0 and dim_idx < edited.edge_attr_v.size(1):
        edited.edge_attr_v[:, dim_idx] = 0.0
    return edited


def mask_global_feature_dim(data, dim_idx):
    edited = clone_data(data)
    if edited.rdkit_s.numel() > 0 and dim_idx < edited.rdkit_s.size(1):
        edited.rdkit_s[:, dim_idx] = 0.0
    if edited.rdkit_v.numel() > 0 and dim_idx < edited.rdkit_v.size(1):
        edited.rdkit_v[:, dim_idx] = 0.0
    return edited


def feature_importance_analysis(model, data, device, baseline_pred):
    atom_scores = []
    for dim_idx in range(min(len(ATOM_FEATURE_NAMES), data.x_s.size(1), data.x_v.size(1))):
        pred = predict_sample(model, mask_atom_feature_dim(data, dim_idx), device)["prediction"]
        atom_scores.append(baseline_pred - pred)

    bond_scores = []
    bond_dim = 0
    if data.edge_attr_s is not None and data.edge_attr_s.dim() == 2:
        bond_dim = max(bond_dim, data.edge_attr_s.size(1))
    if data.edge_attr_v is not None and data.edge_attr_v.dim() == 2:
        bond_dim = max(bond_dim, data.edge_attr_v.size(1))
    for dim_idx in range(min(len(BOND_FEATURE_NAMES), bond_dim)):
        pred = predict_sample(model, mask_bond_feature_dim(data, dim_idx), device)["prediction"]
        bond_scores.append(baseline_pred - pred)

    global_scores = []
    for dim_idx in range(min(len(GLOBAL_FEATURE_NAMES), data.rdkit_s.size(1), data.rdkit_v.size(1))):
        pred = predict_sample(model, mask_global_feature_dim(data, dim_idx), device)["prediction"]
        global_scores.append(baseline_pred - pred)

    return (
        np.asarray(atom_scores, dtype=float),
        np.asarray(bond_scores, dtype=float),
        np.asarray(global_scores, dtype=float),
    )


def plot_node_panel(ax, pos, scores, labels, title):
    pos = pos.detach().cpu().numpy()
    scores = np.asarray(scores, dtype=float)
    norm_scores = normalize_scores(scores)
    color_map = plt.get_cmap("RdBu_r")
    sizes = 140 + 260 * np.abs(norm_scores)
    sc = ax.scatter(pos[:, 0], pos[:, 1], c=norm_scores, s=sizes, cmap=color_map, vmin=-1.0, vmax=1.0, edgecolor="black", linewidth=0.6)
    top_idx = np.argsort(-np.abs(scores))[: min(8, len(scores))]
    for idx in top_idx:
        ax.text(pos[idx, 0], pos[idx, 1], labels[idx], ha="center", va="center", fontsize=10, color="black")
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect("equal", adjustable="box")
    return sc


def draw_molecule_heat(ax, smiles, atom_scores, title=None):
    mol, pos, symbols = build_2d_depiction(smiles)
    if mol is None or pos.size == 0:
        ax.text(0.5, 0.5, "Molecule unavailable", ha="center", va="center")
        ax.set_axis_off()
        return
    scores = np.asarray(atom_scores[: len(symbols)], dtype=float)
    if scores.size < len(symbols):
        scores = np.pad(scores, (0, len(symbols) - scores.size))
    norm = normalize_scores(scores)

    x_min, y_min = pos.min(axis=0) - 0.8
    x_max, y_max = pos.max(axis=0) + 0.8
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 240), np.linspace(y_min, y_max, 240))
    intensity = np.zeros_like(xx)
    for (x, y), value in zip(pos, np.abs(norm)):
        sigma = 0.38
        intensity += value * np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2 * sigma**2))
    if np.max(intensity) > 1e-8:
        intensity = intensity / np.max(intensity)
        ax.contourf(xx, yy, intensity, levels=np.linspace(0.08, 1.0, 10), cmap="Greens", alpha=0.62, antialiased=True)

    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        ax.plot(
            [pos[i, 0], pos[j, 0]],
            [pos[i, 1], pos[j, 1]],
            color="#b7c4b7",
            linewidth=2.2,
            zorder=2,
        )

    hetero_colors = {
        "O": "#ff595e",
        "N": "#2b6cb0",
        "Cl": "#2ca02c",
        "Br": "#8c564b",
        "S": "#ff7f0e",
        "F": "#2ca02c",
        "P": "#9467bd",
    }
    for idx, ((x, y), symbol, value) in enumerate(zip(pos, symbols, norm)):
        radius = 0.11 + 0.12 * abs(value)
        facecolor = hetero_colors.get(symbol, "#5a5a5a")
        alpha = 0.95 if symbol != "C" else 0.30
        circle = plt.Circle((x, y), radius=radius, facecolor=facecolor, edgecolor="white", linewidth=0.8, alpha=alpha, zorder=3)
        ax.add_patch(circle)
        if symbol != "C":
            ax.text(x + 0.07, y + 0.05, symbol, color=facecolor, fontsize=10, ha="left", va="center", zorder=4)

    if title:
        ax.set_title(title, color="#0077cc", fontsize=13, pad=2)
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect("equal", adjustable="box")
    for spine in ax.spines.values():
        spine.set_visible(False)


def plot_edge_panel(ax, pos, pairs, scores, labels, title):
    pos = pos.detach().cpu().numpy()
    scores = np.asarray(scores, dtype=float)
    ax.scatter(pos[:, 0], pos[:, 1], s=70, c="#3b528b", edgecolor="white", linewidth=0.8, zorder=3)
    if len(scores) > 0:
        norm_scores = normalize_scores(scores)
        max_width = 5.0
        for pair, value, scaled in zip(pairs, scores, norm_scores):
            i, j = pair
            ax.plot(
                [pos[i, 0], pos[j, 0]],
                [pos[i, 1], pos[j, 1]],
                color=plt.get_cmap("coolwarm")((scaled + 1.0) / 2.0),
                linewidth=1.0 + max_width * abs(scaled),
                alpha=0.85,
                zorder=2,
            )
    top_nodes = np.argsort(-np.abs(scores))[: min(4, len(scores))] if len(scores) > 0 else []
    for edge_rank in top_nodes:
        i, j = pairs[edge_rank]
        mid_x = (pos[i, 0] + pos[j, 0]) / 2.0
        mid_y = (pos[i, 1] + pos[j, 1]) / 2.0
        ax.text(mid_x, mid_y, f"{labels[i]}-{labels[j]}", fontsize=9, ha="center", va="center")
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect("equal", adjustable="box")


def molecule_graph_3d(smiles):
    raw = Chem.MolFromSmiles(smiles)
    if raw is None:
        return None, [], []
    mol = Chem.AddHs(raw)
    if mol is None:
        return None, [], []
    symbols = [atom.GetSymbol() for atom in mol.GetAtoms()]
    bonds = [(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()) for bond in mol.GetBonds()]
    return mol, symbols, bonds


def edge_score_map(pairs, scores):
    mapping = {}
    for pair, score in zip(pairs, scores):
        mapping[tuple(sorted((int(pair[0]), int(pair[1]))))] = float(score)
    return mapping


def draw_molecule_3d(ax, coords, symbols, bond_pairs, atom_scores, bond_scores, title=None, shift=None):
    coords = np.asarray(coords, dtype=float)
    if coords.size == 0:
        ax.set_axis_off()
        return
    if shift is not None:
        coords = coords + np.asarray(shift, dtype=float)

    atom_scores = np.asarray(atom_scores[: len(symbols)], dtype=float)
    if atom_scores.size < len(symbols):
        atom_scores = np.pad(atom_scores, (0, len(symbols) - atom_scores.size))
    norm_atom = normalize_scores(atom_scores)
    cmap = plt.get_cmap("Greens")

    for i, j in bond_pairs:
        score = bond_scores.get(tuple(sorted((i, j))), 0.0)
        scaled = abs(score)
        ax.plot(
            [coords[i, 0], coords[j, 0]],
            [coords[i, 1], coords[j, 1]],
            [coords[i, 2], coords[j, 2]],
            color=plt.get_cmap("copper")(min(1.0, 0.25 + scaled * 2.0)),
            linewidth=1.0 + min(3.5, scaled * 6.0),
            alpha=0.9,
        )

    color_map = {
        "O": "#e63946",
        "N": "#1d4ed8",
        "S": "#f59e0b",
        "Cl": "#2a9d8f",
        "Br": "#8d5524",
        "F": "#2a9d8f",
        "P": "#7c3aed",
        "H": "#d4d4d4",
    }
    for idx, (x, y, z) in enumerate(coords):
        score = max(0.0, abs(norm_atom[idx]))
        base_color = color_map.get(symbols[idx], "#4b5563")
        size = 14 + 52 * score if symbols[idx] != "H" else 4 + 8 * score
        ax.scatter([x], [y], [z], s=size, color=base_color, edgecolors="white", linewidths=0.3, alpha=0.95)
        if symbols[idx] != "C" and symbols[idx] != "H":
            ax.text(x, y, z, symbols[idx], fontsize=7, color=base_color)
        if score > 0.55 and symbols[idx] != "H":
            ax.scatter([x], [y], [z], s=size * 4.0, color=cmap(0.55 + 0.45 * score), alpha=0.08, edgecolors="none")

    if title:
        ax.set_title(title, pad=1, color="#0077cc", fontsize=10)
    ax.set_axis_off()


def trim_text(text, max_len=28):
    text = str(text)
    if len(text) <= max_len:
        return text
    return text[: max_len - 3] + "..."


def pretty_molecule_label(smiles, max_len=22):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return trim_text(smiles, max_len)
    try:
        formula = rdMolDescriptors.CalcMolFormula(mol)
    except Exception:
        formula = ""
    short_smiles = trim_text(smiles, max_len)
    if formula and formula != short_smiles:
        return f"{formula} | {short_smiles}"
    return short_smiles


def set_equal_3d(ax, coords_list, zoom: float = 1.0):
    """Equal 3D axis limits around merged coordinates. ``zoom`` > 1 tightens limits (structure appears larger)."""
    all_coords = [np.asarray(coords, dtype=float) for coords in coords_list if np.asarray(coords).size > 0]
    if not all_coords:
        return
    merged = np.concatenate(all_coords, axis=0)
    mins = merged.min(axis=0)
    maxs = merged.max(axis=0)
    center = (mins + maxs) / 2.0
    radius = max((maxs - mins).max() / 2.0, 1.0)
    if zoom > 1.0:
        radius = max(radius / float(zoom), 1e-6)
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)
    ax.view_init(elev=20, azim=35)


def plot_cross_heatmap(ax, cross_pairs, cross_scores, solvent_labels, solute_labels):
    if len(cross_pairs) == 0:
        ax.text(0.5, 0.5, "No cross edges", ha="center", va="center")
        ax.set_axis_off()
        return
    solvent_size = len(solvent_labels)
    solute_size = len(solute_labels)
    matrix = np.zeros((solvent_size, solute_size), dtype=float)
    for (src, dst), score in zip(cross_pairs, cross_scores):
        matrix[src, dst] += score
    im = ax.imshow(matrix, cmap="RdBu_r", aspect="auto")
    ax.set_title("Cross-Edge Importance")
    top_solvent = min(solvent_size, 12)
    top_solute = min(solute_size, 12)
    ax.set_yticks(range(top_solvent))
    ax.set_yticklabels(solvent_labels[:top_solvent])
    ax.set_xticks(range(top_solute))
    ax.set_xticklabels(solute_labels[:top_solute], rotation=60, ha="right")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)


def plot_graph_bar(ax, graph_scores):
    names = list(graph_scores.keys())
    values = np.asarray([graph_scores[name] for name in names], dtype=float)
    colors = ["#1f77b4" if value >= 0 else "#d62728" for value in values]
    ax.bar(range(len(names)), values, color=colors, edgecolor="black", linewidth=0.8)
    ax.axhline(0.0, color="black", linewidth=1.0)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(
        ["Solvent\nGraph", "Solute\nGraph", "Cross\nEdges", "Solvent\nRDKit", "Solute\nRDKit"],
        rotation=0,
    )
    ax.set_title("Graph-Level Component Importance")
    ax.set_ylabel("Prediction change")


def plot_component_summary(ax, summary_df):
    if summary_df.empty or "component" not in summary_df.columns:
        ax.text(0.5, 0.5, "No graph-level summary", ha="center", va="center")
        ax.set_axis_off()
        return
    grouped = summary_df.groupby("component", as_index=False)["abs_importance"].mean()
    grouped = grouped.sort_values("abs_importance", ascending=False)
    ax.barh(grouped["component"], grouped["abs_importance"], color="#2a9d8f", edgecolor="black", linewidth=0.8)
    ax.invert_yaxis()
    ax.set_xlabel("Mean absolute prediction change")
    ax.set_title("Average Graph-Level Importance")


def plot_interaction_summary(ax, interaction_df):
    if interaction_df.empty or "interaction_type" not in interaction_df.columns:
        ax.text(0.5, 0.5, "No cross-edge summary", ha="center", va="center")
        ax.set_axis_off()
        return
    grouped = interaction_df.groupby("interaction_type", as_index=False)["abs_importance"].sum()
    grouped = grouped.sort_values("abs_importance", ascending=False)
    ax.bar(grouped["interaction_type"], grouped["abs_importance"], color="#e9c46a", edgecolor="black", linewidth=0.8)
    ax.set_ylabel("Summed absolute importance")
    ax.set_title("Top Cross-Interaction Types")
    ax.tick_params(axis="x", rotation=20)


def analyze_single_sample(model, device, row, data, output_dir):
    sample_dir = Path(output_dir) / f"sample_{int(row.name)}"
    ensure_dir(sample_dir)

    baseline = predict_sample(model, data, device)
    baseline_pred = baseline["prediction"]
    y_true = float(row["log-gamma"]) if "log-gamma" in row.index else math.nan

    solvent_labels = [f"S{i}:{sym}" for i, sym in enumerate(get_atom_symbols(row["Solvent_SMILES"]))]
    solute_labels = [f"V{i}:{sym}" for i, sym in enumerate(get_atom_symbols(row["Solute_SMILES"]))]

    atom_scores_s, atom_scores_v = atom_importance_analysis(model, data, device, baseline_pred)
    edge_stats = edge_importance_analysis(model, data, device, baseline_pred)
    graph_scores = graph_importance_analysis(model, data, device, baseline_pred)
    atom_feature_scores, bond_feature_scores, global_feature_scores = feature_importance_analysis(model, data, device, baseline_pred)
    solvent_fg = functional_group_importance(row["Solvent_SMILES"], atom_scores_s)
    solute_fg = functional_group_importance(row["Solute_SMILES"], atom_scores_v)

    atom_rows = []
    atom_rows.extend(ranked_records(atom_scores_s, solvent_labels, "solvent"))
    atom_rows.extend(ranked_records(atom_scores_v, solute_labels, "solute"))
    atom_df = pd.DataFrame(atom_rows)
    atom_df.to_csv(sample_dir / "node_importance.csv", index=False)

    edge_rows = []
    for pair, score in zip(edge_stats["solvent_pairs"], edge_stats["solvent_scores"]):
        edge_rows.append(
            {
                "edge_type": "solvent_intra",
                "src": int(pair[0]),
                "dst": int(pair[1]),
                "src_label": solvent_labels[pair[0]],
                "dst_label": solvent_labels[pair[1]],
                "importance": float(score),
                "abs_importance": float(abs(score)),
            }
        )
    for pair, score in zip(edge_stats["solute_pairs"], edge_stats["solute_scores"]):
        edge_rows.append(
            {
                "edge_type": "solute_intra",
                "src": int(pair[0]),
                "dst": int(pair[1]),
                "src_label": solute_labels[pair[0]],
                "dst_label": solute_labels[pair[1]],
                "importance": float(score),
                "abs_importance": float(abs(score)),
            }
        )
    for pair, score in zip(edge_stats["cross_pairs"], edge_stats["cross_scores"]):
        edge_rows.append(
            {
                "edge_type": "cross",
                "src": int(pair[0]),
                "dst": int(pair[1]),
                "src_label": solvent_labels[pair[0]],
                "dst_label": solute_labels[pair[1]],
                "interaction_type": classify_cross_edge(data, pair[0], pair[1]),
                "importance": float(score),
                "abs_importance": float(abs(score)),
            }
        )
    edge_df = pd.DataFrame(edge_rows).sort_values("abs_importance", ascending=False)
    edge_df.to_csv(sample_dir / "edge_importance.csv", index=False)

    graph_df = pd.DataFrame(
        [
            {"component": name, "importance": float(value), "abs_importance": float(abs(value))}
            for name, value in graph_scores.items()
        ]
    ).sort_values("abs_importance", ascending=False)
    graph_df.to_csv(sample_dir / "graph_importance.csv", index=False)

    feature_rows = []
    for name, score in zip(ATOM_FEATURE_NAMES, atom_feature_scores):
        feature_rows.append({"feature_group": "atom", "feature": name, "importance": float(score), "abs_importance": float(abs(score))})
    for name, score in zip(BOND_FEATURE_NAMES, bond_feature_scores):
        feature_rows.append({"feature_group": "bond", "feature": name, "importance": float(score), "abs_importance": float(abs(score))})
    global_order = np.argsort(-np.abs(global_feature_scores))[:15]
    for idx in global_order:
        feature_rows.append(
            {
                "feature_group": "global",
                "feature": GLOBAL_FEATURE_NAMES[idx],
                "importance": float(global_feature_scores[idx]),
                "abs_importance": float(abs(global_feature_scores[idx])),
            }
        )
    feature_df = pd.DataFrame(feature_rows)
    feature_df.to_csv(sample_dir / "feature_importance.csv", index=False)
    pd.DataFrame(
        [
            {"molecule": "solvent", "group": item["group"], "importance": item["importance"], "match_count": item["match_count"]}
            for item in solvent_fg
        ]
        + [
            {"molecule": "solute", "group": item["group"], "importance": item["importance"], "match_count": item["match_count"]}
            for item in solute_fg
        ]
    ).to_csv(sample_dir / "functional_group_importance.csv", index=False)

    fig, axes = plt.subplots(2, 3, figsize=(18, 10), constrained_layout=True)
    sc1 = plot_node_panel(axes[0, 0], data.pos_s, atom_scores_s, solvent_labels, "Solvent Node Importance")
    sc2 = plot_node_panel(axes[0, 1], data.pos_v, atom_scores_v, solute_labels, "Solute Node Importance")
    plot_graph_bar(axes[0, 2], graph_scores)
    plot_edge_panel(axes[1, 0], data.pos_s, edge_stats["solvent_pairs"], edge_stats["solvent_scores"], solvent_labels, "Solvent Bond Importance")
    plot_edge_panel(axes[1, 1], data.pos_v, edge_stats["solute_pairs"], edge_stats["solute_scores"], solute_labels, "Solute Bond Importance")
    plot_cross_heatmap(axes[1, 2], edge_stats["cross_pairs"], edge_stats["cross_scores"], solvent_labels, solute_labels)
    fig.colorbar(sc1, ax=axes[0, 0], fraction=0.046, pad=0.04)
    fig.colorbar(sc2, ax=axes[0, 1], fraction=0.046, pad=0.04)
    fig.suptitle(
        f"Sample {int(row.name)} | True={y_true:.4f} | Pred={baseline_pred:.4f} | AbsErr={abs(y_true - baseline_pred):.4f}",
        fontsize=16,
    )
    fig.savefig(sample_dir / "interpretability_overview.png", bbox_inches="tight")
    plt.close(fig)

    summary = {
        "sample_index": int(row.name),
        "solvent_smiles": row["Solvent_SMILES"],
        "solute_smiles": row["Solute_SMILES"],
        "true_log_gamma": y_true,
        "pred_log_gamma": baseline_pred,
        "abs_error": abs(y_true - baseline_pred) if not math.isnan(y_true) else math.nan,
        "k1": baseline["k1"],
        "k2": baseline["k2"],
    }
    with open(sample_dir / "summary.json", "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, ensure_ascii=False)

    return summary, atom_df, edge_df, graph_df, feature_df, atom_scores_s, atom_scores_v, edge_stats, solvent_fg, solute_fg


def plot_feature_heatmaps(output_dir, sample_summaries, feature_frames):
    if not feature_frames:
        return
    summary_dir = Path(output_dir)
    sample_ids = [str(item["sample_index"]) for item in sample_summaries]

    atom_matrix = pd.DataFrame(index=ATOM_FEATURE_NAMES, columns=sample_ids, data=0.0)
    bond_matrix = pd.DataFrame(index=BOND_FEATURE_NAMES, columns=sample_ids, data=0.0)
    global_candidates = {}

    for sample_summary, feature_df in zip(sample_summaries, feature_frames):
        sample_id = str(sample_summary["sample_index"])
        for _, row in feature_df.iterrows():
            if row["feature_group"] == "atom" and row["feature"] in atom_matrix.index:
                atom_matrix.loc[row["feature"], sample_id] = row["abs_importance"]
            elif row["feature_group"] == "bond" and row["feature"] in bond_matrix.index:
                bond_matrix.loc[row["feature"], sample_id] = row["abs_importance"]
            elif row["feature_group"] == "global":
                global_candidates.setdefault(row["feature"], 0.0)
                global_candidates[row["feature"]] += float(row["abs_importance"])

    top_global = [name for name, _ in sorted(global_candidates.items(), key=lambda item: item[1], reverse=True)[:12]]
    global_matrix = pd.DataFrame(index=top_global, columns=sample_ids, data=0.0)
    for sample_summary, feature_df in zip(sample_summaries, feature_frames):
        sample_id = str(sample_summary["sample_index"])
        selected_df = feature_df[(feature_df["feature_group"] == "global") & (feature_df["feature"].isin(top_global))]
        for _, row in selected_df.iterrows():
            global_matrix.loc[row["feature"], sample_id] = row["abs_importance"]

    fig, axes = plt.subplots(1, 3, figsize=(18, 7), constrained_layout=True)
    for ax, matrix, title in zip(
        axes,
        [atom_matrix, bond_matrix, global_matrix],
        ["Atom Features", "Bond Features", "Global Features"],
    ):
        im = ax.imshow(matrix.to_numpy(dtype=float), cmap="YlGn", aspect="auto")
        ax.set_title(title)
        ax.set_xticks(range(len(matrix.columns)))
        ax.set_xticklabels(matrix.columns)
        ax.set_yticks(range(len(matrix.index)))
        ax.set_yticklabels(matrix.index)
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                value = matrix.iat[i, j]
                ax.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=8, color="black")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.savefig(summary_dir / "feature_importance_heatmaps.png", bbox_inches="tight")
    plt.close(fig)


def plot_representative_panel(output_dir, sample_summaries, selected_rows, atom_scores_pairs):
    if not sample_summaries:
        return
    n = len(sample_summaries)
    ncols = min(3, n)
    nrows = int(math.ceil(n / ncols))
    fig = plt.figure(figsize=(6.1 * ncols, 7.6 * nrows))
    for panel_idx, (sample_summary, (_, row), (atom_scores_s, atom_scores_v)) in enumerate(
        zip(sample_summaries, selected_rows.iterrows(), atom_scores_pairs)
    ):
        outer_ax = fig.add_subplot(nrows, ncols, panel_idx + 1)
        outer_ax.set_axis_off()
        panel = FancyBboxPatch(
            (0.02, 0.02),
            0.96,
            0.96,
            transform=outer_ax.transAxes,
            boxstyle="round,pad=0.02,rounding_size=0.08",
            linewidth=1.1,
            edgecolor="#cfd8dc",
            facecolor="white",
        )
        outer_ax.add_patch(panel)

        bbox = outer_ax.get_position()
        left, bottom, width, height = bbox.x0, bbox.y0, bbox.width, bbox.height
        ax_top = fig.add_axes([left + 0.08 * width, bottom + 0.54 * height, 0.84 * width, 0.26 * height])
        ax_bottom = fig.add_axes([left + 0.08 * width, bottom + 0.23 * height, 0.84 * width, 0.26 * height])
        text_ax = fig.add_axes([left + 0.08 * width, bottom + 0.04 * height, 0.84 * width, 0.14 * height])
        text_ax.set_axis_off()

        draw_molecule_heat(ax_top, row["Solute_SMILES"], atom_scores_v, "Solute")
        draw_molecule_heat(ax_bottom, row["Solvent_SMILES"], atom_scores_s, "Solvent")
        text_ax.text(
            0.5,
            0.72,
            f"lnγ$^{{\\prime}}_{{True}}$ = {sample_summary['true_log_gamma']:.4f}",
            ha="center",
            va="center",
            fontsize=12,
        )
        text_ax.text(
            0.5,
            0.43,
            f"lnγ$^{{\\prime}}_{{Pred}}$ = {sample_summary['pred_log_gamma']:.4f}",
            ha="center",
            va="center",
            fontsize=12,
        )
        text_ax.text(
            0.5,
            0.14,
            f"T = {float(row['T']) + 273.15:.2f} K",
            ha="center",
            va="center",
            fontsize=12,
        )
    total_slots = nrows * ncols
    for empty_idx in range(n, total_slots):
        empty_ax = fig.add_subplot(nrows, ncols, empty_idx + 1)
        empty_ax.set_axis_off()
    fig.savefig(Path(output_dir) / "representative_molecule_panel.png", bbox_inches="tight")
    plt.close(fig)


def plot_representative_panel_nature(output_dir, sample_summaries, selected_rows, atom_scores_pairs):
    if not sample_summaries:
        return
    n = len(sample_summaries)
    ncols = min(3, n)
    nrows = int(math.ceil(n / ncols))
    fig = plt.figure(figsize=(5.5 * ncols, 6.6 * nrows))
    for panel_idx, (sample_summary, (_, row), (atom_scores_s, atom_scores_v)) in enumerate(
        zip(sample_summaries, selected_rows.iterrows(), atom_scores_pairs)
    ):
        outer_ax = fig.add_subplot(nrows, ncols, panel_idx + 1)
        outer_ax.set_axis_off()
        panel = FancyBboxPatch(
            (0.02, 0.02),
            0.96,
            0.96,
            transform=outer_ax.transAxes,
            boxstyle="round,pad=0.012,rounding_size=0.055",
            linewidth=0.9,
            edgecolor="#d9e1e6",
            facecolor="white",
        )
        outer_ax.add_patch(panel)

        bbox = outer_ax.get_position()
        left, bottom, width, height = bbox.x0, bbox.y0, bbox.width, bbox.height
        ax_top = fig.add_axes([left + 0.06 * width, bottom + 0.55 * height, 0.88 * width, 0.23 * height])
        ax_bottom = fig.add_axes([left + 0.06 * width, bottom + 0.27 * height, 0.88 * width, 0.23 * height])
        text_ax = fig.add_axes([left + 0.06 * width, bottom + 0.07 * height, 0.88 * width, 0.11 * height])
        text_ax.set_axis_off()

        draw_molecule_heat(ax_top, row["Solute_SMILES"], atom_scores_v, "Solute")
        draw_molecule_heat(ax_bottom, row["Solvent_SMILES"], atom_scores_s, "Solvent")
        text_ax.text(0.5, 0.72, f"lnγ$^{{\\prime}}_{{True}}$ = {sample_summary['true_log_gamma']:.4f}", ha="center", va="center", fontsize=12)
        text_ax.text(0.5, 0.43, f"lnγ$^{{\\prime}}_{{Pred}}$ = {sample_summary['pred_log_gamma']:.4f}", ha="center", va="center", fontsize=12)
        text_ax.text(0.5, 0.14, f"T = {float(row['T']) + 273.15:.2f} K", ha="center", va="center", fontsize=12)

    total_slots = nrows * ncols
    for empty_idx in range(n, total_slots):
        empty_ax = fig.add_subplot(nrows, ncols, empty_idx + 1)
        empty_ax.set_axis_off()
    fig.subplots_adjust(left=0.025, right=0.975, top=0.985, bottom=0.025, wspace=0.06, hspace=0.08)
    fig.savefig(Path(output_dir) / "representative_molecule_panel.png", bbox_inches="tight")
    plt.close(fig)


# Shared figure geometry for the two 3D gallery PNGs (per-cell layout matches joint-style slots).
GALLERY_3D_FIGSIZE_CELL = (4.5, 4.35)
GALLERY_3D_SUBPLOT_ADJUST = dict(left=0.02, right=0.985, top=0.985, bottom=0.02, wspace=0.08, hspace=0.34)
# Only scales the 3D view (tighter x/y/z limits); does not change slot layout or fonts.
GALLERY_3D_MOL_ZOOM = 1.8


def _render_3d_gallery_cell_joint_layout(
    fig: plt.Figure,
    nrows: int,
    ncols: int,
    cell_idx: int,
    row,
    sample_summary: dict,
    entry: dict,
    *,
    extra_info_lines: tuple[str, ...] | None = None,
) -> None:
    """One gallery cell: title strip + middle 3D (dual mol + optional cross edges) + bottom info box."""
    slot_ax = fig.add_subplot(nrows, ncols, cell_idx + 1)
    slot_ax.set_axis_off()
    bbox = slot_ax.get_position()
    left, bottom, width, height = bbox.x0, bbox.y0, bbox.width, bbox.height

    title_ax = fig.add_axes([left + 0.02 * width, bottom + 0.90 * height, 0.96 * width, 0.08 * height])
    title_ax.set_axis_off()
    mol_ax = fig.add_axes([left + 0.03 * width, bottom + 0.27 * height, 0.94 * width, 0.60 * height], projection="3d")
    info_ax = fig.add_axes([left + 0.03 * width, bottom + 0.02 * height, 0.94 * width, 0.20 * height])
    info_ax.set_axis_off()
    ax = mol_ax

    atom_scores_s = entry["atom_scores_s"]
    atom_scores_v = entry["atom_scores_v"]
    edge_stats = entry["edge_stats"]

    _mol_s, symbols_s, bonds_s = molecule_graph_3d(row["Solvent_SMILES"])
    _mol_v, symbols_v, bonds_v = molecule_graph_3d(row["Solute_SMILES"])
    coords_s = np.asarray(entry["coords_s"], dtype=float)
    coords_v = np.asarray(entry["coords_v"], dtype=float)

    draw_molecule_3d(
        ax,
        coords_s,
        symbols_s,
        bonds_s,
        atom_scores_s,
        edge_score_map(edge_stats["solvent_pairs"], np.abs(edge_stats["solvent_scores"])),
        title=None,
    )
    draw_molecule_3d(
        ax,
        coords_v,
        symbols_v,
        bonds_v,
        atom_scores_v,
        edge_score_map(edge_stats["solute_pairs"], np.abs(edge_stats["solute_scores"])),
        title=None,
    )

    cross_pairs = edge_stats.get("cross_pairs", [])
    cross_scores = np.asarray(edge_stats.get("cross_scores", []), dtype=float)
    if len(cross_pairs) > 0:
        norm_cross = normalize_scores(cross_scores)
        for (src, dst), scaled in zip(cross_pairs, norm_cross):
            if src >= len(coords_s) or dst >= len(coords_v):
                continue
            ax.plot(
                [coords_s[src, 0], coords_v[dst, 0]],
                [coords_s[src, 1], coords_v[dst, 1]],
                [coords_s[src, 2], coords_v[dst, 2]],
                color=plt.get_cmap("viridis")(0.35 + 0.55 * abs(float(scaled))),
                linewidth=0.4 + 1.6 * abs(float(scaled)),
                alpha=0.24 + 0.25 * abs(float(scaled)),
            )

    set_equal_3d(ax, [coords_s, coords_v], zoom=GALLERY_3D_MOL_ZOOM)

    title_line = f"#{sample_summary['sample_index']}  |  T={float(row['T']) + 273.15:.2f} K"
    title_ax.text(0.5, 0.5, title_line, ha="center", va="center", fontsize=15)
    lines = [
        f"True: {sample_summary['true_log_gamma']:.3f}",
        f"Pred: {sample_summary['pred_log_gamma']:.3f}",
        f"Solute: {pretty_molecule_label(row['Solute_SMILES'], 18)}",
        f"Solvent: {pretty_molecule_label(row['Solvent_SMILES'], 18)}",
    ]
    if extra_info_lines:
        lines.extend(extra_info_lines)
    info_text = "\n".join(lines)
    info_ax.text(
        0.02,
        0.95,
        info_text,
        fontsize=12,
        va="top",
        ha="left",
        bbox=dict(boxstyle="round,pad=0.22", facecolor="white", edgecolor="#d9e1e6", linewidth=0.7, alpha=0.96),
    )


def plot_3d_gallery(output_dir, sample_summaries, selected_rows, gallery_entries, nrows=8, ncols=6):
    if not sample_summaries:
        return
    total = min(len(sample_summaries), nrows * ncols)
    cw, ch = GALLERY_3D_FIGSIZE_CELL
    fig = plt.figure(figsize=(cw * ncols, ch * nrows))
    for idx, (sample_summary, (_, row), entry) in enumerate(zip(sample_summaries[:total], selected_rows.iterrows(), gallery_entries[:total])):
        extras: list[str] = [f"|AE|: {sample_summary['abs_error']:.3f}"]
        fg_bits: list[str] = []
        if entry.get("solute_fg"):
            fg_bits.append(f"Solu: {entry['solute_fg'][0]['group']}")
        if entry.get("solvent_fg"):
            fg_bits.append(f"Solv: {entry['solvent_fg'][0]['group']}")
        if fg_bits:
            extras.append("FG: " + " | ".join(fg_bits))
        _render_3d_gallery_cell_joint_layout(
            fig, nrows, ncols, idx, row, sample_summary, entry, extra_info_lines=tuple(extras)
        )

    total_slots = nrows * ncols
    for empty_idx in range(total, total_slots):
        ax = fig.add_subplot(nrows, ncols, empty_idx + 1)
        ax.set_axis_off()
    fig.subplots_adjust(**GALLERY_3D_SUBPLOT_ADJUST)
    fig.savefig(Path(output_dir) / "representative_molecule_panel_3d_8x6.png", bbox_inches="tight")
    plt.close(fig)


def plot_3d_gallery_joint(output_dir, sample_summaries, selected_rows, gallery_entries, nrows=8, ncols=6):
    if not sample_summaries:
        return
    total = min(len(sample_summaries), nrows * ncols)
    cw, ch = GALLERY_3D_FIGSIZE_CELL
    fig = plt.figure(figsize=(cw * ncols, ch * nrows))
    for idx, (sample_summary, (_, row), entry) in enumerate(zip(sample_summaries[:total], selected_rows.iterrows(), gallery_entries[:total])):
        _render_3d_gallery_cell_joint_layout(fig, nrows, ncols, idx, row, sample_summary, entry, extra_info_lines=None)

    total_slots = nrows * ncols
    for empty_idx in range(total, total_slots):
        ax = fig.add_subplot(nrows, ncols, empty_idx + 1)
        ax.set_axis_off()
    fig.subplots_adjust(**GALLERY_3D_SUBPLOT_ADJUST)
    fig.savefig(Path(output_dir) / "representative_molecule_panel_3d_joint_8x6.png", bbox_inches="tight")
    plt.close(fig)


def choose_samples(df, predictions_df, selection, num_samples, exclude_small_solvents=True):
    merged = df.copy()
    if predictions_df is not None and "pred_log-gamma" in predictions_df.columns:
        _assert_prediction_rows_aligned(df, predictions_df)
        pred_cols = [col for col in predictions_df.columns if col in {"pred_log-gamma", "abs_error"}]
        merged = merged.join(predictions_df[pred_cols], how="left")
        if "abs_error" not in merged.columns and "pred_log-gamma" in merged.columns and "log-gamma" in merged.columns:
            merged["abs_error"] = (merged["log-gamma"] - merged["pred_log-gamma"]).abs()
    if exclude_small_solvents and "Solvent_SMILES" in merged.columns:
        merged = merged[merged["Solvent_SMILES"].map(heavy_atom_count) >= 3].copy()
        if merged.empty:
            merged = df.copy()
    if selection == "top_error" and "abs_error" in merged.columns:
        chosen = merged.sort_values("abs_error", ascending=False).head(num_samples)
    elif selection == "lowest_error" and "abs_error" in merged.columns:
        chosen = merged.sort_values("abs_error", ascending=True).head(num_samples)
    else:
        chosen = merged.head(num_samples)
    return chosen


def save_global_summary(output_dir, sample_summaries, atom_df, edge_df, graph_df):
    summary_dir = Path(output_dir)
    pd.DataFrame(sample_summaries).to_csv(summary_dir / "sample_summary.csv", index=False)
    atom_df.to_csv(summary_dir / "all_node_importance.csv", index=False)
    edge_df.to_csv(summary_dir / "all_edge_importance.csv", index=False)
    graph_df.to_csv(summary_dir / "all_graph_importance.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)
    plot_component_summary(axes[0], graph_df)
    if not edge_df.empty and "edge_type" in edge_df.columns:
        cross_df = edge_df[edge_df["edge_type"] == "cross"].copy()
    else:
        cross_df = pd.DataFrame()
    if not cross_df.empty and "interaction_type" in cross_df.columns:
        plot_interaction_summary(axes[1], cross_df)
    else:
        axes[1].text(0.5, 0.5, "No cross-edge summary", ha="center", va="center")
        axes[1].set_axis_off()
    fig.savefig(summary_dir / "interpretability_summary.png", bbox_inches="tight")
    plt.close(fig)


def main():
    args = parse_args()
    apply_pgssi_model_flags(args)
    ensure_dir(args.output_dir)
    ensure_dir(args.cache_dir)

    df = pd.read_csv(args.data_path)
    pred_path = Path(args.predictions_path)
    predictions_df = pd.read_csv(pred_path) if pred_path.exists() else None
    selected = choose_samples(df, predictions_df, args.selection, args.num_samples, args.exclude_small_solvents)

    model, device = load_model(args)
    data_list = build_selected_data(df, selected.index.tolist(), args.cache_dir)
    data_by_sample = {}
    for data in data_list:
        sample_idx = int(data.sample_index.view(-1)[0].item())
        data_by_sample[sample_idx] = data

    sample_summaries = []
    atom_frames = []
    edge_frames = []
    graph_frames = []
    feature_frames = []
    atom_score_pairs = []
    gallery_entries = []

    for row_idx, row in selected.iterrows():
        if row_idx not in data_by_sample:
            continue
        summary, atom_df, edge_df, graph_df, feature_df, atom_scores_s, atom_scores_v, edge_stats, solvent_fg, solute_fg = analyze_single_sample(
            model=model,
            device=device,
            row=row,
            data=data_by_sample[row_idx],
            output_dir=args.output_dir,
        )
        atom_df.insert(0, "sample_index", row_idx)
        edge_df.insert(0, "sample_index", row_idx)
        graph_df.insert(0, "sample_index", row_idx)
        sample_summaries.append(summary)
        atom_frames.append(atom_df)
        edge_frames.append(edge_df)
        graph_frames.append(graph_df)
        feature_frames.append(feature_df)
        atom_score_pairs.append((atom_scores_s, atom_scores_v))
        gallery_entries.append(
            {
                "atom_scores_s": atom_scores_s,
                "atom_scores_v": atom_scores_v,
                "edge_stats": edge_stats,
                "solvent_fg": solvent_fg,
                "solute_fg": solute_fg,
                "coords_s": data_by_sample[row_idx].pos_s.detach().cpu().numpy(),
                "coords_v": data_by_sample[row_idx].pos_v.detach().cpu().numpy(),
            }
        )

    all_atom_df = pd.concat(atom_frames, ignore_index=True) if atom_frames else pd.DataFrame()
    all_edge_df = pd.concat(edge_frames, ignore_index=True) if edge_frames else pd.DataFrame()
    all_graph_df = pd.concat(graph_frames, ignore_index=True) if graph_frames else pd.DataFrame()
    save_global_summary(args.output_dir, sample_summaries, all_atom_df, all_edge_df, all_graph_df)
    plot_feature_heatmaps(args.output_dir, sample_summaries, feature_frames)
    plot_representative_panel_nature(args.output_dir, sample_summaries, selected, atom_score_pairs)
    plot_3d_gallery(args.output_dir, sample_summaries, selected, gallery_entries, nrows=args.grid_rows, ncols=args.grid_cols)
    plot_3d_gallery_joint(args.output_dir, sample_summaries, selected, gallery_entries, nrows=args.grid_rows, ncols=args.grid_cols)

    run_meta = {
        "model_path": args.model_path,
        "data_path": args.data_path,
        "predictions_path": args.predictions_path,
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
    }
    meta_path = Path(args.output_dir) / "interpretability_run_config.json"
    with open(meta_path, "w", encoding="utf-8") as fh:
        json.dump(run_meta, fh, indent=2, ensure_ascii=False)

    print(f"Interpretability analysis completed. Outputs: {Path(args.output_dir).resolve()}")


if __name__ == "__main__":
    main()
