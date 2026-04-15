import os
from pathlib import Path

import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, rdmolops
from tqdm import tqdm

try:
    from PGSSI_3D_architecture import PairData
except ImportError:
    from src.models.PGSSI.PGSSI_3D_architecture import PairData


GEOM_CACHE_VERSION = "v11_jointpair_typedcross"
# 复用 RDKit 全部描述符，并在后续做 log1p 压缩。
RDKIT_DESC_CALC = [x[1] for x in Descriptors._descList]


def ensure_mol_ready(mol):
    """Initialize caches RDKit expects before ring- or descriptor-based queries."""
    if mol is None:
        return None
    try:
        mol.UpdatePropertyCache(strict=False)
    except Exception:
        pass
    try:
        Chem.GetSymmSSSR(mol)
    except Exception:
        try:
            rdmolops.FastFindRings(mol)
        except Exception:
            pass
    return mol


def get_38_atom_features(atom):
    """Build the fixed-size atom feature vector used by PGSSI."""
    feats = []
    symbols = ["C", "N", "O", "Cl", "S", "F", "Br", "I", "P", "H", "Si", "Sn", "Pb", "Ge", "Hg", "Te"]
    feats += [1 if atom.GetSymbol() == symbol else 0 for symbol in symbols]
    try:
        is_in_ring = atom.IsInRing()
    except Exception:
        is_in_ring = False
    try:
        is_aromatic = atom.GetIsAromatic()
    except Exception:
        is_aromatic = False
    feats += [
        is_in_ring,
        is_aromatic,
        atom.GetTotalDegree(),
        atom.GetFormalCharge(),
        atom.GetTotalNumHs(),
    ]
    hybridization = atom.GetHybridization()
    feats += [
        hybridization == Chem.rdchem.HybridizationType.SP,
        hybridization == Chem.rdchem.HybridizationType.SP2,
        hybridization == Chem.rdchem.HybridizationType.SP3,
    ]
    feats.append(1 if atom.GetChiralTag() != Chem.rdchem.ChiralType.CHI_UNSPECIFIED else 0)
    while len(feats) < 38:
        feats.append(0)
    return feats


def _safe_gasteiger_charge(atom):
    """安全读取 Gasteiger 电荷；遇到异常或 NaN 时退化为 0。"""
    try:
        charge = float(atom.GetProp("_GasteigerCharge")) if atom.HasProp("_GasteigerCharge") else 0.0
    except Exception:
        charge = 0.0
    if np.isnan(charge) or np.isinf(charge):
        return 0.0
    return charge


def _is_hbond_donor(atom):
    atomic_num = atom.GetAtomicNum()
    if atomic_num not in (7, 8, 15, 16):
        return 0.0
    return 1.0 if atom.GetTotalNumHs(includeNeighbors=True) > 0 else 0.0


def _is_hbond_acceptor(atom):
    atomic_num = atom.GetAtomicNum()
    if atomic_num not in (7, 8, 15, 16):
        return 0.0
    if atom.GetFormalCharge() > 0:
        return 0.0
    if atom.GetIsAromatic() and atomic_num == 7:
        return 0.0
    return 1.0


def _atom_role_features(atom):
    atomic_num = atom.GetAtomicNum()
    donor = _is_hbond_donor(atom)
    acceptor = _is_hbond_acceptor(atom)
    aromatic = 1.0 if atom.GetIsAromatic() else 0.0
    hydrophobic = 1.0 if atomic_num in (6, 9, 17, 35, 53) and atom.GetFormalCharge() == 0 else 0.0
    polar = 1.0 if atomic_num in (7, 8, 15, 16) or abs(atom.GetFormalCharge()) > 0 else 0.0
    return [donor, acceptor, aromatic, hydrophobic, polar]


def _safe_unit_vector(vec):
    norm = np.linalg.norm(vec)
    if norm < 1e-8 or not np.isfinite(norm):
        return np.zeros(3, dtype=np.float32)
    return (vec / norm).astype(np.float32)


def _molecular_dipole_vectors(charges, positions):
    dipole = np.sum(np.asarray(charges, dtype=np.float32) * np.asarray(positions, dtype=np.float32), axis=0)
    unit = _safe_unit_vector(dipole)
    return np.repeat(unit[None, :], repeats=len(charges), axis=0)


def _aromatic_normals(mol, positions):
    normals = np.zeros((mol.GetNumAtoms(), 3), dtype=np.float32)
    for atom in mol.GetAtoms():
        idx = atom.GetIdx()
        if not atom.GetIsAromatic():
            continue
        aromatic_neighbors = [nbr.GetIdx() for nbr in atom.GetNeighbors() if nbr.GetIsAromatic()]
        if len(aromatic_neighbors) < 2:
            continue
        vec1 = positions[aromatic_neighbors[0]] - positions[idx]
        vec2 = positions[aromatic_neighbors[1]] - positions[idx]
        normal = np.cross(vec1, vec2)
        normals[idx] = _safe_unit_vector(normal)
    return normals


def _optimize_conformer(mol, conf_id=-1):
    """优先 MMFF，再退化到 UFF，对嵌入后的 3D 构象做局部优化。"""
    try:
        if AllChem.MMFFHasAllMoleculeParams(mol):
            AllChem.MMFFOptimizeMolecule(mol, mmffVariant="MMFF94s", confId=conf_id, maxIters=200)
            return
    except Exception:
        pass
    try:
        if AllChem.UFFHasAllMoleculeParams(mol):
            AllChem.UFFOptimizeMolecule(mol, confId=conf_id, maxIters=200)
    except Exception:
        pass


def _bond_angle(center_idx, anchor_idx, other_idx, positions):
    """计算三点夹角，返回弧度。"""
    v1 = positions[anchor_idx] - positions[center_idx]
    v2 = positions[other_idx] - positions[center_idx]
    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)
    if n1 < 1e-8 or n2 < 1e-8:
        return None
    cos_theta = np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0)
    return float(np.arccos(cos_theta))


def _dihedral_angle(a_idx, b_idx, c_idx, d_idx, positions):
    """计算四点二面角，返回弧度绝对值。"""
    p0, p1, p2, p3 = positions[a_idx], positions[b_idx], positions[c_idx], positions[d_idx]
    b0 = p1 - p0
    b1 = p2 - p1
    b2 = p3 - p2
    b1_norm = np.linalg.norm(b1)
    if b1_norm < 1e-8:
        return None

    b1_unit = b1 / b1_norm
    v = b0 - np.dot(b0, b1_unit) * b1_unit
    w = b2 - np.dot(b2, b1_unit) * b1_unit
    v_norm = np.linalg.norm(v)
    w_norm = np.linalg.norm(w)
    if v_norm < 1e-8 or w_norm < 1e-8:
        return None

    x = np.dot(v, w)
    y = np.dot(np.cross(b1_unit, v), w)
    return float(abs(np.arctan2(y, x)))


def _directed_edge_geometry(src_idx, dst_idx, positions, neighbor_map):
    """为有向键边提取局部几何信息。

    返回：
    - 平均键角
    - 平均二面角
    - 是否存在可用键角
    - 是否存在可用二面角
    """
    angle_values = []
    for nbr_idx in neighbor_map[src_idx]:
        if nbr_idx == dst_idx:
            continue
        angle = _bond_angle(src_idx, dst_idx, nbr_idx, positions)
        if angle is not None and np.isfinite(angle):
            angle_values.append(angle)
    for nbr_idx in neighbor_map[dst_idx]:
        if nbr_idx == src_idx:
            continue
        angle = _bond_angle(dst_idx, src_idx, nbr_idx, positions)
        if angle is not None and np.isfinite(angle):
            angle_values.append(angle)

    dihedral_values = []
    left_neighbors = [nbr for nbr in neighbor_map[src_idx] if nbr != dst_idx]
    right_neighbors = [nbr for nbr in neighbor_map[dst_idx] if nbr != src_idx]
    for left_idx in left_neighbors:
        for right_idx in right_neighbors:
            dihedral = _dihedral_angle(left_idx, src_idx, dst_idx, right_idx, positions)
            if dihedral is not None and np.isfinite(dihedral):
                dihedral_values.append(dihedral)

    angle_value = float(np.mean(angle_values)) if angle_values else 0.0
    dihedral_value = float(np.mean(dihedral_values)) if dihedral_values else 0.0
    return [
        angle_value,
        dihedral_value,
        1.0 if angle_values else 0.0,
        1.0 if dihedral_values else 0.0,
    ]


def _embed_joint_pair(solvent_smiles, solute_smiles):
    """把溶剂和溶质作为一个联合体系嵌入到同一个 3D 空间中。"""
    solvent = Chem.MolFromSmiles(solvent_smiles)
    solute = Chem.MolFromSmiles(solute_smiles)
    if solvent is None or solute is None:
        return None

    solvent = ensure_mol_ready(Chem.AddHs(solvent))
    solute = ensure_mol_ready(Chem.AddHs(solute))
    combined = Chem.CombineMols(solvent, solute)
    combined = Chem.Mol(combined)
    combined = ensure_mol_ready(combined)

    params = AllChem.ETKDGv3()
    params.randomSeed = 42
    params.useSmallRingTorsions = True
    if AllChem.EmbedMolecule(combined, params) < 0:
        fallback_params = AllChem.ETKDGv2()
        fallback_params.randomSeed = 42
        if AllChem.EmbedMolecule(combined, fallback_params) < 0:
            return None

    _optimize_conformer(combined)
    return solvent, solute, combined


def _build_fragment_graph(mol, positions):
    """把单个分子片段转换成图张量。

    节点包含原子特征、坐标和电荷；
    边包含有向键以及局部几何特征。
    """
    mol = ensure_mol_ready(Chem.Mol(mol))
    AllChem.ComputeGasteigerCharges(mol)
    x, charges, atom_roles, edge, edge_attr = [], [], [], [], []
    for atom in mol.GetAtoms():
        x.append(get_38_atom_features(atom))
        charges.append([_safe_gasteiger_charge(atom)])
        atom_roles.append(_atom_role_features(atom))

    charge_array = np.asarray(charges, dtype=np.float32)
    dipole_vectors = _molecular_dipole_vectors(charge_array, positions)
    aromatic_normals = _aromatic_normals(mol, positions)

    neighbor_map = {atom.GetIdx(): [nbr.GetIdx() for nbr in atom.GetNeighbors()] for atom in mol.GetAtoms()}
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        geom_ij = _directed_edge_geometry(i, j, positions, neighbor_map)
        geom_ji = _directed_edge_geometry(j, i, positions, neighbor_map)
        edge += [[i, j], [j, i]]
        edge_attr += [geom_ij, geom_ji]

    return (
        torch.tensor(x, dtype=torch.float),
        torch.tensor(positions, dtype=torch.float),
        torch.tensor(charges, dtype=torch.float),
        torch.tensor(edge, dtype=torch.long).t().contiguous() if edge else torch.empty((2, 0), dtype=torch.long),
        torch.tensor(edge_attr, dtype=torch.float) if edge_attr else torch.empty((0, 4), dtype=torch.float),
        torch.tensor(atom_roles, dtype=torch.float),
        torch.tensor(aromatic_normals, dtype=torch.float),
        torch.tensor(dipole_vectors, dtype=torch.float),
    )


def _build_cross_edge_index(pos_s, pos_v, cutoff):
    """基于 3D 距离为 solvent/solute 建立跨分子边。"""
    if pos_s.size == 0 or pos_v.size == 0:
        return torch.empty((2, 0), dtype=torch.long)

    diff = pos_s[:, None, :] - pos_v[None, :, :]
    dist = np.linalg.norm(diff, axis=-1)
    src_idx, dst_idx = np.where(dist <= cutoff)

    # 如果 cutoff 内没有跨分子边，至少保留最近的一对原子，避免图断开。
    if src_idx.size == 0:
        flat_idx = int(np.argmin(dist))
        src_idx, dst_idx = np.unravel_index(flat_idx, dist.shape)
        src_idx = np.asarray([src_idx], dtype=np.int64)
        dst_idx = np.asarray([dst_idx], dtype=np.int64)

    return torch.tensor(np.stack([src_idx, dst_idx], axis=0), dtype=torch.long)


def process_pair_3d(solvent_smiles, solute_smiles, interaction_cutoff=4.5):
    """对一个溶剂-溶质对生成完整 3D 图表示。"""
    embedded = _embed_joint_pair(solvent_smiles, solute_smiles)
    if embedded is None:
        return None

    solvent, solute, combined = embedded
    conf = combined.GetConformer()
    positions = np.asarray(conf.GetPositions(), dtype=np.float32)
    positions = positions - positions.mean(axis=0, keepdims=True)

    num_solvent_atoms = solvent.GetNumAtoms()
    solvent_pos = positions[:num_solvent_atoms]
    solute_pos = positions[num_solvent_atoms:]

    solvent_graph = _build_fragment_graph(solvent, solvent_pos)
    solute_graph = _build_fragment_graph(solute, solute_pos)
    cross_edge_index = _build_cross_edge_index(solvent_pos, solute_pos, interaction_cutoff)

    return {
        "solvent_graph": solvent_graph,
        "solute_graph": solute_graph,
        "cross_edge_index": cross_edge_index,
    }


def get_rdkit_features(smiles):
    """计算整分子 RDKit 描述符，并做数值稳定的 log 压缩。"""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return [0.0] * len(RDKIT_DESC_CALC)
    mol = ensure_mol_ready(mol)

    feats = []
    for calc in RDKIT_DESC_CALC:
        try:
            value = float(calc(mol))
            if np.isnan(value) or np.isinf(value):
                feats.append(0.0)
            else:
                sign = 1.0 if value >= 0 else -1.0
                feats.append(sign * np.log1p(abs(value)))
        except Exception:
            feats.append(0.0)
    return feats


def _resolve_cache_path(cache_path):
    """为缓存文件名补上版本号，避免旧缓存和新结构冲突。"""
    cache_path_obj = Path(cache_path)
    if cache_path_obj.suffix == ".pt" and not cache_path_obj.stem.endswith(GEOM_CACHE_VERSION):
        return cache_path_obj.with_name(f"{cache_path_obj.stem}_{GEOM_CACHE_VERSION}.pt")
    return cache_path_obj


def _safe_cache_load(path):
    """加载缓存；失败时返回 None 让上层重建。"""
    try:
        return torch.load(str(path), weights_only=False)
    except Exception as exc:
        print(f"cache reload failed for {path}: {exc}; rebuilding")
        return None


def build_pair_dataset(df, cache_path, include_k_targets=False, cache_desc="build joint 3d pair dataset"):
    """把表格数据构造成 PGSSI 使用的 PairData 列表。

    每一条样本会同时包含：
    - solvent 图
    - solute 图
    - 跨分子边
    - 温度
    - log-gamma 标签
    - 分子级 RDKit 描述符
    """
    resolved_cache_path = _resolve_cache_path(cache_path)
    if resolved_cache_path.exists():
        cached = _safe_cache_load(resolved_cache_path)
        if cached is not None:
            return cached
        try:
            resolved_cache_path.unlink()
        except OSError:
            pass

    data_list = []
    num_rdkit = len(RDKIT_DESC_CALC)
    for row_idx, row in tqdm(df.iterrows(), total=len(df), desc=cache_desc, dynamic_ncols=True):
        pair_graph = process_pair_3d(row["Solvent_SMILES"], row["Solute_SMILES"])
        if pair_graph is None:
            continue

        solvent_graph = pair_graph["solvent_graph"]
        solute_graph = pair_graph["solute_graph"]
        data = PairData(
            x_s=solvent_graph[0],
            pos_s=solvent_graph[1],
            charge_s=solvent_graph[2],
            edge_index_s=solvent_graph[3],
            edge_attr_s=solvent_graph[4],
            atom_role_s=solvent_graph[5],
            aromatic_normal_s=solvent_graph[6],
            dipole_vec_s=solvent_graph[7][0:1],
            x_v=solute_graph[0],
            pos_v=solute_graph[1],
            charge_v=solute_graph[2],
            edge_index_v=solute_graph[3],
            edge_attr_v=solute_graph[4],
            atom_role_v=solute_graph[5],
            aromatic_normal_v=solute_graph[6],
            dipole_vec_v=solute_graph[7][0:1],
            cross_edge_index=pair_graph["cross_edge_index"],
            temp=torch.tensor([row["T"]], dtype=torch.float),
        )
        # 训练入口当前只依赖 log-gamma 标签；K1/K2 仅作为可选扩展保留。
        data.y = torch.tensor([float(row["log-gamma"])], dtype=torch.float) if "log-gamma" in df.columns else torch.tensor([0.0], dtype=torch.float)
        if include_k_targets:
            data.y_k1 = torch.tensor([float(row["K1"])], dtype=torch.float)
            data.y_k2 = torch.tensor([float(row["K2"])], dtype=torch.float)

        data.num_nodes_s = solvent_graph[0].size(0)
        data.num_nodes_v = solute_graph[0].size(0)

        rdkit_s = get_rdkit_features(row["Solvent_SMILES"])
        rdkit_v = get_rdkit_features(row["Solute_SMILES"])
        if len(rdkit_s) != num_rdkit or len(rdkit_v) != num_rdkit:
            rdkit_s = [0.0] * num_rdkit
            rdkit_v = [0.0] * num_rdkit
        data.rdkit_s = torch.tensor(rdkit_s, dtype=torch.float).view(1, -1)
        data.rdkit_v = torch.tensor(rdkit_v, dtype=torch.float).view(1, -1)
        data.sample_index = torch.tensor([int(row_idx)], dtype=torch.long)
        data_list.append(data)

    torch.save(data_list, str(resolved_cache_path))
    return data_list


def clear_cache_file(path):
    """显式删除单个缓存文件。"""
    if os.path.exists(path):
        try:
            os.remove(path)
        except OSError:
            pass
