import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from rdkit.Chem import Descriptors
from torch_geometric.data import Data
from torch_geometric.nn import MessagePassing, global_max_pool, global_mean_pool, radius
from torch_scatter import scatter_add


def scatter_mean_safe(src, index, dim_size):
    """scatter-mean without introducing an extra dependency."""
    if src.numel() == 0:
        return src.new_zeros((dim_size, src.size(-1)))
    out = scatter_add(src, index, dim=0, dim_size=dim_size)
    count = scatter_add(
        torch.ones((src.size(0), 1), device=src.device, dtype=src.dtype),
        index,
        dim=0,
        dim_size=dim_size,
    ).clamp_min(1.0)
    return out / count


class PairData(Data):
    """同时保存 solvent/solute 两张分子图的 PyG 数据对象。"""

    def __init__(
        self,
        x_s=None,
        pos_s=None,
        charge_s=None,
        edge_index_s=None,
        edge_attr_s=None,
        atom_role_s=None,
        aromatic_normal_s=None,
        dipole_vec_s=None,
        x_v=None,
        pos_v=None,
        charge_v=None,
        edge_index_v=None,
        edge_attr_v=None,
        atom_role_v=None,
        aromatic_normal_v=None,
        dipole_vec_v=None,
        cross_edge_index=None,
        temp=None,
    ):
        super().__init__()
        # s 表示 solvent，v 表示 solute；两套图会在同一个样本对象里同时保存。
        self.x_s, self.pos_s, self.charge_s, self.edge_index_s, self.edge_attr_s = (
            x_s,
            pos_s,
            charge_s,
            edge_index_s,
            edge_attr_s,
        )
        self.atom_role_s = atom_role_s
        self.aromatic_normal_s = aromatic_normal_s
        self.dipole_vec_s = dipole_vec_s
        self.x_v, self.pos_v, self.charge_v, self.edge_index_v, self.edge_attr_v = (
            x_v,
            pos_v,
            charge_v,
            edge_index_v,
            edge_attr_v,
        )
        self.atom_role_v = atom_role_v
        self.aromatic_normal_v = aromatic_normal_v
        self.dipole_vec_v = dipole_vec_v
        self.cross_edge_index = cross_edge_index
        self.temp = temp

        if x_s is not None and x_v is not None:
            self.num_nodes = x_s.size(0) + x_v.size(0)

    def __inc__(self, key, value, *args, **kwargs):
        if key == "edge_index_s":
            return self.x_s.size(0)
        if key == "edge_index_v":
            return self.x_v.size(0)
        if key == "cross_edge_index":
            return torch.tensor([[self.x_s.size(0)], [self.x_v.size(0)]])
        if key == "sample_index":
            return 0
        return super().__inc__(key, value, *args, **kwargs)

    def __cat_dim__(self, key, value, *args, **kwargs):
        if key in [
            "x_s",
            "x_v",
            "pos_s",
            "pos_v",
            "charge_s",
            "charge_v",
            "edge_attr_s",
            "edge_attr_v",
            "atom_role_s",
            "atom_role_v",
            "aromatic_normal_s",
            "aromatic_normal_v",
            "dipole_vec_s",
            "dipole_vec_v",
        ]:
            return 0
        return super().__cat_dim__(key, value, *args, **kwargs)


class SphericalBesselBasis(nn.Module):
    """将距离编码为一组径向基函数。"""

    def __init__(self, num_radial=16, cutoff=5.0):
        super().__init__()
        self.cutoff = cutoff
        self.register_buffer(
            "freq",
            torch.arange(1, num_radial + 1).float() * math.pi / cutoff,
            persistent=False,
        )

    def forward(self, dist):
        dist = dist.unsqueeze(-1) if dist.dim() == 1 else dist
        clipped = torch.clamp(dist, min=1e-6)
        return torch.sin(self.freq * clipped) / clipped


class EquivariantEGNNLayer(nn.Module):
    """EGNN-style equivariant message passing with scalar-node updates and coordinate refinement."""

    def __init__(self, hidden_dim, num_radial=16, cutoff=5.0, dropout=0.10):
        super().__init__()
        self.cutoff = cutoff
        self.radial = SphericalBesselBasis(num_radial=num_radial, cutoff=cutoff)
        self.legendre = LegendreBasis(degree=4)
        edge_dim = (hidden_dim * 2) + num_radial + 3
        self.edge_mlp = nn.Sequential(
            nn.Linear(edge_dim, hidden_dim),
            nn.SiLU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )
        self.node_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.LayerNorm(hidden_dim),
        )
        self.coord_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def _edge_attr_gate(self, edge_attr, dist):
        if edge_attr is None or edge_attr.numel() == 0:
            return torch.ones_like(dist)
        angle_in = edge_attr[:, 0:1]
        dihedral_in = edge_attr[:, 1:2] if edge_attr.size(-1) > 1 else torch.zeros_like(angle_in)
        angle_feat = self.legendre(angle_in)
        dihedral_feat = self.legendre(dihedral_in)
        angle_score = angle_feat.abs().mean(dim=-1, keepdim=True)
        dihedral_score = dihedral_feat.abs().mean(dim=-1, keepdim=True)
        if edge_attr.size(-1) > 2:
            angle_score = angle_score * edge_attr[:, 2:3]
        if edge_attr.size(-1) > 3:
            dihedral_score = dihedral_score * edge_attr[:, 3:4]
        gate = 1.0 + 0.6 * (angle_score + dihedral_score)
        return torch.clamp(gate, min=0.35, max=2.5)

    def forward(self, x, pos, charge, edge_index, edge_attr=None):
        if edge_index is None or edge_index.numel() == 0:
            return x, pos

        src, dst = edge_index[0], edge_index[1]
        rel_vec = pos[dst] - pos[src]
        dist = torch.sqrt(torch.sum(rel_vec**2, dim=-1, keepdim=True) + 1e-8)
        safe_dist = torch.clamp(dist, min=1e-6)
        unit_vec = rel_vec / safe_dist
        edge_gate = self._edge_attr_gate(edge_attr, dist)
        edge_feat = torch.cat(
            [
                x[src],
                x[dst],
                self.radial(dist) * edge_gate,
                (dist / self.cutoff) * edge_gate,
                (charge[src] * charge[dst]) * edge_gate,
                torch.abs(charge[src] - charge[dst]) * edge_gate,
            ],
            dim=-1,
        )
        edge_hidden = self.edge_mlp(edge_feat)

        agg_msg = scatter_mean_safe(edge_hidden, src, x.size(0))
        delta = scatter_mean_safe(unit_vec * self.coord_head(edge_hidden) * edge_gate, src, pos.size(0))

        x = x + self.node_mlp(torch.cat([x, agg_msg], dim=-1))
        pos = pos + 0.1 * delta
        return x, pos


class LegendreBasis(nn.Module):
    """将角度相关标量展开为勒让德多项式基。"""

    def __init__(self, degree=4):
        super().__init__()
        self.degree = degree

    def forward(self, angle):
        cos_a = torch.cos(angle)
        polys = [torch.ones_like(cos_a), cos_a]
        for l in range(1, self.degree - 1):
            polys.append(((2 * l + 1) * cos_a * polys[-1] - l * polys[-2]) / (l + 1))
        return torch.cat(polys, dim=-1)


class SphericalHarmonicsEncoding(nn.Module):
    """保留一个低阶方向编码器，供需要时扩展方向相关特征。"""

    def forward(self, unit_vec):
        x, y, z = unit_vec[:, 0:1], unit_vec[:, 1:2], unit_vec[:, 2:3]
        y_1 = unit_vec
        y_2 = torch.cat([x * y, x * z, y * z, x**2 - y**2, 3 * z**2 - 1], dim=-1)
        return torch.cat([y_1, y_2], dim=-1)


class PhysicsGatedMPNN(MessagePassing):
    """
    分子内 3D 消息传递层。

    标量消息仅依赖旋转不变的几何量和物理量，坐标更新通过“标量系数 × 单位方向向量”
    的形式施加，从而在平移/旋转下保持等变。
    """

    def __init__(self, hidden_dim, num_radial=16, cutoff=5.0, angle_degree=4, update_coords=False):
        super().__init__(aggr="add")
        # 这一层负责“分子内”消息传递，把几何量和物理量都编码进边消息。
        self.hidden_dim = hidden_dim
        self.cutoff = cutoff
        self.update_coords = update_coords
        self.bessel = SphericalBesselBasis(num_radial=num_radial, cutoff=cutoff)
        self.legendre = LegendreBasis(degree=angle_degree)
        self.phys_param_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 2),
        )

        scalar_dim = (hidden_dim * 2) + num_radial + angle_degree + angle_degree + 5
        self.msg_mlp = nn.Sequential(
            nn.Linear(scalar_dim, hidden_dim),
            nn.SiLU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(0.15),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.phys_gate = nn.Sequential(nn.Linear(2, hidden_dim), nn.Sigmoid())
        self.coord_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, 1),
        )
        self.update_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.SiLU(),
            nn.Dropout(0.15),
            nn.LayerNorm(hidden_dim),
        )

    def _angle_features(self, edge_attr, dist):
        if edge_attr is None or edge_attr.numel() == 0:
            zeros = dist.new_zeros((dist.size(0), self.legendre.degree))
            return zeros, zeros

        angle_in = edge_attr[:, 0:1]
        if edge_attr.size(-1) > 1:
            dihedral_in = edge_attr[:, 1:2]
        else:
            dihedral_in = torch.zeros_like(angle_in)
        angle_feat = self.legendre(angle_in)
        dihedral_feat = self.legendre(dihedral_in)

        # Optional masks let the data builder distinguish "missing local geometry"
        # from a genuine zero-valued angle/dihedral.
        if edge_attr.size(-1) > 2:
            angle_feat = angle_feat * edge_attr[:, 2:3]
        if edge_attr.size(-1) > 3:
            dihedral_feat = dihedral_feat * edge_attr[:, 3:4]
        return angle_feat, dihedral_feat

    def forward(self, x, pos, charge, edge_index, edge_attr):
        # 先根据节点表征预测可学习的 LJ 参数，再参与后续消息构造。
        lj_params = F.softplus(self.phys_param_proj(x)) + 1e-3
        return self.propagate(
            edge_index,
            x=x,
            pos=pos,
            charge=charge,
            eps=lj_params[:, 0:1],
            sig=lj_params[:, 1:2],
            edge_attr=edge_attr,
        )

    def message(self, x_i, x_j, pos_i, pos_j, charge_i, charge_j, eps_i, eps_j, sig_i, sig_j, edge_attr):
        # 用相对位移构造距离、方向和物理相互作用项。
        rel_vec = pos_j - pos_i
        dist = torch.sqrt(torch.sum(rel_vec**2, dim=-1, keepdim=True) + 1e-8)
        safe_dist = torch.clamp(dist, min=0.35)
        unit_vec = rel_vec / safe_dist

        eps_ij = torch.sqrt(eps_i * eps_j + 1e-8)
        sig_ij = 0.5 * (sig_i + sig_j)
        ratio = torch.clamp(sig_ij / safe_dist, max=1.5)
        e_lj = 4.0 * eps_ij * (ratio**12 - ratio**6)
        e_coul = (charge_i * charge_j) / safe_dist

        dist_feat = self.bessel(dist)
        angle_feat, dihedral_feat = self._angle_features(edge_attr, dist)
        scalar_feat = torch.cat(
            [
                x_i,
                x_j,
                dist_feat,
                angle_feat,
                dihedral_feat,
                dist / self.cutoff,
                charge_i * charge_j,
                torch.abs(charge_i - charge_j),
                e_lj,
                e_coul,
            ],
            dim=-1,
        )

        # 先得到原始消息，再用物理门控抑制不合理边消息。
        raw_msg = self.msg_mlp(scalar_feat)
        gated_msg = raw_msg * (self.phys_gate(torch.cat([e_lj, e_coul], dim=-1)) + 0.05)
        coord_scale = 0.1 * torch.tanh(self.coord_head(gated_msg))
        coord_update = unit_vec * coord_scale
        return gated_msg, coord_update

    def aggregate(self, inputs, index, ptr=None, dim_size=None):
        msg, coord = inputs
        return (
            super().aggregate(msg, index, ptr, dim_size),
            super().aggregate(coord, index, ptr, dim_size),
        )

    def update(self, aggr_out, x, pos):
        agg_msg, agg_coord = aggr_out
        updated_x = x + self.update_mlp(torch.cat([x, agg_msg], dim=-1))
        updated_pos = pos + agg_coord if self.update_coords else pos
        return updated_x, updated_pos


class Explicit3DInteraction(nn.Module):
    """
    显式的溶质-溶剂三维微观交互层。

    在联合三维空间中建立跨分子半径图，并在跨分子边上同时施加几何筛选和物理势门控，
    且双向更新 solvent / solute 两侧表示。
    """

    def __init__(self, hidden_dim, cutoff=4.5, num_radial=32, update_coords=False):
        super().__init__()
        # 这一层负责“跨分子”相互作用，也就是 solvent 和 solute 之间的显式耦合。
        self.cutoff = cutoff
        self.update_coords = update_coords
        self.radial_expansion = SphericalBesselBasis(num_radial=num_radial, cutoff=cutoff)
        self.cross_phys_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 2),
        )
        edge_dim = (hidden_dim * 2) + num_radial + 6
        self.edge_norm = nn.LayerNorm(edge_dim)
        self.edge_mlp = nn.Sequential(
            nn.Linear(edge_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(0.15),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )
        self.phys_gate = nn.Sequential(nn.Linear(2, hidden_dim), nn.Sigmoid())
        self.msg_from_s = nn.Linear(hidden_dim, hidden_dim)
        self.msg_from_v = nn.Linear(hidden_dim, hidden_dim)
        self.update_s = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.SiLU(),
            nn.Dropout(0.15),
            nn.LayerNorm(hidden_dim),
        )
        self.update_v = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.SiLU(),
            nn.Dropout(0.15),
            nn.LayerNorm(hidden_dim),
        )
        self.coord_head_s = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, 1),
        )
        self.coord_head_v = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, h_s, p_s, bs, h_v, p_v, bv, q_s, q_v, cross_edge_index=None):
        # 如果数据侧已经构好了跨分子边，就直接复用；否则根据坐标动态构图。
        if cross_edge_index is not None and cross_edge_index.numel() > 0:
            idx_s = cross_edge_index[0]
            idx_v = cross_edge_index[1]
        else:
            edge_index_inter = radius(p_v, p_s, self.cutoff, bv, bs)
            if edge_index_inter.numel() == 0:
                return h_s, p_s, h_v, p_v

            raw_idx0 = edge_index_inter[0]
            raw_idx1 = edge_index_inter[1]
            if raw_idx0.max().item() < p_s.size(0) and raw_idx1.max().item() < p_v.size(0):
                idx_s, idx_v = raw_idx0, raw_idx1
            else:
                idx_v, idx_s = raw_idx0, raw_idx1

        rel_vec = p_s[idx_s] - p_v[idx_v]
        dist = torch.sqrt(torch.sum(rel_vec**2, dim=-1, keepdim=True) + 1e-8)
        safe_dist = torch.clamp(dist, min=0.5)
        unit_vec = rel_vec / safe_dist

        lj_s = F.softplus(self.cross_phys_proj(h_s)) + 1e-3
        lj_v = F.softplus(self.cross_phys_proj(h_v)) + 1e-3
        eps_ij = torch.sqrt(lj_s[idx_s, 0:1] * lj_v[idx_v, 0:1] + 1e-8)
        sig_ij = 0.5 * (lj_s[idx_s, 1:2] + lj_v[idx_v, 1:2])
        ratio = torch.clamp(sig_ij / safe_dist, max=1.5)
        e_lj = 4.0 * eps_ij * (ratio**12 - ratio**6)
        e_coul = (q_s[idx_s] * q_v[idx_v]) / safe_dist

        edge_scalar = torch.cat(
            [
                h_s[idx_s],
                h_v[idx_v],
                self.radial_expansion(dist),
                dist / self.cutoff,
                1.0 / safe_dist,
                q_s[idx_s] * q_v[idx_v],
                torch.abs(q_s[idx_s] - q_v[idx_v]),
                e_lj,
                e_coul,
            ],
            dim=-1,
        )
        # edge_hidden 是跨分子边的交互表征，后面同时用于消息和坐标微调。
        edge_hidden = self.edge_mlp(self.edge_norm(edge_scalar))
        gate = torch.sigmoid(edge_hidden) * (self.phys_gate(torch.cat([e_lj, e_coul], dim=-1)) + 0.05)

        msg_to_s = self.msg_from_v(h_v[idx_v]) * gate
        msg_to_v = self.msg_from_s(h_s[idx_s]) * gate

        agg_s = scatter_mean_safe(msg_to_s, idx_s, h_s.size(0))
        agg_v = scatter_mean_safe(msg_to_v, idx_v, h_v.size(0))

        delta_s = scatter_mean_safe(
            unit_vec * (0.12 * torch.tanh(self.coord_head_s(edge_hidden))),
            idx_s,
            p_s.size(0),
        )
        delta_v = scatter_mean_safe(
            -unit_vec * (0.12 * torch.tanh(self.coord_head_v(edge_hidden))),
            idx_v,
            p_v.size(0),
        )

        h_s = h_s + self.update_s(torch.cat([h_s, agg_s], dim=-1))
        h_v = h_v + self.update_v(torch.cat([h_v, agg_v], dim=-1))
        if self.update_coords:
            p_s = p_s + delta_s
            p_v = p_v + delta_v
        return h_s, p_s, h_v, p_v


class TypedCrossInteraction(nn.Module):
    """Cross-molecular interaction layer with interaction-type features and mixture-of-experts gating."""

    def __init__(
        self,
        hidden_dim,
        cutoff=4.5,
        num_radial=24,
        num_experts=4,
        dropout=0.10,
        use_interaction_types=True,
        use_moe=True,
        use_physics_prior=True,
        topology_only=False,
    ):
        super().__init__()
        self.cutoff = cutoff
        self.num_experts = num_experts
        self.topology_only = topology_only
        self.use_physics_prior = use_physics_prior
        self.radial = SphericalBesselBasis(num_radial=num_radial, cutoff=cutoff)
        self.use_interaction_types = use_interaction_types
        self.use_moe = use_moe
        self.cross_phys_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 2),
        )
        self.type_proj = nn.Sequential(
            nn.Linear(8, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, hidden_dim // 2),
        )
        edge_dim = (hidden_dim * 2) + num_radial + 9 + (hidden_dim // 2)
        self.edge_norm = nn.LayerNorm(edge_dim)
        self.experts = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(edge_dim, hidden_dim),
                    nn.SiLU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.SiLU(),
                )
                for _ in range(num_experts)
            ]
        )
        self.gate = nn.Sequential(
            nn.Linear(edge_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, num_experts),
        )
        self.energy_head = nn.Linear(hidden_dim, 1)
        self.charge_head = nn.Linear(hidden_dim, 1)
        self.msg_from_s = nn.Linear(hidden_dim, hidden_dim)
        self.msg_from_v = nn.Linear(hidden_dim, hidden_dim)
        self.update_s = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.LayerNorm(hidden_dim),
        )
        self.update_v = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.LayerNorm(hidden_dim),
        )
        self.coord_head_s = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, 1),
        )
        self.coord_head_v = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, 1),
        )
        self.last_aux = {}

    def _resolve_interaction_feature_mask(self, interaction_feature_mask, device, dtype):
        num_features = 13
        if interaction_feature_mask is None:
            return torch.ones((num_features,), device=device, dtype=dtype)
        if not torch.is_tensor(interaction_feature_mask):
            interaction_feature_mask = torch.tensor(interaction_feature_mask, device=device, dtype=dtype)
        mask = interaction_feature_mask.to(device=device, dtype=dtype).view(-1)
        if mask.numel() < num_features:
            pad = torch.ones((num_features - mask.numel(),), device=device, dtype=dtype)
            mask = torch.cat([mask, pad], dim=0)
        return mask[:num_features]

    def _build_edges(self, p_s, p_v, bs, bv, cross_edge_index):
        if cross_edge_index is not None and cross_edge_index.numel() > 0:
            return cross_edge_index[0], cross_edge_index[1]
        edge_index_inter = radius(p_v, p_s, self.cutoff, bv, bs)
        if edge_index_inter.numel() == 0:
            return None, None
        raw_idx0, raw_idx1 = edge_index_inter[0], edge_index_inter[1]
        if raw_idx0.max().item() < p_s.size(0) and raw_idx1.max().item() < p_v.size(0):
            return raw_idx0, raw_idx1
        return raw_idx1, raw_idx0

    def _interaction_types(self, role_s, role_v, normal_s, normal_v, dipole_s, dipole_v):
        donor_acceptor = role_s[:, 0:1] * role_v[:, 1:2] + role_s[:, 1:2] * role_v[:, 0:1]
        aromatic_pair = role_s[:, 2:3] * role_v[:, 2:3]
        normal_align = aromatic_pair * torch.abs(torch.sum(normal_s * normal_v, dim=-1, keepdim=True))
        dipole_align = torch.sum(dipole_s * dipole_v, dim=-1, keepdim=True)
        dipole_opposition = 1.0 - torch.abs(dipole_align)
        hydrophobic_pair = role_s[:, 3:4] * role_v[:, 3:4]
        polar_pair = role_s[:, 4:5] * role_v[:, 4:5]
        hydrophobic_polar = role_s[:, 3:4] * role_v[:, 4:5] + role_s[:, 4:5] * role_v[:, 3:4]
        return torch.cat(
            [
                donor_acceptor,
                aromatic_pair,
                normal_align,
                dipole_align,
                dipole_opposition,
                hydrophobic_pair,
                polar_pair,
                hydrophobic_polar,
            ],
            dim=-1,
        )

    def forward(
        self,
        h_s,
        p_s,
        bs,
        h_v,
        p_v,
        bv,
        q_s,
        q_v,
        role_s,
        role_v,
        normal_s,
        normal_v,
        dipole_s,
        dipole_v,
        cross_edge_index=None,
        interaction_feature_mask=None,
    ):
        idx_s, idx_v = self._build_edges(p_s, p_v, bs, bv, cross_edge_index)
        if idx_s is None or idx_v is None:
            return h_s, p_s, h_v, p_v

        feature_mask = self._resolve_interaction_feature_mask(interaction_feature_mask, h_s.device, h_s.dtype)
        rel_vec = p_s[idx_s] - p_v[idx_v]
        if self.topology_only:
            dist = torch.ones((idx_s.size(0), 1), dtype=rel_vec.dtype, device=rel_vec.device)
        else:
            dist = torch.sqrt(torch.sum(rel_vec**2, dim=-1, keepdim=True) + 1e-8)
        safe_dist = torch.clamp(dist, min=0.5)
        unit_vec = rel_vec / safe_dist
        masked_dist = dist * feature_mask[0]
        masked_inv_dist = (1.0 / safe_dist) * feature_mask[0]

        if self.use_physics_prior:
            lj_s = F.softplus(self.cross_phys_proj(h_s)) + 1e-3
            lj_v = F.softplus(self.cross_phys_proj(h_v)) + 1e-3
            eps_ij = torch.sqrt(lj_s[idx_s, 0:1] * lj_v[idx_v, 0:1] + 1e-8)
            sig_ij = 0.5 * (lj_s[idx_s, 1:2] + lj_v[idx_v, 1:2])
            ratio = torch.clamp(sig_ij / safe_dist, max=1.5)
            e_lj = 4.0 * eps_ij * (ratio**12 - ratio**6)
            e_coul = (q_s[idx_s] * q_v[idx_v]) / safe_dist
        else:
            sig_ij = torch.zeros_like(safe_dist)
            e_lj = torch.zeros_like(safe_dist)
            e_coul = torch.zeros_like(safe_dist)
        charge_product = q_s[idx_s] * q_v[idx_v]
        charge_difference = torch.abs(q_s[idx_s] - q_v[idx_v])
        e_lj = e_lj * feature_mask[1]
        e_coul = e_coul * feature_mask[2]
        charge_product = charge_product * feature_mask[3]
        charge_difference = charge_difference * feature_mask[4]

        if self.use_interaction_types:
            interaction_type = self._interaction_types(
                role_s[idx_s],
                role_v[idx_v],
                normal_s[idx_s],
                normal_v[idx_v],
                dipole_s[idx_s],
                dipole_v[idx_v],
            )
            interaction_type = interaction_type * feature_mask[5:13].view(1, -1)
            type_hidden = self.type_proj(interaction_type)
        else:
            interaction_type = torch.zeros((idx_s.size(0), 8), dtype=h_s.dtype, device=h_s.device)
            type_hidden = torch.zeros(
                (idx_s.size(0), self.type_proj[-1].out_features),
                dtype=h_s.dtype,
                device=h_s.device,
            )
        edge_scalar = torch.cat(
            [
                h_s[idx_s],
                h_v[idx_v],
                self.radial(dist) * feature_mask[0],
                masked_dist / self.cutoff,
                masked_inv_dist,
                charge_product,
                charge_difference,
                e_lj,
                e_coul,
                interaction_type[:, 0:1],
                interaction_type[:, 2:3],
                interaction_type[:, 5:6],
                type_hidden,
            ],
            dim=-1,
        )
        edge_scalar = self.edge_norm(edge_scalar)

        if self.use_moe:
            expert_weights = torch.softmax(self.gate(edge_scalar), dim=-1)
            expert_msgs = torch.stack([expert(edge_scalar) for expert in self.experts], dim=1)
            edge_hidden = torch.sum(expert_msgs * expert_weights.unsqueeze(-1), dim=1)
        else:
            edge_hidden = self.experts[0](edge_scalar)
        predicted_energy = self.energy_head(edge_hidden)
        charge_response = self.charge_head(edge_hidden)

        msg_to_s = self.msg_from_v(h_v[idx_v]) * edge_hidden
        msg_to_v = self.msg_from_s(h_s[idx_s]) * edge_hidden
        agg_s = scatter_mean_safe(msg_to_s, idx_s, h_s.size(0))
        agg_v = scatter_mean_safe(msg_to_v, idx_v, h_v.size(0))

        delta_s = scatter_mean_safe(unit_vec * self.coord_head_s(edge_hidden), idx_s, p_s.size(0))
        delta_v = scatter_mean_safe(-unit_vec * self.coord_head_v(edge_hidden), idx_v, p_v.size(0))

        h_s = h_s + self.update_s(torch.cat([h_s, agg_s], dim=-1))
        h_v = h_v + self.update_v(torch.cat([h_v, agg_v], dim=-1))
        if not self.topology_only:
            p_s = p_s + 0.08 * delta_s
            p_v = p_v + 0.08 * delta_v
        self.last_aux = {
            "dist": masked_dist,
            "predicted_energy": predicted_energy,
            "lj_energy": e_lj,
            "coulomb_energy": e_coul,
            "charge_product": charge_product,
            "charge_difference": charge_difference,
            "charge_response": charge_response,
            "sigma": sig_ij,
            "hbond_tendency": interaction_type[:, 0:1],
            "aromatic_pair": interaction_type[:, 1:2],
            "pi_stacking_align": interaction_type[:, 2:3],
            "dipole_align": interaction_type[:, 3:4],
            "dipole_opposition": interaction_type[:, 4:5],
            "hydrophobic_pair": interaction_type[:, 5:6],
            "polar_pair": interaction_type[:, 6:7],
            "hydrophobic_polar": interaction_type[:, 7:8],
            "cutoff": torch.full_like(dist, self.cutoff),
        }
        return h_s, p_s, h_v, p_v


class ExternalAttention(nn.Module):
    """用固定大小的外部记忆槽压缩节点表示。"""

    def __init__(self, hidden_dim, mem_size=128):
        super().__init__()
        self.Mk = nn.Parameter(torch.randn(mem_size, hidden_dim))
        self.Mv = nn.Parameter(torch.randn(mem_size, hidden_dim))

    def forward(self, x):
        attn = torch.matmul(x, self.Mk.T)
        attn = F.softmax(attn, dim=-1)
        attn = attn / (torch.sum(attn, dim=-1, keepdim=True) + 1e-9)
        return torch.matmul(attn, self.Mv)


class Shared_Layer(nn.Module):
    """Shared 3D encoder, organized similarly to GHGNN_MTL's Shared_Layer."""

    def __init__(
        self,
        hidden_dim=256,
        enable_cross_interaction=True,
        num_intra_layers=2,
        use_interaction_types=True,
        use_moe=True,
        use_physics_prior=True,
        disable_cross_refine=False,
        topology_only=False,
    ):
        super().__init__()
        self.enable_cross_interaction = enable_cross_interaction
        self.num_intra_layers = max(0, int(num_intra_layers))
        self.disable_cross_refine = disable_cross_refine
        self.topology_only = topology_only
        self.atom_enc = nn.Sequential(
            nn.Linear(38, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(0.10),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(0.10),
        )
        self.intra_convs = nn.ModuleList(
            [
                EquivariantEGNNLayer(hidden_dim, num_radial=16, cutoff=5.0, dropout=0.10)
                for _ in range(2)
            ]
        )
        self.micro_inter = TypedCrossInteraction(
            hidden_dim,
            cutoff=4.5,
            num_radial=24,
            num_experts=4,
            dropout=0.10,
            use_interaction_types=use_interaction_types,
            use_moe=use_moe,
            use_physics_prior=use_physics_prior,
            topology_only=topology_only,
        )
        self.micro_inter_refine = TypedCrossInteraction(
            hidden_dim,
            cutoff=6.0,
            num_radial=24,
            num_experts=4,
            dropout=0.10,
            use_interaction_types=use_interaction_types,
            use_moe=use_moe,
            use_physics_prior=use_physics_prior,
            topology_only=topology_only,
        )
        self.ext_attn = ExternalAttention(hidden_dim)
        self.ext_attn_residual_scale = 0.2

        num_rdkit = len(Descriptors._descList)
        self.out_dim = (hidden_dim * 6) + (num_rdkit * 2)
        self.readout_norm = nn.LayerNorm(self.out_dim)
        self.readout_dropout = nn.Dropout(0.25)

    def forward(self, data):
        self.last_physics_aux = {}
        # 先清洗数值，避免 3D 构图或缓存中的异常值污染训练。
        data.x_s = torch.nan_to_num(data.x_s, nan=0.0)
        data.x_v = torch.nan_to_num(data.x_v, nan=0.0)
        data.pos_s = torch.clamp(
            torch.nan_to_num(data.pos_s, nan=0.0, posinf=0.0, neginf=0.0),
            -100.0,
            100.0,
        )
        data.pos_v = torch.clamp(
            torch.nan_to_num(data.pos_v, nan=0.0, posinf=0.0, neginf=0.0),
            -100.0,
            100.0,
        )
        data.charge_s = torch.nan_to_num(data.charge_s, nan=0.0)
        data.charge_v = torch.nan_to_num(data.charge_v, nan=0.0)

        # bs / bv 标记每个原子属于 batch 中的哪一个样本图。
        if hasattr(data, "num_nodes_s") and isinstance(data.num_nodes_s, torch.Tensor) and data.num_nodes_s.dim() > 0:
            bs = torch.repeat_interleave(
                torch.arange(data.num_nodes_s.size(0), device=data.x_s.device),
                data.num_nodes_s,
            )
            bv = torch.repeat_interleave(
                torch.arange(data.num_nodes_v.size(0), device=data.x_v.device),
                data.num_nodes_v,
            )
        else:
            bs = torch.zeros(data.x_s.size(0), dtype=torch.long, device=data.x_s.device)
            bv = torch.zeros(data.x_v.size(0), dtype=torch.long, device=data.x_v.device)

        role_s = torch.nan_to_num(
            getattr(data, "atom_role_s", torch.zeros((data.x_s.size(0), 5), dtype=data.x_s.dtype, device=data.x_s.device)),
            nan=0.0,
        )
        role_v = torch.nan_to_num(
            getattr(data, "atom_role_v", torch.zeros((data.x_v.size(0), 5), dtype=data.x_v.dtype, device=data.x_v.device)),
            nan=0.0,
        )
        normal_s = torch.nan_to_num(getattr(data, "aromatic_normal_s", torch.zeros_like(data.pos_s)), nan=0.0)
        normal_v = torch.nan_to_num(getattr(data, "aromatic_normal_v", torch.zeros_like(data.pos_v)), nan=0.0)
        dipole_s = torch.nan_to_num(
            getattr(
                data,
                "dipole_vec_s",
                torch.zeros((int(bs.max().item()) + 1 if bs.numel() else 1, 3), dtype=data.x_s.dtype, device=data.x_s.device),
            ),
            nan=0.0,
        )
        dipole_v = torch.nan_to_num(
            getattr(
                data,
                "dipole_vec_v",
                torch.zeros((int(bv.max().item()) + 1 if bv.numel() else 1, 3), dtype=data.x_v.dtype, device=data.x_v.device),
            ),
            nan=0.0,
        )
        atom_dipole_s = dipole_s[bs]
        atom_dipole_v = dipole_v[bv]

        # 分子内先分别编码两侧图，再做跨分子显式交互。
        h_s = self.atom_enc(data.x_s)
        h_v = self.atom_enc(data.x_v)

        if self.topology_only:
            p_s = torch.zeros_like(data.pos_s)
            p_v = torch.zeros_like(data.pos_v)
        else:
            p_s = data.pos_s
            p_v = data.pos_v
        for layer_idx, intra_conv in enumerate(self.intra_convs):
            if layer_idx >= self.num_intra_layers:
                break
            h_s, p_s = intra_conv(h_s, p_s, data.charge_s, data.edge_index_s, data.edge_attr_s)
            h_v, p_v = intra_conv(h_v, p_v, data.charge_v, data.edge_index_v, data.edge_attr_v)

        if self.enable_cross_interaction:
            cross_edge_index = getattr(data, "cross_edge_index", None)
            interaction_feature_mask = getattr(data, "interaction_feature_mask", None)
            h_s, p_s, h_v, p_v = self.micro_inter(
                h_s,
                p_s,
                bs,
                h_v,
                p_v,
                bv,
                data.charge_s,
                data.charge_v,
                role_s,
                role_v,
                normal_s,
                normal_v,
                atom_dipole_s,
                atom_dipole_v,
                cross_edge_index=cross_edge_index,
                interaction_feature_mask=interaction_feature_mask,
            )
            if not self.disable_cross_refine:
                h_s, p_s, h_v, p_v = self.micro_inter_refine(
                    h_s,
                    p_s,
                    bs,
                    h_v,
                    p_v,
                    bv,
                    data.charge_s,
                    data.charge_v,
                    role_s,
                    role_v,
                    normal_s,
                    normal_v,
                    atom_dipole_s,
                    atom_dipole_v,
                    cross_edge_index=cross_edge_index,
                    interaction_feature_mask=interaction_feature_mask,
                )
            aux_tensors = {}
            aux_groups = [("cross", self.micro_inter.last_aux)]
            if not self.disable_cross_refine:
                aux_groups.append(("cross_refine", self.micro_inter_refine.last_aux))
            for prefix, aux in aux_groups:
                for key, value in aux.items():
                    aux_tensors[f"{prefix}_{key}"] = value
            self.last_physics_aux = aux_tensors

        # Preserve node-wise variation with a residual path; pure external-attention
        # can collapse different atoms to the same memory readout.
        attn_s = h_s + (self.ext_attn_residual_scale * self.ext_attn(h_s))
        attn_v = h_v + (self.ext_attn_residual_scale * self.ext_attn(h_v))

        z_s_mean = global_mean_pool(attn_s, bs)
        z_s_max = global_max_pool(attn_s, bs)
        z_v_mean = global_mean_pool(attn_v, bv)
        z_v_max = global_max_pool(attn_v, bv)
        z_pair_mul = z_s_mean * z_v_mean
        z_pair_gap = torch.abs(z_s_mean - z_v_mean)

        # 最终系统表征同时保留 pooled 图特征和分子级 RDKit 描述符。
        z_sys = torch.cat(
            [
                z_s_mean,
                z_s_max,
                z_v_mean,
                z_v_max,
                z_pair_mul,
                z_pair_gap,
                data.rdkit_s,
                data.rdkit_v,
            ],
            dim=-1,
        )
        z_sys = self.readout_norm(z_sys)
        z_sys = self.readout_dropout(z_sys)
        return z_sys


class ParameterHead(nn.Module):
    """从系统级表征回归单个热力学参数。"""

    def __init__(self, in_dim, hidden_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(p=0.25),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Dropout(p=0.15),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, xg):
        return self.mlp(xg)


class LogGammaFormulaLayer(nn.Module):
    """Apply log-gamma = K1 + K2 / T_K to the two predicted parameters."""

    def forward(self, k1, k2, temp_c):
        temp_k = torch.clamp(temp_c.view(-1, 1).float() + 273.15, min=10.0)
        return k1 + (k2 / temp_k)


class PGSSIParametricModel(nn.Module):
    """PGSSI backbone that keeps the 3D encoder and predicts only K1/K2."""

    def __init__(
        self,
        hidden_dim=256,
        enable_cross_interaction=True,
        num_intra_layers=2,
        use_interaction_types=True,
        use_moe=True,
        use_physics_prior=True,
        disable_cross_refine=False,
        topology_only=False,
        direct_loggamma_head=False,
        disable_formula_layer=False,
    ):
        super().__init__()
        self.direct_loggamma_head = direct_loggamma_head
        self.disable_formula_layer = disable_formula_layer
        self.shared_layer = Shared_Layer(
            hidden_dim=hidden_dim,
            enable_cross_interaction=enable_cross_interaction,
            num_intra_layers=num_intra_layers,
            use_interaction_types=use_interaction_types,
            use_moe=use_moe,
            use_physics_prior=use_physics_prior,
            disable_cross_refine=disable_cross_refine,
            topology_only=topology_only,
        )
        self.task_k1 = ParameterHead(self.shared_layer.out_dim, hidden_dim)
        self.task_k2 = ParameterHead(self.shared_layer.out_dim, hidden_dim)
        self.direct_head = ParameterHead(self.shared_layer.out_dim, hidden_dim)
        self.formula = LogGammaFormulaLayer()

    def encode_parameters(self, data):
        # 共享编码器输出一个系统向量，再由两个头分别预测 K1 和 K2。
        xg = self.shared_layer(data)
        if self.direct_loggamma_head:
            direct_log_gamma = self.direct_head(xg)
            zero_like = torch.zeros_like(direct_log_gamma)
            return xg, zero_like, zero_like, direct_log_gamma
        k1 = self.task_k1(xg)
        k2 = self.task_k2(xg)
        direct_log_gamma = (k1 + k2) if self.disable_formula_layer else None
        return xg, k1, k2, direct_log_gamma

    def predict_parameters(self, data):
        _, k1, k2, _ = self.encode_parameters(data)
        return k1, k2

    def predict_log_gamma(self, data):
        _, k1, k2, direct_log_gamma = self.encode_parameters(data)
        if self.direct_loggamma_head or self.disable_formula_layer:
            return direct_log_gamma
        return self.formula(k1, k2, data.temp)


class PGSSIModel(PGSSIParametricModel):
    """PGSSI model with formula-based output from the predicted K1/K2 parameters."""

    def forward(self, data, return_dict=False):
        # 最终监督目标是 log-gamma，但中间参数 K1/K2 也可以一起返回。
        _, k1, k2, direct_log_gamma = self.encode_parameters(data)
        log_gamma = direct_log_gamma if (self.direct_loggamma_head or self.disable_formula_layer) else self.formula(k1, k2, data.temp)
        if return_dict:
            return {
                "k1": k1,
                "k2": k2,
                "log_gamma": log_gamma,
                "physics_aux": getattr(self.shared_layer, "last_physics_aux", {}),
            }
        return log_gamma


class PGSSI3DMTLModel(PGSSIParametricModel):
    """Compatibility wrapper for the parametric PGSSI model."""

    def __getitem__(self, index):
        # 保留旧接口：有些旧代码会把模型当作“两个任务头的容器”来访问。
        if index == 0:
            return self.task_k1
        if index == 1:
            return self.task_k2
        raise IndexError("Index is out of range for the task heads of PGSSI3DMTLModel")

    def forward(self, data, return_dict=True):
        # 兼容旧调用方式：默认返回字典，而不是只返回 log-gamma。
        _, k1, k2, direct_log_gamma = self.encode_parameters(data)
        if return_dict:
            return {
                "k1": k1,
                "k2": k2,
                "log_gamma": direct_log_gamma if (self.direct_loggamma_head or self.disable_formula_layer) else self.formula(k1, k2, data.temp),
                "physics_aux": getattr(self.shared_layer, "last_physics_aux", {}),
            }
        return k1, k2
