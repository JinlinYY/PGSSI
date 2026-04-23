import torch
import torch.nn as nn
import torch.optim
import torch.utils.data
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool
from torch_geometric.data import Data
import torch_geometric.nn as gnn
import numpy as np
from torch_geometric.utils import to_dense_adj
from torch_scatter import scatter_mean
from torch_scatter import scatter_add
from scr.models.utilities_v2.mol2graph import cat_dummy_graph

# 设置设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class ExternalAttentionLayer(nn.Module):
    def __init__(self, in_features, hidden_dim,u_in, memory_size):
        super(ExternalAttentionLayer, self).__init__()
        self.memory_size = memory_size
        self.in_features = in_features
        self.hidden_dim = hidden_dim

        # 外部记忆矩阵 Mk 和 Mv
        self.Mk = nn.Parameter(torch.randn(memory_size, hidden_dim * 2 + u_in))
        self.Mv = nn.Parameter(torch.randn(memory_size, hidden_dim * 2 + u_in))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        # x形状: (batch_size, num_nodes, in_features)
        batch_size,num_nodes, _ = x.shape

        # 计算注意力分数，形状变化为 (batch_size, num_nodes, memory_size)
        attention_scores = torch.matmul(x, self.Mk.T)

        # 计算注意力权重，形状变化为 (batch_size, num_nodes, memory_size)
        attention_weights = self.softmax(attention_scores)

        # 使用注意力权重聚合 Mv，形状变化为 (batch_size, num_nodes, hidden_dim)
        out = torch.matmul(attention_weights, self.Mv)

        return out


class MPNNconv(nn.Module):
    def __init__(self, node_in_feats, edge_in_feats, node_out_feats,
                 edge_hidden_feats=32, num_step_message_passing=1):
        super(MPNNconv, self).__init__()

        self.project_node_feats = nn.Sequential(
            nn.Linear(node_in_feats, node_out_feats),#(38,38)
            nn.ReLU()
        )
        self.num_step_message_passing = num_step_message_passing

        edge_network = nn.Sequential(
            nn.Linear(edge_in_feats, edge_hidden_feats),#（1,32)
            nn.ReLU(),
            nn.Linear(edge_hidden_feats, node_out_feats * node_out_feats)# (32,1444)
        )
        self.gnn_layer = gnn.NNConv(
            node_out_feats,
            node_out_feats,
            edge_network,
            aggr='add'
        )
        self.gru = nn.GRU(node_out_feats, node_out_feats)

    def reset_parameters(self):
        self.project_node_feats[0].reset_parameters()
        self.gnn_layer.reset_parameters()
        for layer in self.gnn_layer.edge_func:
            if isinstance(layer, nn.Linear):
                layer.reset_parameters()
        self.gru.reset_parameters()

    def forward(self, system_graph):

        node_feats = system_graph.x
        edge_index = system_graph.edge_index
        edge_feats = system_graph.edge_attr

        node_feats = self.project_node_feats(node_feats)  # (V, node_out_feats)
        hidden_feats = node_feats.unsqueeze(0)  # (1, V, node_out_feats)

        for _ in range(self.num_step_message_passing):
            if torch.cuda.is_available():
                node_feats = F.relu(self.gnn_layer(x=node_feats.type(torch.FloatTensor).cuda(),
                                                   edge_index=edge_index.type(torch.LongTensor).cuda(),
                                                   edge_attr=edge_feats.type(torch.FloatTensor).cuda()))
            else:
                node_feats = F.relu(self.gnn_layer(x=node_feats.type(torch.FloatTensor),
                                                   edge_index=edge_index.type(torch.LongTensor),
                                                   edge_attr=edge_feats.type(torch.FloatTensor)))
            node_feats, hidden_feats = self.gru(node_feats.unsqueeze(0), hidden_feats)
            node_feats = node_feats.squeeze(0)
        return node_feats

class EdgeModel(torch.nn.Module):
    def __init__(self, v_in, e_in, u_in, hidden_dim):
        super().__init__()
        self.edge_mlp = nn.Sequential(
            nn.Linear(v_in * 2 + e_in + u_in, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

    def forward(self, src, dest, edge_attr, u, batch):
        out = torch.cat([src, dest, edge_attr, u[batch]], axis=1)
        return self.edge_mlp(out).to(device)  # 添加 .to(device)


class NodeModel(torch.nn.Module):
    def __init__(self, v_in, u_in, hidden_dim, memory_size=128, use_external_attention=True, attn_alpha=0.5):
        super().__init__()
        self.use_external_attention = use_external_attention
        self.attn_alpha = attn_alpha
        self.ext_attention = ExternalAttentionLayer(v_in + hidden_dim + u_in, hidden_dim, u_in, memory_size) if use_external_attention else None
        self.pre_norm = nn.LayerNorm(v_in + hidden_dim + u_in)
        self.node_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2 + u_in, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

    def forward(self, x, edge_index, edge_attr, u, batch):
        out = scatter_add(edge_attr, edge_index[1], dim=0, dim_size=x.size(0))
        out = torch.cat([x, out, u[batch]], dim=1)
        out = self.pre_norm(out)
        if self.use_external_attention and self.ext_attention is not None:
            attn_out = self.ext_attention(out.unsqueeze(0)).squeeze(0)
            out = self.attn_alpha * attn_out + (1.0 - self.attn_alpha) * out
        return self.node_mlp(out)

class GlobalModel(torch.nn.Module):
    def __init__(self, u_in, hidden_dim):
        super().__init__()
        self.global_mlp = nn.Sequential(
            nn.Linear(hidden_dim + hidden_dim + u_in, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

    def forward(self, x, edge_index, edge_attr, u, batch):
        node_aggregate = scatter_mean(x, batch, dim=0)
        edge_aggregate = scatter_mean(edge_attr, batch[edge_index[1]], dim=0,
                                      out=edge_attr.new_zeros(node_aggregate.shape))
        out = torch.cat([u, node_aggregate, edge_aggregate], dim=1)
        return self.global_mlp(out).to(device)  # 添加 .to(device)

# 共享表示层:两个图神经网络+池化 输出是混合物的分子指纹
class Shared_Layer(nn.Module):
    def __init__(self, v_in, e_in, u_in, hidden_dim, use_external_attention=True, attn_alpha=0.5, memory_size=128):
        super(Shared_Layer, self).__init__()

        self.graphnet1 = gnn.MetaLayer(EdgeModel(v_in, e_in, u_in, hidden_dim),
                                       NodeModel(v_in, u_in, hidden_dim, memory_size=memory_size, use_external_attention=use_external_attention, attn_alpha=attn_alpha),
                                       GlobalModel(u_in, hidden_dim))
        self.graphnet2 = gnn.MetaLayer(EdgeModel(hidden_dim, hidden_dim, hidden_dim, hidden_dim),
                                       NodeModel(hidden_dim, hidden_dim, hidden_dim, memory_size=memory_size, use_external_attention=use_external_attention, attn_alpha=attn_alpha),
                                       GlobalModel(hidden_dim, hidden_dim))

        self.gnorm1 = gnn.GraphNorm(hidden_dim)
        self.gnorm2 = gnn.GraphNorm(hidden_dim)

        self.pool = global_mean_pool

        self.global_conv1 = MPNNconv(node_in_feats=hidden_dim * 2, #输入为2倍的hidden_dim
                                     edge_in_feats=1,
                                     node_out_feats=hidden_dim * 2) # 普通图神经网络中最重要的一部分

    def generate_sys_graph(self, x, edge_attr, batch_size, n_mols=2):

        src = np.arange(batch_size)
        dst = np.arange(batch_size, n_mols * batch_size)

        # Self-connections (within same molecule)
        self_connection = np.arange(n_mols * batch_size)

        # Biderectional connections (between each molecule in the system)
        # and self-connection
        one_way = np.concatenate((src, dst, self_connection))
        other_way = np.concatenate((dst, src, self_connection))

        edge_index = torch.tensor([list(one_way),
                                   list(other_way)], dtype=torch.long)
        sys_graph = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
        return sys_graph

    def forward(self, solvent, solute):

        # Molecular descriptors based on MOSCED model

        # -- Induction via polarizability
        # -- (i.e., dipole-induced dipole or induced dipole - induced dipole)

        # ---- Atomic polarizability
        ap1 = solvent.ap.reshape(-1, 1)
        ap2 = solute.ap.reshape(-1, 1)

        # ---- Bond polarizability
        bp1 = solvent.bp.reshape(-1, 1)
        bp2 = solute.bp.reshape(-1, 1)

        # -- Polarity via topological polar surface area
        topopsa1 = solvent.topopsa.reshape(-1, 1)
        topopsa2 = solute.topopsa.reshape(-1, 1)

        # -- Hydrogen-bond acidity and basicity
        intra_hb1 = solvent.inter_hb
        intra_hb2 = solute.inter_hb

        u1 = torch.cat((ap1, bp1, topopsa1), axis=1)  # Molecular descriptors solvent
        u2 = torch.cat((ap2, bp2, topopsa2), axis=1)  # Molecular descriptors solute

        #### - Security check for predicting single node graphs (e.g. water)
        single_node_batch = False
        if solvent.edge_attr.shape[0] == 0 or solute.edge_attr.shape[0] == 0:
            solvent = cat_dummy_graph(solvent)
            solute = cat_dummy_graph(solute)
            u1_dummy = torch.tensor([1, 1, 1]).reshape(1, -1)
            u2_dummy = torch.tensor([1, 1, 1]).reshape(1, -1)
            if torch.cuda.is_available():
                u1_dummy = u1_dummy.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
                u1_dummy = u1_dummy.cuda()
                u2_dummy = u1_dummy.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
                u2_dummy = u1_dummy.cuda()

            u1 = torch.cat((u1, u1_dummy), axis=0)
            u2 = torch.cat((u2, u2_dummy), axis=0)
            single_node_batch = True

        # Solvent GraphNet
        x1, edge_attr1, u1 = self.graphnet1(solvent.x,
                                            solvent.edge_index,
                                            solvent.edge_attr,
                                            u1,
                                            solvent.batch)
        x1 = self.gnorm1(x1, solvent.batch)

        x1, edge_attr1, u1 = self.graphnet2(x1,
                                            solvent.edge_index,
                                            edge_attr1,
                                            u1,
                                            solvent.batch)
        x1 = self.gnorm2(x1, solvent.batch)

        xg1 = self.pool(x1, solvent.batch)

        # Solute GraphNet
        x2, edge_attr2, u2 = self.graphnet1(solute.x,
                                            solute.edge_index,
                                            solute.edge_attr,
                                            u2,
                                            solute.batch)
        x2 = self.gnorm1(x2, solute.batch)
        x2, edge_attr2, u2 = self.graphnet2(x2,
                                            solute.edge_index,
                                            edge_attr2,
                                            u2,
                                            solute.batch)
        x2 = self.gnorm2(x2, solute.batch)

        xg2 = self.pool(x2, solute.batch)

        if single_node_batch:  # Eliminate prediction for dummy graph
            xg1 = xg1[:-1, :]
            xg2 = xg2[:-1, :]
            u1 = u1[:-1, :]
            u2 = u2[:-1, :]
            solvent.inter_hb = solvent.inter_hb[:-1]
            solute.inter_hb = solute.inter_hb[:-1]
            batch_size = solvent.y.shape[0] - 1
        else:
            batch_size = solvent.y.shape[0]

        # Intermolecular descriptors
        # -- Hydrogen bonding
        inter_hb = solvent.inter_hb

        # Construct binary system graph
        node_feat = torch.cat((
            torch.cat((xg1, u1), axis=1),
            torch.cat((xg2, u2), axis=1)), axis=0)
        edge_feat = torch.cat((inter_hb.repeat(2),
                               intra_hb1,
                               intra_hb2)).unsqueeze(1)

        binary_sys_graph = self.generate_sys_graph(x=node_feat,
                                                   edge_attr=edge_feat,
                                                   batch_size=batch_size)

        # Binary system fingerprint
        xg = self.global_conv1(binary_sys_graph)
        xg = torch.cat((xg[0:len(xg) // 2, :], xg[len(xg) // 2:, :]), axis=1)

        return xg


'''共享表示层的输出为混合物的分子指纹，将会作为多层感知器的输入来分别预测两个参数'''


# 用于预测参数A的多层感知器
class Task_A(nn.Module):
    def __init__(self,hidden_dim):
        super(Task_A,self).__init__()
        # MLP for A-预测第一个参数使用的多层感知器(三个全连接层)
        self.mlp1a = nn.Linear(hidden_dim * 4, hidden_dim)

        self.mlp2a = nn.Linear(hidden_dim, hidden_dim)

        self.mlp3a = nn.Linear(hidden_dim, 1)
        self.activation = nn.GELU()

    def forward(self,xg):
        A = self.activation(self.mlp1a(xg))

        A = self.activation(self.mlp2a(A))

        A = self.mlp3a(A)

        return A

# 用于预测参数B的多层感知器（增强版：更深的网络和正则化以改善K2预测，包含温度特征）
class Task_B(nn.Module):
    def __init__(self, hidden_dim, use_dropout=True, dropout_rate=0.2, use_batch_norm=True, temp_feat_dim=5):
        super(Task_B, self).__init__()
        # MLP for B-预测第二个参数使用的多层感知器（增强版：4层 + 正则化 + 温度特征）
        self.use_dropout = use_dropout
        self.use_batch_norm = use_batch_norm
        self.temp_feat_dim = temp_feat_dim
        
        # 第一层：将图特征和温度特征concat后输入（hidden_dim*4 + temp_feat_dim -> hidden_dim*2）
        self.mlp1b = nn.Linear(hidden_dim * 4 + temp_feat_dim, hidden_dim * 2)
        self.bn1 = nn.BatchNorm1d(hidden_dim * 2) if use_batch_norm else nn.Identity()
        self.dropout1 = nn.Dropout(dropout_rate) if use_dropout else nn.Identity()
        
        # 第二层
        self.mlp2b = nn.Linear(hidden_dim * 2, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim) if use_batch_norm else nn.Identity()
        self.dropout2 = nn.Dropout(dropout_rate) if use_dropout else nn.Identity()
        
        # 第三层（残差连接的跳跃层）
        self.mlp3b = nn.Linear(hidden_dim, hidden_dim)
        self.bn3 = nn.BatchNorm1d(hidden_dim) if use_batch_norm else nn.Identity()
        self.dropout3 = nn.Dropout(dropout_rate) if use_dropout else nn.Identity()
        
        # 第四层：输出层
        self.mlp4b = nn.Linear(hidden_dim, 1)
        
        self.activation = nn.GELU()

    def forward(self, xg, temp_features=None):
        """
        Forward pass for Task_B (K2 prediction)
        
        Parameters:
        -----------
        xg : torch.Tensor
            共享层的输出特征 (batch_size, hidden_dim * 4)
        temp_features : torch.Tensor, optional
            温度特征 (batch_size, temp_feat_dim)，如果为None则使用零向量
        """
        # 处理温度特征
        if temp_features is None:
            # 如果没有提供温度特征，使用零向量
            batch_size = xg.shape[0]
            device = xg.device
            temp_features = torch.zeros(batch_size, self.temp_feat_dim, device=device)
        else:
            # 确保温度特征在正确的设备上
            if temp_features.device != xg.device:
                temp_features = temp_features.to(xg.device)
        
        # 将图特征和温度特征拼接
        xg_with_temp = torch.cat([xg, temp_features], dim=1)
        
        # 第一层
        B = self.mlp1b(xg_with_temp)
        if self.use_batch_norm and B.shape[0] > 1:  # BatchNorm需要batch_size > 1
            B = self.bn1(B)
        B = self.activation(B)
        B = self.dropout1(B)
        
        # 第二层
        B = self.mlp2b(B)
        if self.use_batch_norm and B.shape[0] > 1:
            B = self.bn2(B)
        B = self.activation(B)
        B = self.dropout2(B)
        
        # 第三层（残差连接风格，虽然这里没有真正的残差，但保持深度）
        B = self.mlp3b(B)
        if self.use_batch_norm and B.shape[0] > 1:
            B = self.bn3(B)
        B = self.activation(B)
        B = self.dropout3(B)
        
        # 输出层（无激活函数）
        B = self.mlp4b(B)
        
        return B

class GHGEAT_MTL(nn.Module):
    def __init__(self,v_in, e_in, u_in, hidden_dim, use_external_attention=True, attn_alpha=0.5, memory_size=128):
        super(GHGEAT_MTL,self).__init__()
        self.shared_layer = Shared_Layer(v_in, e_in, u_in, hidden_dim, use_external_attention=use_external_attention, attn_alpha=attn_alpha, memory_size=memory_size)
        self.task_A = Task_A(hidden_dim)
        self.task_B = Task_B(hidden_dim)

    def __getitem__(self, index):
        # 返回下标对应的结果
        if index == 0:
            return self.task_A
        elif index == 1:
            return self.task_B
        # 如果索引不在可以访问的范围内，报告索引错误
        else:
            raise IndexError("Index is out of range for the task of GHGEAT model")
    def forward(self, solvent, solute):
        xg = self.shared_layer(solvent, solute)
        x_A = self.task_A(xg)
        
        # 提取温度特征（如果存在）
        temp_features = None
        if hasattr(solvent, 'temp_features') and solvent.temp_features is not None:
            temp_features = solvent.temp_features
            
            # 处理批次数据：Batch会将每个图的temp_features堆叠
            # 如果temp_features是按节点存储的（形状为 (num_nodes, temp_feat_dim)），需要按图聚合
            if hasattr(solvent, 'batch') and temp_features is not None:
                if temp_features.dim() == 2:
                    num_nodes = temp_features.shape[0]
                    batch_size = solvent.batch.max().item() + 1
                    
                    if num_nodes == solvent.batch.shape[0]:
                        # temp_features是按节点存储的，需要按图聚合（取每个图第一个节点的特征）
                        # 因为在sys2graph_MTL中，每个图的所有节点都有相同的temp_features
                        unique_indices = []
                        for i in range(batch_size):
                            node_indices = (solvent.batch == i).nonzero(as_tuple=True)[0]
                            if len(node_indices) > 0:
                                unique_indices.append(node_indices[0].item())
                        if len(unique_indices) == batch_size:
                            temp_features = temp_features[unique_indices]
                        else:
                            # 如果索引数量不匹配，使用scatter_mean聚合
                            from torch_scatter import scatter_mean
                            temp_features = scatter_mean(temp_features, solvent.batch, dim=0, dim_size=batch_size)
                    elif num_nodes == batch_size:
                        # temp_features已经是按图存储的（每个图一个特征向量）
                        pass
                    else:
                        temp_features = None
                elif temp_features.dim() == 1:
                    # 一维特征，尝试reshape
                    batch_size = solvent.batch.max().item() + 1
                    temp_feat_dim = 5  # 假设温度特征维度为5
                    if temp_features.shape[0] == batch_size * temp_feat_dim:
                        temp_features = temp_features.view(batch_size, temp_feat_dim)
                    elif temp_features.shape[0] == temp_feat_dim:
                        # 单个样本，扩展到批次
                        temp_features = temp_features.unsqueeze(0).expand(batch_size, -1)
                    else:
                        temp_features = None
        
        x_B = self.task_B(xg, temp_features=temp_features)
        return x_A, x_B
# 计算模型的参数
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)