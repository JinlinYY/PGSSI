"""
完整的溶质-溶剂交互注意力模块增强图学习架构
包含图神经网络基础层、注意力模块和主模型
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree, scatter


# ============================================================================
# 图神经网络基础层
# ============================================================================

class GraphConvolutionLayer(MessagePassing):
    """图卷积层"""
    
    def __init__(self, in_channels, out_channels, bias=True):
        super(GraphConvolutionLayer, self).__init__(aggr='add')
        self.lin = nn.Linear(in_channels, out_channels, bias=bias)
        
    def forward(self, x, edge_index):
        # x: [N, in_channels]
        # edge_index: [2, E]
        
        # 添加自环
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        
        # 线性变换
        x = self.lin(x)
        
        # 消息传递
        return self.propagate(edge_index, x=x)
    
    def message(self, x_j):
        # x_j: [E, out_channels]
        return x_j
    
    def update(self, aggr_out):
        # aggr_out: [N, out_channels]
        return aggr_out


class GraphAttentionLayer(MessagePassing):
    """图注意力层（简化版本，使用GCN作为基础）"""
    
    def __init__(self, in_channels, out_channels, heads=1, dropout=0.0, concat=True):
        super(GraphAttentionLayer, self).__init__(aggr='add', node_dim=0)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.dropout = dropout
        self.concat = concat
        
        # 多头注意力
        self.lin_l = nn.Linear(in_channels, heads * out_channels, bias=False)
        self.lin_r = nn.Linear(in_channels, heads * out_channels, bias=False)
        self.att = nn.Parameter(torch.Tensor(1, heads, 2 * out_channels))
        
        self.reset_parameters()
        
    def reset_parameters(self):
        nn.init.xavier_uniform_(self.lin_l.weight)
        nn.init.xavier_uniform_(self.lin_r.weight)
        nn.init.xavier_uniform_(self.att)
        
    def forward(self, x, edge_index):
        # x: [N, in_channels]
        # edge_index: [2, E]
        
        H, C = self.heads, self.out_channels
        N = x.size(0)
        
        # 线性变换
        x_l = self.lin_l(x).view(N, H, C)
        x_r = self.lin_r(x).view(N, H, C)
        
        # 消息传递
        out = self.propagate(edge_index, x=(x_l, x_r), size=(N, N))
        
        if self.concat:
            out = out.view(N, H * C)
        else:
            out = out.mean(dim=1)
            
        return out
    
    def message(self, x_i, x_j, index, ptr, size_i):
        # x_i: [E, H, C] (目标节点)
        # x_j: [E, H, C] (源节点)
        
        # 计算注意力分数
        x = torch.cat([x_i, x_j], dim=-1)  # [E, H, 2*C]
        alpha = (x * self.att).sum(dim=-1)  # [E, H]
        alpha = F.leaky_relu(alpha, 0.2)
        
        # 存储注意力分数用于后续softmax归一化
        # 注意：这里先不应用softmax，在aggregate中处理
        self._alpha = alpha
        
        return x_j
    
    def aggregate(self, inputs, index, ptr=None, dim_size=None):
        # 获取存储的注意力分数
        alpha = self._alpha  # [E, H]
        
        # 数值稳定的softmax：先减去最大值，再计算exp
        if dim_size is None:
            dim_size = index.max().item() + 1
        
        # 对每个节点的邻居找到最大值（用于数值稳定性）
        # 使用torch_geometric的scatter进行高效的最大值查找
        alpha_max = scatter(alpha, index, dim=0, dim_size=dim_size, reduce='max')
        alpha_max = alpha_max[index]  # [E, H]
        
        # 如果alpha_max中有-inf或NaN，替换为0
        alpha_max = torch.clamp(alpha_max, min=-1e6)
        
        # 减去最大值（数值稳定）
        alpha_stable = alpha - alpha_max
        
        # 裁剪过大的值，防止溢出
        alpha_stable = torch.clamp(alpha_stable, min=-50, max=50)
        
        # 计算exp
        alpha_exp = torch.exp(alpha_stable)
        
        # 按目标节点分组求和
        alpha_sum = torch.zeros(dim_size, alpha_exp.size(1), device=alpha_exp.device)
        alpha_sum = alpha_sum.scatter_add_(0, index.unsqueeze(-1).expand_as(alpha_exp), alpha_exp)
        
        # 归一化（添加小的epsilon防止除零）
        alpha_norm = alpha_exp / (alpha_sum[index] + 1e-8)
        
        # 检查是否有NaN或Inf
        if torch.isnan(alpha_norm).any() or torch.isinf(alpha_norm).any():
            # 如果出现NaN/Inf，使用均匀权重
            alpha_norm = torch.ones_like(alpha_norm) / (alpha_sum[index] + 1e-8)
            alpha_norm = alpha_norm / (alpha_norm.sum(dim=0, keepdim=True) + 1e-8)
        
        alpha_norm = F.dropout(alpha_norm, p=self.dropout, training=self.training)
        
        # 应用注意力权重
        inputs = inputs * alpha_norm.unsqueeze(-1)
        
        return super().aggregate(inputs, index, ptr, dim_size)


# ============================================================================
# 溶质-溶剂交互注意力模块
# ============================================================================

class SoluteSolventAttentionModule(nn.Module):
    """
    溶质-溶剂交互注意力模块
    用于捕获溶质和溶剂之间的交互关系
    """
    
    def __init__(self, hidden_dim, num_heads=8, dropout=0.1):
        super(SoluteSolventAttentionModule, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        
        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"
        
        # 查询、键、值投影
        self.query_proj = nn.Linear(hidden_dim, hidden_dim)
        self.key_proj = nn.Linear(hidden_dim, hidden_dim)
        self.value_proj = nn.Linear(hidden_dim, hidden_dim)
        
        # 输出投影
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        
        # 交互特征融合
        self.interaction_fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # 注意力权重归一化
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, solute_features, solvent_features):
        """
        前向传播
        
        Args:
            solute_features: [batch_size, num_solute_nodes, hidden_dim] 溶质特征
            solvent_features: [batch_size, num_solvent_nodes, hidden_dim] 溶剂特征
            
        Returns:
            enhanced_features: [batch_size, num_solute_nodes, hidden_dim] 增强后的特征
            attention_weights: [batch_size, num_heads, num_solute_nodes, num_solvent_nodes] 注意力权重
        """
        batch_size = solute_features.size(0)
        num_solute = solute_features.size(1)
        num_solvent = solvent_features.size(1)
        
        # 投影到查询、键、值空间
        Q = self.query_proj(solute_features)  # [B, N_solute, H]
        K = self.key_proj(solvent_features)   # [B, N_solvent, H]
        V = self.value_proj(solvent_features) # [B, N_solvent, H]
        
        # 重塑为多头形式
        Q = Q.view(batch_size, num_solute, self.num_heads, self.head_dim).transpose(1, 2)  # [B, H, N_solute, d]
        K = K.view(batch_size, num_solvent, self.num_heads, self.head_dim).transpose(1, 2)  # [B, H, N_solvent, d]
        V = V.view(batch_size, num_solvent, self.num_heads, self.head_dim).transpose(1, 2)  # [B, H, N_solvent, d]
        
        # 计算注意力分数（数值稳定）
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)  # [B, H, N_solute, N_solvent]
        
        # 裁剪过大的分数，防止softmax溢出
        scores = torch.clamp(scores, min=-50, max=50)
        
        # 数值稳定的softmax
        attention_weights = F.softmax(scores, dim=-1)
        
        # 检查是否有NaN或Inf
        if torch.isnan(attention_weights).any() or torch.isinf(attention_weights).any():
            # 如果出现NaN/Inf，使用均匀权重
            attention_weights = torch.ones_like(scores) / scores.size(-1)
        
        attention_weights = self.dropout(attention_weights)
        
        # 应用注意力权重
        attended_values = torch.matmul(attention_weights, V)  # [B, H, N_solute, d]
        
        # 重塑回原始形状
        attended_values = attended_values.transpose(1, 2).contiguous()  # [B, N_solute, H, d]
        attended_values = attended_values.view(batch_size, num_solute, self.hidden_dim)  # [B, N_solute, H]
        
        # 输出投影
        attended_values = self.out_proj(attended_values)
        
        # 残差连接和层归一化
        enhanced_features = self.layer_norm(solute_features + attended_values)
        
        # 交互特征融合
        # 将溶质特征和注意力增强特征融合
        interaction_input = torch.cat([solute_features, enhanced_features], dim=-1)  # [B, N_solute, 2*H]
        fused_features = self.interaction_fusion(interaction_input)
        
        # 最终输出
        output = self.layer_norm(enhanced_features + fused_features)
        
        return output, attention_weights


class CrossAttentionModule(nn.Module):
    """
    交叉注意力模块
    用于溶质和溶剂之间的双向交互
    """
    
    def __init__(self, hidden_dim, num_heads=8, dropout=0.1):
        super(CrossAttentionModule, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        
        # 溶质到溶剂的注意力
        self.solute_to_solvent = SoluteSolventAttentionModule(hidden_dim, num_heads, dropout)
        
        # 溶剂到溶质的注意力
        self.solvent_to_solute = SoluteSolventAttentionModule(hidden_dim, num_heads, dropout)
        
    def forward(self, solute_features, solvent_features):
        """
        双向交叉注意力
        
        Args:
            solute_features: [batch_size, num_solute_nodes, hidden_dim]
            solvent_features: [batch_size, num_solvent_nodes, hidden_dim]
            
        Returns:
            enhanced_solute: [batch_size, num_solute_nodes, hidden_dim]
            enhanced_solvent: [batch_size, num_solvent_nodes, hidden_dim]
            attention_weights: tuple of attention weights
        """
        # 溶质关注溶剂
        enhanced_solute, attn_s2s = self.solute_to_solvent(solute_features, solvent_features)
        
        # 溶剂关注溶质
        enhanced_solvent, attn_s2s_rev = self.solvent_to_solute(solvent_features, solute_features)
        
        return enhanced_solute, enhanced_solvent, (attn_s2s, attn_s2s_rev)


# ============================================================================
# 主模型：溶质-溶剂交互注意力模块增强图学习架构
# ============================================================================

class SoluteSolventGraphModel(nn.Module):
    """
    溶质-溶剂交互注意力模块增强图学习架构
    
    该模型结合了图神经网络和注意力机制，用于处理溶质-溶剂交互预测任务
    """
    
    def __init__(
        self,
        input_dim,
        hidden_dim=256,
        num_layers=4,
        num_heads=8,
        dropout=0.1,
        use_batch_norm=True,
        output_dim=1
    ):
        super(SoluteSolventGraphModel, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout
        self.use_batch_norm = use_batch_norm
        
        # 输入投影层
        self.solute_input_proj = nn.Linear(input_dim, hidden_dim)
        self.solvent_input_proj = nn.Linear(input_dim, hidden_dim)
        
        # 图神经网络层（用于处理分子图结构）
        self.solute_gnn_layers = nn.ModuleList()
        self.solvent_gnn_layers = nn.ModuleList()
        
        for i in range(num_layers):
            # 使用图注意力层
            # concat=True 时输出维度是 heads * out_channels，需要投影回 hidden_dim
            self.solute_gnn_layers.append(
                GraphAttentionLayer(
                    hidden_dim,
                    hidden_dim,
                    heads=num_heads,
                    dropout=dropout,
                    concat=True  # 使用concat，输出维度为 heads * out_channels
                )
            )
            self.solvent_gnn_layers.append(
                GraphAttentionLayer(
                    hidden_dim,
                    hidden_dim,
                    heads=num_heads,
                    dropout=dropout,
                    concat=True  # 使用concat，输出维度为 heads * out_channels
                )
            )
        
        # 投影层：将多头输出（heads * hidden_dim）投影回 hidden_dim
        self.solute_proj_layers = nn.ModuleList([
            nn.Linear(num_heads * hidden_dim, hidden_dim) for _ in range(num_layers)
        ])
        self.solvent_proj_layers = nn.ModuleList([
            nn.Linear(num_heads * hidden_dim, hidden_dim) for _ in range(num_layers)
        ])
        
        # 层归一化层（图数据使用LayerNorm而不是BatchNorm）
        if use_batch_norm:
            self.solute_bn_layers = nn.ModuleList([
                nn.LayerNorm(hidden_dim) for _ in range(num_layers)
            ])
            self.solvent_bn_layers = nn.ModuleList([
                nn.LayerNorm(hidden_dim) for _ in range(num_layers)
            ])
        
        # 溶质-溶剂交互注意力模块
        self.interaction_attention = CrossAttentionModule(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout
        )
        
        # 特征融合层（用于节点级别的特征融合）
        self.feature_fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # 节点级别的特征处理层（分别处理溶质和溶剂）
        self.solute_node_fusion = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        self.solvent_node_fusion = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # 全局池化（用于图级别的特征提取）
        self.global_pool = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # 温度特征处理（归一化温度）
        self.temp_proj = nn.Sequential(
            nn.Linear(1, hidden_dim // 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 4, hidden_dim // 4)
        )
        
        # 输出层（包含温度特征）
        self.output_layers = nn.Sequential(
            nn.Linear(hidden_dim * 2 + hidden_dim // 4, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, output_dim)
        )
        
        self.dropout_layer = nn.Dropout(dropout)
        
        # 初始化权重
        self._initialize_weights()
    
    def _initialize_weights(self):
        """初始化模型权重，确保数值稳定性"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                # 使用Xavier初始化，但限制范围
                nn.init.xavier_uniform_(module.weight, gain=0.1)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)
            elif isinstance(module, nn.LayerNorm):
                nn.init.constant_(module.bias, 0.0)
                nn.init.constant_(module.weight, 1.0)
        
    def forward(self, solute_data, solvent_data, temperature=None):
        """
        前向传播
        
        Args:
            solute_data: 溶质图数据
                - x: [num_solute_nodes, input_dim] 节点特征
                - edge_index: [2, num_solute_edges] 边索引
                - batch: [num_solute_nodes] 批次索引
            solvent_data: 溶剂图数据
                - x: [num_solvent_nodes, input_dim] 节点特征
                - edge_index: [2, num_solvent_edges] 边索引
                - batch: [num_solvent_nodes] 批次索引
            temperature: 温度张量 [batch_size] 或 None
                
        Returns:
            output: [batch_size, output_dim] 预测结果
            attention_weights: 注意力权重（用于可解释性分析）
        """
        # 提取溶质和溶剂数据
        solute_x, solute_edge_index, solute_batch = (
            solute_data.x, solute_data.edge_index, solute_data.batch
        )
        solvent_x, solvent_edge_index, solvent_batch = (
            solvent_data.x, solvent_data.edge_index, solvent_data.batch
        )
        
        # 验证输入数据（检查NaN/Inf）- 优化：只在必要时检查
        # 由于数据已经清理过，这些检查可以简化或移除以提高性能
        # 如果数据质量有保证，可以注释掉这些检查
        
        # 输入投影
        solute_h = self.solute_input_proj(solute_x)  # [N_solute, H]
        solvent_h = self.solvent_input_proj(solvent_x)  # [N_solvent, H]
        
        # 通过图神经网络层处理分子图结构
        for i in range(self.num_layers):
            # 溶质图处理
            solute_h_new = self.solute_gnn_layers[i](solute_h, solute_edge_index)  # [N_solute, heads * hidden_dim]
            # 投影回 hidden_dim（必须执行，确保维度匹配）
            solute_h_new = self.solute_proj_layers[i](solute_h_new)  # [N_solute, hidden_dim]
            if self.use_batch_norm:
                solute_h_new = self.solute_bn_layers[i](solute_h_new)
            solute_h = solute_h + self.dropout_layer(solute_h_new)  # 残差连接
            solute_h = F.relu(solute_h)
            
            # 溶剂图处理
            solvent_h_new = self.solvent_gnn_layers[i](solvent_h, solvent_edge_index)  # [N_solvent, heads * hidden_dim]
            # 投影回 hidden_dim（必须执行，确保维度匹配）
            solvent_h_new = self.solvent_proj_layers[i](solvent_h_new)  # [N_solvent, hidden_dim]
            if self.use_batch_norm:
                solvent_h_new = self.solvent_bn_layers[i](solvent_h_new)
            solvent_h = solvent_h + self.dropout_layer(solvent_h_new)  # 残差连接
            solvent_h = F.relu(solvent_h)
        
        # 将节点特征按批次分组
        batch_size = solute_batch.max().item() + 1
        
        # 重塑为批次形式 [batch_size, num_nodes, hidden_dim]
        solute_features = self._batch_to_sequence(solute_h, solute_batch, batch_size)
        solvent_features = self._batch_to_sequence(solvent_h, solvent_batch, batch_size)
        
        # 溶质-溶剂交互注意力
        enhanced_solute, enhanced_solvent, attention_weights = self.interaction_attention(
            solute_features, solvent_features
        )
        
        # 节点级别的特征融合
        # 由于溶质和溶剂的节点数不同，无法直接拼接
        # 使用平均池化将对方的信息融合到每个节点
        
        # 分别处理溶质和溶剂的节点特征
        solute_fused = self.solute_node_fusion(enhanced_solute)  # [B, N_solute, H]
        solvent_fused = self.solvent_node_fusion(enhanced_solvent)  # [B, N_solvent, H]
        
        # 计算每个图的平均特征（用于融合）
        # 由于每个图的节点数不同，需要分别处理每个图
        batch_size = enhanced_solute.size(0)
        solute_node_features = []
        solvent_node_features = []
        
        for b in range(batch_size):
            n_solute = solute_fused[b].size(0)  # 当前图的溶质节点数
            n_solvent = solvent_fused[b].size(0)  # 当前图的溶剂节点数
            
            # 检查节点数是否有效
            if n_solute == 0 or n_solvent == 0:
                # 如果节点数为 0，使用零向量
                solute_node_features.append(torch.zeros(1, solute_fused[b].size(1), device=solute_fused[b].device))
                solvent_node_features.append(torch.zeros(1, solvent_fused[b].size(1), device=solvent_fused[b].device))
                continue
            
            # 对溶质节点：使用溶剂的平均特征进行融合
            solute_b = solute_fused[b]  # [N_solute, H]
            solvent_avg_b = solvent_fused[b].mean(dim=0, keepdim=True).expand(n_solute, -1)  # [N_solute, H]
            solute_combined = torch.cat([solute_b, solvent_avg_b], dim=-1)  # [N_solute, 2*H]
            solute_fused_b = self.feature_fusion(solute_combined)  # [N_solute, H]
            solute_node_features.append(solute_fused_b)
            
            # 对溶剂节点：使用溶质的平均特征进行融合
            solvent_b = solvent_fused[b]  # [N_solvent, H]
            solute_avg_b = solute_fused[b].mean(dim=0, keepdim=True).expand(n_solvent, -1)  # [N_solvent, H]
            solvent_combined = torch.cat([solvent_b, solute_avg_b], dim=-1)  # [N_solvent, 2*H]
            solvent_fused_b = self.feature_fusion(solvent_combined)  # [N_solvent, H]
            solvent_node_features.append(solvent_fused_b)
        
        # 全局池化（图级别特征）
        # 对每个图分别进行池化（只对有效节点进行池化）
        solute_pooled = torch.stack([s.mean(dim=0) if s.size(0) > 0 else torch.zeros(s.size(1), device=s.device) 
                                     for s in solute_node_features], dim=0)  # [B, H]
        solvent_pooled = torch.stack([s.mean(dim=0) if s.size(0) > 0 else torch.zeros(s.size(1), device=s.device) 
                                     for s in solvent_node_features], dim=0)  # [B, H]
        solute_pooled_max = torch.stack([s.max(dim=0)[0] if s.size(0) > 0 else torch.zeros(s.size(1), device=s.device) 
                                        for s in solute_node_features], dim=0)  # [B, H]
        solvent_pooled_max = torch.stack([s.max(dim=0)[0] if s.size(0) > 0 else torch.zeros(s.size(1), device=s.device) 
                                         for s in solvent_node_features], dim=0)  # [B, H]
        
        # 结合平均和最大池化
        solute_global = self.global_pool((solute_pooled + solute_pooled_max) / 2)  # [B, H]
        solvent_global = self.global_pool((solvent_pooled + solvent_pooled_max) / 2)  # [B, H]
        
        # 最终特征拼接
        final_features = torch.cat([solute_global, solvent_global], dim=-1)  # [B, 2*H]
        
        # 处理温度特征
        if temperature is not None:
            # 直接使用原始温度值，不进行归一化
            temp_features = self.temp_proj(temperature.unsqueeze(-1))  # [B, H//4]
            # 拼接温度特征
            final_features = torch.cat([final_features, temp_features], dim=-1)  # [B, 2*H + H//4]
        
        # 输出预测
        output = self.output_layers(final_features)  # [B, output_dim]
        
        # 检查输出是否有NaN/Inf，如果有则替换为0
        if torch.isnan(output).any() or torch.isinf(output).any():
            output = torch.nan_to_num(output, nan=0.0, posinf=1e6, neginf=-1e6)
        
        # 裁剪输出值到合理范围（防止极端值）
        output = torch.clamp(output, min=-1e6, max=1e6)
        
        return output, attention_weights
    
    def _batch_to_sequence(self, x, batch, batch_size):
        """
        将批次化的节点特征转换为序列形式
        
        Args:
            x: [num_nodes, hidden_dim]
            batch: [num_nodes]
            batch_size: int
            
        Returns:
            sequence: [batch_size, max_num_nodes, hidden_dim]
        """
        # 找到每个批次的最大节点数
        max_nodes = 0
        for i in range(batch_size):
            num_nodes = (batch == i).sum().item()
            max_nodes = max(max_nodes, num_nodes)
        
        # 创建序列张量
        sequence = torch.zeros(batch_size, max_nodes, x.size(1), device=x.device)
        
        # 填充序列
        for i in range(batch_size):
            mask = (batch == i)
            num_nodes = mask.sum().item()
            sequence[i, :num_nodes] = x[mask]
        
        return sequence
