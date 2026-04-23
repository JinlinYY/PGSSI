"""
GNN-Gibbs-Helmholtz温度训练
"""
import copy
import math
import os
import random
import sys
import time
from collections import OrderedDict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from rdkit import Chem
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    CosineAnnealingWarmRestarts,
    CyclicLR,
    LambdaLR,
    ReduceLROnPlateau
)

# GHGEAT 代码在 src/models/GHGEAT/；公共工具在 src/models/utilities_v2/（仅后者可在本文件夹外）
_MODELS_ROOT = Path(__file__).resolve().parents[1]
if str(_MODELS_ROOT) not in sys.path:
    sys.path.insert(0, str(_MODELS_ROOT))

from GHGEAT_architecture import GHGEAT, count_parameters
from test_model_on_testset import evaluate_on_testset
from utilities_v2.Train_eval import EarlyStopping
from utilities_v2.Train_eval_T import MAE, R2, eval, train
from utilities_v2.mol2graph import (
    get_dataloader_pairs_T,
    n_atom_features,
    n_bond_features,
    sys2graph,
)
from utilities_v2.save_info import save_train_traj


class FirstEpochMAEThresholdExceeded(Exception):
    """第一轮MAE超过阈值，应提前终止试验"""
    def __init__(self, first_epoch_mae: float, threshold: float):
        self.first_epoch_mae = first_epoch_mae
        self.threshold = threshold
        super().__init__(
            f"第一轮MAE ({first_epoch_mae:.6f}) 超过阈值 "
            f"({threshold:.6f})，提前终止试验"
        )

def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, num_cycles=0.5, last_epoch=-1, eta_min=0.0):
    """
    创建一个带warmup的余弦退火学习率调度器
    
    Parameters:
    -----------
    optimizer : torch.optim.Optimizer
        优化器
    num_warmup_steps : int
        Warmup步数（epoch数）
    num_training_steps : int
        总训练步数（epoch数）
    num_cycles : float
        余弦周期的数量，默认0.5（半周期）
    last_epoch : int
        最后一个epoch的索引，默认-1
    eta_min : float
        最小学习率，默认0.0
    
    Returns:
    --------
    torch.optim.lr_scheduler.LambdaLR
        带warmup的余弦退火学习率调度器
    """
    # 优化：预先计算初始学习率，避免每次调用时访问optimizer.defaults
    initial_lr = optimizer.defaults['lr']
    
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            # Warmup阶段：线性增加学习率
            return float(current_step) / float(max(1, num_warmup_steps))
        # Cosine annealing阶段
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        # 计算余弦退火因子，考虑eta_min
        cosine_factor = 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress))
        # 将eta_min映射到学习率范围（使用预先计算的initial_lr）
        lr_scale = (1 - eta_min / initial_lr) * cosine_factor + eta_min / initial_lr
        return lr_scale
    
    return LambdaLR(optimizer, lr_lambda, last_epoch)


def mix_brouwer_data(df_train, brouwer_path='data/raw/Brouwer_2021.csv', mix_ratio=0.15, random_seed=42):
    """
    将Brouwer_2021数据混合到训练集中
    
    Parameters:
    -----------
    df_train : pd.DataFrame
        原始训练数据
    brouwer_path : str
        Brouwer_2021数据文件路径
    mix_ratio : float
        混合比例，默认0.15 (15%)，范围建议0.1-0.2 (10%-20%)
    random_seed : int
        随机种子，确保可复现
    
    Returns:
    --------
    pd.DataFrame
        混合后的训练数据
    """
    # 设置随机种子
    random.seed(random_seed)
    np.random.seed(random_seed)
    
    # 加载Brouwer_2021数据
    df_brouwer = pd.read_csv(brouwer_path)
    
    # 计算需要采样的数量
    n_brouwer = len(df_brouwer)
    n_sample = int(n_brouwer * mix_ratio)
    
    # 随机采样
    sampled_indices = random.sample(range(n_brouwer), n_sample)
    df_brouwer_sampled = df_brouwer.iloc[sampled_indices].copy()
    
    # 确保列名一致（如果需要）
    required_columns = ['Solute_SMILES', 'Solvent_SMILES', 'log-gamma', 'T']
    if not all(col in df_brouwer_sampled.columns for col in required_columns):
        raise ValueError(f"Brouwer_2021数据缺少必需的列: {required_columns}")
    
    # 合并数据
    df_mixed = pd.concat([df_train, df_brouwer_sampled], ignore_index=True)
    
    print(f"原始训练集大小: {len(df_train)}")
    print(f"Brouwer_2021总大小: {n_brouwer}")
    print(f"采样Brouwer_2021数量: {n_sample} ({mix_ratio*100:.1f}%)")
    print(f"混合后训练集大小: {len(df_mixed)}")
    
    return df_mixed

    
def train_GNNGH_T(df, model_name, hyperparameters, mix_brouwer_ratio=None, resume_checkpoint=None, start_epoch_override=None, val_df=None):
    """
    训练GHGEAT模型
    
    Parameters:
    -----------
    df : pd.DataFrame
        训练数据
    model_name : str
        模型名称（用于保存路径）
    hyperparameters : dict
        超参数字典，包含 hidden_dim, lr, n_epochs, batch_size
    mix_brouwer_ratio : float or None, optional
        混合Brouwer_2021数据的比例，默认None表示不混合
    resume_checkpoint : str or None, optional
        检查点文件路径，如果提供则从该检查点恢复训练，默认None表示从头训练
    start_epoch_override : int or None, optional
        强制从指定轮次开始训练，覆盖checkpoint中的epoch信息，默认None表示使用checkpoint中的epoch
    val_df : pd.DataFrame or None, optional
        验证数据，如果提供则使用验证集进行评估和早停，默认None表示使用训练集进行评估
    """
    def _to_epoch_set(value):
        if value is None:
            return set()
        if isinstance(value, (int, float)):
            return {int(value)}
        if isinstance(value, str):
            try:
                return {int(value)}
            except ValueError:
                return set()
        try:
            return {int(item) for item in value}
        except Exception:
            return set()

    scheduler_restart_epochs_relative = _to_epoch_set(hyperparameters.get('scheduler_restart_epochs_relative'))
    scheduler_restart_epochs_absolute = _to_epoch_set(hyperparameters.get('scheduler_restart_epochs_absolute'))
    use_cyclic_lr = hyperparameters.get('use_cyclic_lr', False)
    cyclic_lr_params = hyperparameters.get('cyclic_lr_params', {})
    use_cosine_warm_restarts = hyperparameters.get('use_cosine_warm_restarts', False)
    cosine_restart_params = hyperparameters.get('cosine_restart_params', {})
    scheduler_type = 'plateau'
    def _create_scheduler(optimizer):
        nonlocal scheduler_type
        current_lr = hyperparameters.get('lr', lr)
        if use_cyclic_lr:
            scheduler_type = 'cyclic'
            cyclic_defaults = {
                'base_lr': cyclic_lr_params.get('base_lr', current_lr),
                'max_lr': cyclic_lr_params.get('max_lr', current_lr * 3),
                'step_size_up': cyclic_lr_params.get('step_size_up', 10),
                'mode': cyclic_lr_params.get('mode', 'triangular2'),
                'cycle_momentum': cyclic_lr_params.get('cycle_momentum', False),
                'last_epoch': cyclic_lr_params.get('last_epoch', -1),
            }
            cyclic_defaults.update({k: v for k, v in cyclic_lr_params.items() if k not in cyclic_defaults})
            return CyclicLR(optimizer, **cyclic_defaults)
        if use_cosine_warm_restarts:
            scheduler_type = 'cosine_restart'
            restart_defaults = {
                'T_0': cosine_restart_params.get('T_0', 20),
                'T_mult': cosine_restart_params.get('T_mult', 2),
                'eta_min': cosine_restart_params.get('eta_min', 1e-6),
                'last_epoch': cosine_restart_params.get('last_epoch', -1),
            }
            restart_defaults.update({k: v for k, v in cosine_restart_params.items() if k not in restart_defaults})
            return CosineAnnealingWarmRestarts(optimizer, **restart_defaults)
        scheduler_type = 'plateau'
        return ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=10,
            min_lr=1e-7
        )
    
    def _reset_optimizer_momentum(optimizer):
        """清空 AdamW/momentum 缓存，让重启时重新积累动量"""
        for param_group in optimizer.param_groups:
            for p in param_group['params']:
                state = optimizer.state.get(p, {})
                state.pop('exp_avg', None)
                state.pop('exp_avg_sq', None)
                state.pop('max_exp_avg_sq', None)
                state.pop('momentum_buffer', None)
                state.pop('step', None)
                if state:
                    optimizer.state[p] = state

    checkpoint_interval = max(1, int(hyperparameters.get('checkpoint_interval', 1)))

    # 测试集评估配置（仅在训练完成后评估，避免数据泄露）
    test_eval_path = hyperparameters.get('test_eval_path')
    test_eval_subset = hyperparameters.get('test_eval_subset_size')
    test_eval_batch = int(hyperparameters.get('test_eval_batch_size', 64))
    test_df_cache = None

    # 构建保存路径：检查是否存在ReLU目录（超参数搜索模式）
    # 超参数搜索模式下，模型保存在 ReLU/{model_name}/ 目录下
    # 如果提供了自定义保存路径（微调模式），使用自定义路径
    custom_save_path = hyperparameters.get('custom_save_path')
    checkpoint_save_dir = hyperparameters.get('checkpoint_save_dir')
    training_files_save_dir = hyperparameters.get('training_files_save_dir')
    
    if custom_save_path:
        # 微调模式：使用自定义路径结构
        path = custom_save_path
        if checkpoint_save_dir:
            checkpoint_dir = checkpoint_save_dir
        else:
            checkpoint_dir = os.path.join(path, 'checkpoint')
        if training_files_save_dir:
            training_files_dir = training_files_save_dir
        else:
            training_files_dir = os.path.join(path, 'Training_files')
    else:
        # 常规模式：检查是否存在ReLU目录（超参数搜索模式）
        relu_path = os.path.join(os.getcwd(), 'ReLU', model_name)
        normal_path = os.path.join(os.getcwd(), model_name)
        
        # 优先检查ReLU目录是否存在（超参数搜索模式）
        # 或者检查是否存在ReLU目录下的检查点文件
        checkpoint_in_relu = os.path.join(relu_path, 'checkpoint', f'{model_name}_checkpoint.pth')
        
        if resume_checkpoint and 'ReLU' in resume_checkpoint:
            # 从检查点路径推断：提取目录路径
            path = os.path.dirname(resume_checkpoint)
        elif os.path.exists(relu_path) or os.path.exists(checkpoint_in_relu):
            # ReLU目录存在或检查点存在于ReLU目录：使用ReLU路径（超参数搜索模式）
            path = relu_path
        else:
            # 常规模式：保存到当前目录下的 {model_name}/ 目录
            path = normal_path
        
        checkpoint_dir = os.path.join(path, 'checkpoint')
        training_files_dir = path  # 常规模式下，训练文件保存在主目录
    
    # 创建必要的目录
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir, exist_ok=True)
    if training_files_dir and not os.path.exists(training_files_dir):
        os.makedirs(training_files_dir, exist_ok=True)
    
    checkpoint_path = os.path.join(checkpoint_dir, f'{model_name}_checkpoint.pth')

    # Open report file（保存到训练文件目录）
    report_path = training_files_dir if custom_save_path else path
    report = open(os.path.join(report_path, 'Report_training_' + model_name + '.txt'), 'w', encoding='utf-8')
    def print_report(string, file=report):
        print(string)
        file.write('\n' + string)

    print_report(' Report for ' + model_name)
    print_report('-'*50)
    
    # 记录数据信息
    print_report(f'Training data size: {len(df)}')
    if val_df is not None:
        print_report(f'Validation data size: {len(val_df)}')
    if mix_brouwer_ratio is not None:
        print_report(f'Brouwer_2021 mix ratio: {mix_brouwer_ratio*100:.1f}%')
    
    # Build molecule from SMILES
    mol_column_solvent     = 'Molecule_Solvent'
    df[mol_column_solvent] = df['Solvent_SMILES'].apply(Chem.MolFromSmiles)

    mol_column_solute      = 'Molecule_Solute'
    df[mol_column_solute]  = df['Solute_SMILES'].apply(Chem.MolFromSmiles)
    
    train_index = df.index.tolist()
    
    target = 'log-gamma'
    
    # targets = ['K1', 'K2']
    # scaler = MinMaxScaler()
    # scaler = scaler.fit(df[targets].to_numpy())
    scaler = None
    
    graphs_solv, graphs_solu = 'g_solv', 'g_solu'
    df[graphs_solv], df[graphs_solu] = sys2graph(df, mol_column_solvent, mol_column_solute, target, y_scaler=scaler)
    
    # Hyperparameters（需要在创建验证集dataloader之前获取batch_size）
    hidden_dim        = hyperparameters['hidden_dim']
    lr                = hyperparameters['lr']
    n_epochs          = hyperparameters['n_epochs']
    batch_size        = hyperparameters['batch_size']
    
    # 处理验证集（如果提供）
    val_loader = None
    if val_df is not None:
        val_df = val_df.copy()
        val_df[mol_column_solvent] = val_df['Solvent_SMILES'].apply(Chem.MolFromSmiles)
        val_df[mol_column_solute] = val_df['Solute_SMILES'].apply(Chem.MolFromSmiles)
        val_index = val_df.index.tolist()
        val_df[graphs_solv], val_df[graphs_solu] = sys2graph(val_df, mol_column_solvent, mol_column_solute, target, y_scaler=scaler)
        
        # 获取数据加载参数（禁用所有加速优化，使用最保守的默认值）
        num_workers = hyperparameters.get('num_workers', 0)  # 单进程模式，禁用多进程加速
        pin_memory = hyperparameters.get('pin_memory', False)  # 禁用内存固定
        persistent_workers = hyperparameters.get('persistent_workers', False)  # 禁用持久化工作进程
        prefetch_factor = hyperparameters.get('prefetch_factor', 2)  # 最小预取因子
        
        val_loader = get_dataloader_pairs_T(val_df, 
                                            val_index, 
                                            graphs_solv,
                                            graphs_solu,
                                            batch_size, 
                                            shuffle=False, 
                                            drop_last=False,
                                            num_workers=num_workers,
                                            pin_memory=pin_memory,
                                            persistent_workers=persistent_workers,
                                            prefetch_factor=prefetch_factor)
        print_report(f'✓ 验证集已加载，将使用验证集进行评估和早停')
    fine_tune_epochs  = hyperparameters.get('fine_tune_epochs')
    force_layered_lr  = hyperparameters.get('force_layered_lr', False)
    
    start       = time.time()
    
    # 获取数据加载参数（禁用所有加速优化，使用最保守的默认值）
    num_workers = hyperparameters.get('num_workers', 0)  # 单进程模式，禁用多进程加速
    pin_memory = hyperparameters.get('pin_memory', False)  # 禁用内存固定
    persistent_workers = hyperparameters.get('persistent_workers', False)  # 禁用持久化工作进程
    prefetch_factor = hyperparameters.get('prefetch_factor', 2)  # 最小预取因子
    
    # Data loaders（优化版：多进程加载和内存固定）
    train_loader = get_dataloader_pairs_T(df, 
                                          train_index, 
                                          graphs_solv,
                                          graphs_solu,
                                          batch_size, 
                                          shuffle=True, 
                                          drop_last=True,
                                          num_workers=num_workers,
                                          pin_memory=pin_memory,
                                          persistent_workers=persistent_workers,
                                          prefetch_factor=prefetch_factor)
    
    available_device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(available_device)
    
    # 初始化混合精度训练的GradScaler
    from torch.amp import GradScaler
    use_mixed_precision = hyperparameters.get('use_mixed_precision', False)  # 默认禁用混合精度（避免 GradScaler 兼容性问题）
    scaler = GradScaler('cuda') if (use_mixed_precision and torch.cuda.is_available()) else None
    if scaler is not None:
        print_report('✓ 混合精度训练已启用（FP16/FP32混合）')
    else:
        if not torch.cuda.is_available():
            print_report('⚠️ 混合精度训练已禁用（CPU模式不支持）')
        else:
            print_report('⚠️ 混合精度训练已禁用（use_mixed_precision=False）')
    
    # 检查并报告CUDA使用情况
    print_report('='*60)
    print_report('【设备检查】')
    print_report(f'PyTorch版本: {torch.__version__}')
    print_report(f'CUDA可用: {torch.cuda.is_available()}')
    if torch.cuda.is_available():
        print_report(f'CUDA版本: {torch.version.cuda}')
        print_report(f'GPU设备数量: {torch.cuda.device_count()}')
        print_report(f'当前GPU: {torch.cuda.get_device_name(0)}')
        print_report(f'GPU内存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB')
        print_report(f'使用设备: {device}')
    else:
        print_report(f'⚠️ 警告: CUDA不可用，将使用CPU训练（速度会很慢）')
        print_report(f'使用设备: {device}')
    print_report('='*60)
    
    # Model
    v_in = n_atom_features()
    e_in = n_bond_features()
    u_in = 3 # ap, bp, topopsa
    # 注意力使用比例：1.0表示完全使用注意力（原始行为），0.0-1.0之间可以调整
    attention_weight = hyperparameters.get('attention_weight', 1.0)
    model = GHGEAT(v_in, e_in, u_in, hidden_dim, attention_weight=attention_weight)
    model = model.to(device)
    
    # 🔑 诊断模型参数初始化状态（检查是否有异常大的权重）
    print_report('【模型参数诊断】检查模型参数范围...')
    try:
        param_stats = []
        for name, param in model.named_parameters():
            if param.requires_grad:
                param_min = param.data.min().item()
                param_max = param.data.max().item()
                param_mean = param.data.mean().item()
                param_std = param.data.std().item()
                param_stats.append({
                    'name': name,
                    'min': param_min,
                    'max': param_max,
                    'mean': param_mean,
                    'std': param_std
                })
                # 检查是否有异常大的权重（可能导致数值不稳定）
                if abs(param_max) > 1000 or abs(param_min) > 1000:
                    print_report(f'  ⚠️  {name}: 范围=[{param_min:.2f}, {param_max:.2f}], 均值={param_mean:.2f}, 标准差={param_std:.2f}')
        
        # 特别检查输出层（mlp3a和mlp3b），这些层的权重直接影响预测值
        output_layers = ['mlp3a', 'mlp3b']
        for layer_name in output_layers:
            for name, param in model.named_parameters():
                if layer_name in name and 'weight' in name and param.requires_grad:
                    weight_max = param.data.abs().max().item()
                    if weight_max > 100:
                        print_report(f'  ⚠️  输出层 {name} 的权重绝对值较大: {weight_max:.2f}，可能影响数值稳定性')
        
        print_report('✓ 模型参数诊断完成')
    except Exception as e:
        print_report(f'  ⚠️  参数诊断失败: {e}')
    
    # 输出注意力使用比例信息
    if attention_weight != 1.0:
        print_report(f'注意力使用比例: {attention_weight:.2f} (1.0=完全使用, 0.0=完全跳过)')
    
    # PyTorch 2.0+ 编译优化（可选，可显著加速训练）
    # 默认禁用，因为需要 Triton 且可能有兼容性问题
    use_torch_compile = hyperparameters.get('use_torch_compile', False)  # 默认禁用（避免 Triton 依赖问题）
    model_compiled = False
    if use_torch_compile and hasattr(torch, 'compile'):
        compile_mode = hyperparameters.get('torch_compile_mode', 'reduce-overhead')  # 默认模式
        print_report('')
        print_report('='*60)
        print_report('【模型编译优化】')
        print_report('='*60)
        print_report(f'尝试使用 torch.compile() 优化模型（模式: {compile_mode}）')
        print_report('注意：首次运行会进行编译，可能需要额外时间（通常10-60秒）')
        print_report('编译后的模型运行速度会显著提升（通常20-50%加速）')
        print_report('='*60)
        print_report('')
        
        # 检查 Triton 是否可用（某些模式需要 Triton）
        triton_available = False
        try:
            import triton
            triton_available = True
            print_report('✓ Triton 已安装')
        except ImportError:
            print_report('⚠️ Triton 未安装，某些编译模式可能不可用')
            # 如果模式需要 Triton，切换到不需要 Triton 的模式
            if compile_mode in ['reduce-overhead', 'max-autotune']:
                print_report(f'   模式 {compile_mode} 需要 Triton，将切换到 "default" 模式')
                compile_mode = 'default'
        
        try:
            # 尝试编译模型
            model = torch.compile(model, mode=compile_mode)
            model_compiled = True
            print_report(f'✓ 模型编译成功（模式: {compile_mode}）')
        except Exception as e:
            error_msg = str(e)
            print_report(f'⚠️ 模型编译失败: {error_msg}')
            
            # 如果是 Triton 相关错误，尝试使用不需要 Triton 的模式
            if 'triton' in error_msg.lower() or 'TritonMissing' in str(type(e)):
                print_report('   检测到 Triton 相关错误，尝试使用 "default" 模式（不需要 Triton）')
                try:
                    model = torch.compile(model, mode='default')
                    model_compiled = True
                    print_report('✓ 使用 "default" 模式编译成功')
                except Exception as e2:
                    print_report(f'⚠️ 使用 "default" 模式也失败: {e2}')
                    print_report('   将使用未编译的模型继续训练')
            else:
                print_report('   将使用未编译的模型继续训练')
    elif use_torch_compile and not hasattr(torch, 'compile'):
        print_report('⚠️ torch.compile() 不可用（需要 PyTorch 2.0+），跳过编译优化')
    else:
        print_report('⚠️ torch.compile() 已禁用（use_torch_compile=False）')
    
    # 验证模型确实在正确的设备上（已禁用输出）
    # if torch.cuda.is_available():
    #     next_param = next(model.parameters())
    #     actual_device = next_param.device
    #     if actual_device.type != 'cuda':
    #         print_report(f'⚠️ 警告: 模型参数不在CUDA设备上！实际设备: {actual_device}')
    #     else:
    #         print_report(f'✓ 模型已加载到CUDA设备: {actual_device}')
    
    print('    Number of model parameters: ', count_parameters(model))
    
    # 检查是否有检查点需要恢复（优先级高于预训练权重）
    # 需要在优化器创建之前检查，以便决定是否使用微调学习率
    checkpoint_to_load = resume_checkpoint if resume_checkpoint else checkpoint_path
    has_checkpoint = os.path.exists(checkpoint_to_load)
    
    # 检查是否是.pth文件（只包含模型权重，不包含优化器状态等）
    is_pth_file = False
    if has_checkpoint and checkpoint_to_load.endswith('.pth') and not checkpoint_to_load.endswith('_checkpoint.pth'):
        is_pth_file = True
        print_report(f'检测到.pth文件（仅模型权重），将从该文件加载模型权重')
        print_report(f'注意：.pth文件不包含优化器状态和训练历史，训练将从第0轮开始')
    
    # To save trajectory
    mae_train = []
    r2_train = []
    mae_valid = []  # 验证集MAE历史（如果提供了验证集）
    r2_valid = []   # 验证集R²历史（如果提供了验证集）
    best_MAE = np.inf
    best_model = None
    start_epoch = 0
    
    # 初始化微调策略变量（在加载预训练权重后设置）
    use_pretrained = hyperparameters.get('use_pretrained', False)
    fine_tune_stage = hyperparameters.get('fine_tune_stage', 'two_stage')  # 'output_only', 'full', 'two_stage'
    freeze_shared_layers = False
    freeze_epochs = 0
    original_lr = lr  # 保存原始学习率
    
    # 智能学习率降低策略：根据原始学习率大小动态调整降低倍数
    def get_fine_tune_lr_reduction_factor(base_lr):
        """
        根据基础学习率返回合适的降低倍数
        
        策略：
        - 如果学习率 >= 1e-3: 降低10倍（0.1）
        - 如果学习率 >= 1e-4: 降低5倍（0.2）
        - 如果学习率 >= 1e-5: 降低3倍（0.33）
        - 如果学习率 < 1e-5: 降低2倍（0.5）
        """
        if base_lr >= 1e-3:
            return 0.1  # 降低10倍
        elif base_lr >= 1e-4:
            return 0.2  # 降低5倍
        elif base_lr >= 1e-5:
            return 0.33  # 降低3倍（约）
        else:
            return 0.5  # 降低2倍
    
    fine_tune_lr_factor = get_fine_tune_lr_reduction_factor(original_lr)
    task_type = 'regression'
    
    # 初始化检查点状态变量（如果检查点不存在，这些变量为None）
    optimizer_state_dict = None
    scheduler_state_dict = None
    checkpoint_uses_layered_lr = False  # 标记检查点是否使用分层学习率
    resumed_from_checkpoint = False  # 标记是否从检查点恢复
    resumed_best_mae = None  # 记录恢复时的最佳MAE
    lr_reduced = False  # 标记是否已经降低过学习率（用于从.pth文件加载时）
    
    # 注意：optimizer 和 scheduler 将在加载预训练权重后创建
    # 因为需要根据是否冻结层来决定优化器的参数
    
    # Early stopping mechanism
    early_stop_requires_best_update = hyperparameters.get('early_stop_requires_best_update', False)
    early_stop_resume_patience = hyperparameters.get('early_stopping_patience', hyperparameters.get('patience', 25))
    early_stop_min_delta = hyperparameters.get('early_stopping_min_delta', 1e-4)
    early_stop_pause_patience = hyperparameters.get('early_stopping_pause_patience', 999999)
    early_stopper = EarlyStopping(patience=early_stop_resume_patience, min_delta=early_stop_min_delta)
    early_stop_active = not early_stop_requires_best_update
    print_report(f'早停机制: patience={early_stop_resume_patience}, min_delta={early_stop_min_delta}')
    
    # 如果没有检查点，尝试加载预训练模型权重（选择性加载，避免负迁移）
    if not has_checkpoint:
        # 加载预训练模型权重（PyTorch 2.6 需要设置 weights_only=False 来加载包含 sklearn 对象的文件）
        # 尝试多个可能的路径
        possible_paths = [
            'hyperparameter_research/GHGEAT_Ki_search/GHGEAT_Ki_best/GHGEAT_Ki_best.pth',  # 从项目根目录
            'scr/models/hyperparameter_research/GHGEAT_Ki_search/GHGEAT_Ki_best/GHGEAT_Ki_best.pth',  # 旧路径（向后兼容）
            os.path.join(os.path.dirname(__file__), '..', '..', 'hyperparameter_research', 'GHGEAT_Ki_search', 'GHGEAT_Ki_best', 'GHGEAT_Ki_best.pth'),  # 从当前文件位置
        ]
        
        pretrained_path = None
        for path in possible_paths:
            if os.path.exists(path):
                pretrained_path = path
                break
        
        # 如果所有路径都不存在，使用第一个路径（用于错误提示）
        if pretrained_path is None:
            pretrained_path = possible_paths[0]
        
        use_pretrained = hyperparameters.get('use_pretrained', False)  # 默认禁用预训练权重
        
        if use_pretrained and os.path.exists(pretrained_path):
            try:
                checkpoint = torch.load(pretrained_path, 
                                       map_location=torch.device(available_device), 
                                       weights_only=False)
                
                # 获取预训练权重字典
                if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                    pretrained_dict = checkpoint['model_state_dict']
                else:
                    pretrained_dict = checkpoint
                
                # 智能权重加载：加载所有兼容的权重（共享层 + 输出层）
                # 模仿GHGNN的成功策略：使用strict=False自动匹配所有兼容层
                # GHGNN通过加载所有层（包括输出层）实现了MAE从0.096降到0.041的显著提升
                # 输出层权重提供更好的初始化，即使任务不完全匹配也能快速适应
                model_dict = model.state_dict()
                matched_keys = []
                skipped_keys = []
                
                # 统计加载的层类型
                shared_layer_count = 0
                task_a_count = 0
                task_b_count = 0
                other_count = 0
                
                # 调试：输出预训练模型的键名示例（仅在首次加载时输出，避免重复）
                print_report(f'  预训练模型键名总数: {len(pretrained_dict.keys())}')
                pretrained_key_samples = list(pretrained_dict.keys())[:10]
                print_report(f'  预训练模型键名示例（前10个）:')
                for sample_key in pretrained_key_samples:
                    print_report(f'    - {sample_key}')
                
                # 调试：输出当前模型的键名示例
                model_key_samples = list(model_dict.keys())[:10]
                print_report(f'  当前模型键名示例（前10个）:')
                for sample_key in model_key_samples:
                    print_report(f'    - {sample_key}')
                
                for key in pretrained_dict.keys():
                    model_key = None
                    layer_type = None
                    original_key = key
                    
                    # 处理shared_layer前缀（Ki模型使用MTL架构）
                    # 这些层学习的是通用的分子特征表示，在Ki预测和T预测之间高度可迁移
                    if 'shared_layer.' in key:
                        model_key = key.replace('shared_layer.', '')
                        layer_type = 'shared'
                    # 处理Task_A输出层 -> mlp1a/mlp2a/mlp3a
                    # K1和参数A有一定相关性，输出层权重可能提供更好的初始化
                    elif 'task_A.' in key:
                        model_key = key.replace('task_A.', '')
                        layer_type = 'task_A'
                    # 处理Task_B输出层 -> mlp1b/mlp2b/mlp3b
                    # 虽然K2和参数B相关性较弱，但输出层权重仍可能提供有用的初始化
                    elif 'task_B.' in key:
                        model_key = key.replace('task_B.', '')
                        layer_type = 'task_B'
                    else:
                        # 其他未识别的键，尝试直接匹配
                        model_key = key
                        layer_type = 'other'
                    
                    # 检查是否可以加载
                    if model_key and model_key in model_dict:
                        if model_dict[model_key].shape == pretrained_dict[key].shape:
                            model_dict[model_key] = pretrained_dict[key]
                            matched_keys.append(f"{key} -> {model_key}")
                            # 统计各类型层的数量
                            if layer_type == 'shared':
                                shared_layer_count += 1
                            elif layer_type == 'task_A':
                                task_a_count += 1
                            elif layer_type == 'task_B':
                                task_b_count += 1
                            else:
                                other_count += 1
                        else:
                            skipped_keys.append(f"Shape mismatch: {key} (预训练: {pretrained_dict[key].shape}) vs {model_key} (当前模型: {model_dict[model_key].shape})")
                    elif model_key:
                        # 尝试模糊匹配：通过键名结构匹配
                        found_alternative = False
                        
                        # 方法1：尝试通过键名后缀匹配（例如：graphnet1.node_model.ext_attention.Mk）
                        pretrained_key_parts = key.split('.')
                        pretrained_suffix = pretrained_key_parts[-1] if len(pretrained_key_parts) > 0 else key
                        
                        # 如果预训练键有多个部分，尝试匹配结构
                        if len(pretrained_key_parts) > 1:
                            # 尝试匹配：移除 shared_layer/task_A/task_B 前缀后的结构
                            pretrained_structure = '.'.join(pretrained_key_parts[1:])  # 跳过第一个前缀
                            
                            for model_key_candidate in model_dict.keys():
                                model_key_parts = model_key_candidate.split('.')
                                # 检查结构是否匹配（从第二个部分开始）
                                if len(model_key_parts) >= len(pretrained_key_parts) - 1:
                                    model_structure = '.'.join(model_key_parts[-(len(pretrained_key_parts)-1):])
                                    if model_structure == pretrained_structure:
                                        if model_dict[model_key_candidate].shape == pretrained_dict[key].shape:
                                            model_dict[model_key_candidate] = pretrained_dict[key]
                                            matched_keys.append(f"{key} -> {model_key_candidate} (结构匹配)")
                                            found_alternative = True
                                            if layer_type == 'shared':
                                                shared_layer_count += 1
                                            elif layer_type == 'task_A':
                                                task_a_count += 1
                                            elif layer_type == 'task_B':
                                                task_b_count += 1
                                            else:
                                                other_count += 1
                                            break
                        
                        # 方法2：如果方法1失败，尝试通过最后部分（层名）匹配
                        if not found_alternative:
                            for model_key_candidate in model_dict.keys():
                                model_key_parts = model_key_candidate.split('.')
                                model_suffix = model_key_parts[-1] if len(model_key_parts) > 0 else model_key_candidate
                                
                                # 检查后缀是否相同且形状匹配
                                if pretrained_suffix == model_suffix and model_dict[model_key_candidate].shape == pretrained_dict[key].shape:
                                    # 额外检查：确保键名结构相似（例如都包含 graphnet1 或 gnorm1）
                                    pretrained_middle = '.'.join(pretrained_key_parts[1:-1]) if len(pretrained_key_parts) > 2 else ''
                                    model_middle = '.'.join(model_key_parts[1:-1]) if len(model_key_parts) > 2 else ''
                                    
                                    # 如果中间部分匹配或都为空，则认为是匹配的
                                    if pretrained_middle == model_middle or (not pretrained_middle and not model_middle):
                                        model_dict[model_key_candidate] = pretrained_dict[key]
                                        matched_keys.append(f"{key} -> {model_key_candidate} (后缀匹配)")
                                        found_alternative = True
                                        if layer_type == 'shared':
                                            shared_layer_count += 1
                                        elif layer_type == 'task_A':
                                            task_a_count += 1
                                        elif layer_type == 'task_B':
                                            task_b_count += 1
                                        else:
                                            other_count += 1
                                        break
                        
                        if not found_alternative:
                            skipped_keys.append(f"Key not found in model: {model_key} (原始键: {original_key})")
                
                model.load_state_dict(model_dict, strict=False)
                
                # 为注意力机制层设置更好的权重初始化（因为无法从预训练模型加载）
                # GHGEAT的注意力机制（ExternalAttentionLayer）在MTL版本中不存在或层名不匹配
                # 使用Xavier初始化可以更好地配合预训练的特征提取层
                attention_init_count = 0
                for name, param in model.named_parameters():
                    if 'ext_attention' in name and ('Mk' in name or 'Mv' in name):
                        # 使用Xavier均匀初始化，适合注意力机制
                        # 这有助于注意力层更好地与预训练的特征提取层配合
                        nn.init.xavier_uniform_(param, gain=1.0)
                        attention_init_count += 1
                
                print_report(f'已加载预训练模型权重: {pretrained_path}')
                print_report(f'  策略: 加载所有兼容的权重（共享层 + 输出层，模仿GHGNN成功策略）')
                print_report(f'  成功加载: {len(matched_keys)} 层')
                print_report(f'    - 共享特征提取层: {shared_layer_count} 层（graphnet1/2, gnorm1/2, global_conv1）')
                print_report(f'    - Task_A输出层: {task_a_count} 层（mlp1a/mlp2a/mlp3a）')
                print_report(f'    - Task_B输出层: {task_b_count} 层（mlp1b/mlp2b/mlp3b）')
                if other_count > 0:
                    print_report(f'    - 其他层: {other_count} 层')
                if attention_init_count > 0:
                    print_report(f'    - 注意力机制层: {attention_init_count} 层（使用Xavier初始化，因为无法从预训练模型加载）')
                if len(skipped_keys) > 0:
                    print_report(f'  跳过: {len(skipped_keys)} 层（不兼容）')
                    # 输出前10个跳过的键名，帮助诊断问题
                    if len(skipped_keys) <= 10:
                        for skip_info in skipped_keys:
                            print_report(f'    - {skip_info}')
                    else:
                        for skip_info in skipped_keys[:10]:
                            print_report(f'    - {skip_info}')
                        print_report(f'    ... 还有 {len(skipped_keys) - 10} 个键被跳过')
                if len(matched_keys) == 0:
                    print_report('  警告: 没有成功加载任何权重，可能架构不匹配')
                    print_report('  建议: 如果不需要预训练权重，请在超参数中设置 use_pretrained=False')
                else:
                    # 设置分阶段微调策略，保护预训练权重
                    if fine_tune_stage == 'none' or fine_tune_stage is None:
                        # 不使用任何微调策略，直接使用原始学习率进行常规训练
                        # print_report(f'\n{"="*60}')
                        # print_report(f'【常规训练策略】不使用任何微调策略')
                        # print_report(f'{"="*60}')
                        # print_report(f'  - 已加载预训练权重')
                        # print_report(f'  - 使用原始学习率: {original_lr:.6f}')
                        # print_report(f'  - 所有层都参与训练，不降低学习率，不冻结层')
                        # print_report(f'{"="*60}\n')
                        # 不修改lr，使用原始学习率
                        pass  # 不做任何操作，使用原始学习率
                    elif fine_tune_stage == 'two_stage':
                        # 两阶段微调：先只训练输出层，再训练所有层
                        freeze_epochs = hyperparameters.get('freeze_epochs', max(1, int(n_epochs * 0.2)))  # 默认20%的epoch冻结共享层
                        freeze_shared_layers = True
                        # print_report(f'\n{"="*60}')
                        # print_report(f'【两阶段微调策略】保护预训练权重')
                        # print_report(f'{"="*60}')
                        # print_report(f'  阶段1: 前 {freeze_epochs} 轮冻结共享层，只训练输出层')
                        # print_report(f'     - 学习率: {original_lr:.6f}')
                        # print_report(f'     - 保护预训练的特征提取能力')
                        fine_tune_lr_stage2 = original_lr * fine_tune_lr_factor
                        reduction_factor_str = f"{1/fine_tune_lr_factor:.1f}倍" if fine_tune_lr_factor < 1 else "不变"
                        # print_report(f'  阶段2: 后 {n_epochs - freeze_epochs} 轮解冻所有层，全量微调')
                        # print_report(f'     - 学习率: {fine_tune_lr_stage2:.6f} (降低{reduction_factor_str})')
                        # print_report(f'     - 精细调整所有参数')
                        # print_report(f'{"="*60}\n')
                    elif fine_tune_stage == 'output_only':
                        # 只训练输出层，完全冻结共享层
                        freeze_shared_layers = True
                        freeze_epochs = n_epochs  # 始终冻结
                        # print_report(f'\n{"="*60}')
                        # print_report(f'【输出层微调策略】完全保护预训练权重')
                        # print_report(f'{"="*60}')
                        # print_report(f'  - 冻结所有共享层（graphnet1/2, gnorm1/2, global_conv1）')
                        # print_report(f'  - 只训练输出层（mlp1a/mlp2a/mlp3a, mlp1b/mlp2b/mlp3b）')
                        # print_report(f'  - 学习率: {original_lr:.6f}')
                        # print_report(f'{"="*60}\n')
                    else:
                        # 全量微调，但使用更小的学习率
                        fine_tune_lr = original_lr * fine_tune_lr_factor
                        reduction_factor_str = f"{1/fine_tune_lr_factor:.1f}倍" if fine_tune_lr_factor < 1 else "不变"
                        # print_report(f'\n{"="*60}')
                        # print_report(f'【全量微调策略】使用小学习率保护预训练权重')
                        # print_report(f'{"="*60}')
                        # print_report(f'  - 学习率从 {original_lr:.6f} 降低至 {fine_tune_lr:.6f} (降低{reduction_factor_str})')
                        # print_report(f'  - 所有层都参与训练，但使用小学习率避免破坏预训练权重')
                        # print_report(f'{"="*60}\n')
                        lr = fine_tune_lr
                    
                    # 如果使用冻结策略，现在设置冻结状态
                    if freeze_shared_layers:
                        # 冻结共享特征提取层（graphnet1, graphnet2, gnorm1, gnorm2, global_conv1）
                        frozen_count = 0
                        trainable_count = 0
                        for name, param in model.named_parameters():
                            if any(layer in name for layer in ['graphnet1', 'graphnet2', 'gnorm1', 'gnorm2', 'global_conv1']):
                                param.requires_grad = False
                                frozen_count += 1
                            else:
                                trainable_count += 1
                        
                        frozen_params = sum(p.numel() for p in model.parameters() if not p.requires_grad)
                        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
                        print_report(f'冻结参数层数: {frozen_count}, 可训练层数: {trainable_count}')
                        print_report(f'冻结参数数量: {frozen_params:,}, 可训练参数数量: {trainable_params:,}')
                        print_report(f'冻结比例: {frozen_params/(frozen_params+trainable_params)*100:.1f}%')
            except Exception as e:
                print_report(f'警告: 加载预训练模型权重失败: {e}')
                print_report('使用随机初始化的权重')
        else:
            if not use_pretrained:
                print_report('预训练权重已禁用（use_pretrained=False），使用随机初始化')
            else:
                print_report(f'警告: 预训练模型文件不存在: {pretrained_path}')
                print_report('使用随机初始化的权重')
    
    # 如果有检查点，加载检查点（覆盖预训练权重）
    if has_checkpoint:
        resumed_from_checkpoint = True  # 标记从检查点恢复
        print_report(f'发现检查点文件: {checkpoint_to_load}')
        try:
            checkpoint = torch.load(checkpoint_to_load, map_location=torch.device(available_device), weights_only=False)
            
            # 🔑 检查：如果提供了验证集但checkpoint中没有验证集历史，则不使用checkpoint，从头开始训练
            if val_loader is not None and 'mae_valid' not in checkpoint:
                print_report('')
                print_report('='*60)
                print_report('⚠️  检测到验证集，但checkpoint中无验证集历史数据')
                print_report('   为确保训练过程的完整性和准确性，将放弃checkpoint，从头开始训练')
                print_report('   原因：旧checkpoint不包含验证集历史，无法准确恢复历史最佳MAE和R²')
                print_report('   提示：新训练产生的checkpoint将包含完整的验证集历史')
                print_report('='*60)
                print_report('')
                # 抛出异常，让except块处理，从头开始训练
                raise ValueError('Checkpoint缺少验证集历史，无法准确恢复')
            
            # 验证超参数一致性
            if 'hyperparameters' in checkpoint:
                saved_hparams = checkpoint['hyperparameters']
                if saved_hparams.get('hidden_dim') != hyperparameters.get('hidden_dim'):
                    print_report(f'警告: hidden_dim不匹配 (检查点: {saved_hparams.get("hidden_dim")}, 当前: {hyperparameters.get("hidden_dim")})')
                if saved_hparams.get('batch_size') != hyperparameters.get('batch_size'):
                    print_report(f'警告: batch_size不匹配 (检查点: {saved_hparams.get("batch_size")}, 当前: {hyperparameters.get("batch_size")})')
            
            # 加载模型状态
            # 使用 strict=False 以处理架构变化（例如 input_projection 层的动态创建/删除）
            # 当 attention_weight 不同时，模型可能包含或不包含 input_projection 层
            
            # 处理.pth文件：可能是直接的state_dict，也可能是包含model_state_dict的字典
            if is_pth_file:
                # .pth文件可能是直接的state_dict
                if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                    checkpoint_state = checkpoint['model_state_dict']
                elif isinstance(checkpoint, dict) and not any(k in checkpoint for k in ['optimizer_state_dict', 'scheduler_state_dict', 'epoch', 'best_MAE']):
                    # 如果是一个字典但不包含训练状态信息，可能是直接的state_dict
                    checkpoint_state = checkpoint
                else:
                    # 否则假设是直接的state_dict
                    checkpoint_state = checkpoint
            else:
                # 检查点文件应该包含model_state_dict
                if 'model_state_dict' in checkpoint:
                    checkpoint_state = checkpoint['model_state_dict']
                else:
                    print_report('警告: 检查点文件中未找到model_state_dict，尝试直接使用整个checkpoint作为state_dict')
                    checkpoint_state = checkpoint
            
            # 检查检查点中是否有 input_projection 层
            has_input_proj_in_ckpt = any('input_projection' in k for k in checkpoint_state.keys())
            current_attention_weight = hyperparameters.get('attention_weight', 1.0)
            
            # 检查当前模型是否已经包含 input_projection 层
            model_state_before = model.state_dict()
            has_input_proj_in_model = any('input_projection' in k for k in model_state_before.keys())
            
            # 如果检查点中有 input_projection 但当前模型没有，需要先触发创建
            # 无论 attention_weight 是多少，只要检查点中有但模型中没有，就创建
            if has_input_proj_in_ckpt and not has_input_proj_in_model:
                print_report('')
                print_report('='*60)
                print_report('【动态创建 input_projection 层】')
                print_report('='*60)
                print_report(f'检查点中包含 input_projection 层，但当前模型中没有这些层')
                print_report(f'当前 attention_weight={current_attention_weight}')
                print_report('将临时设置 attention_weight < 1.0 来触发 input_projection 层的创建，然后加载权重')
                print_report('='*60)
                print_report('')
                
                try:
                    # 临时修改模型中所有 NodeModel 的 attention_weight 以触发 input_projection 创建
                    def set_attention_weight_recursive(module, value):
                        if hasattr(module, 'attention_weight'):
                            module.attention_weight = value
                        for child in module.children():
                            set_attention_weight_recursive(child, value)
                    
                    # 保存原始 attention_weight
                    original_attention_weight = current_attention_weight
                    
                    # 临时设置 attention_weight = 0.5 以触发 input_projection 创建
                    set_attention_weight_recursive(model, 0.5)
                    
                    # 使用训练数据的一个样本来触发 forward（创建 input_projection 层）
                    # 注意：此时 df, graphs_solv, graphs_solu 应该已经准备好了
                    # get_dataloader_pairs_T 已在文件顶部导入，无需重复导入
                    temp_loader = get_dataloader_pairs_T(
                        df, 
                        df.index.tolist()[:1],  # 只用第一个样本
                        graphs_solv,
                        graphs_solu,
                        batch_size=1, 
                        shuffle=False, 
                        drop_last=False,
                        num_workers=0,  # 使用单进程避免问题
                        pin_memory=False
                    )
                    
                    model.eval()
                    with torch.no_grad():
                        for batch_data in temp_loader:
                            if len(batch_data) == 3:
                                batch_solvent, batch_solute, T = batch_data
                                batch_solvent = batch_solvent.to(device)
                                batch_solute = batch_solute.to(device)
                                T = T.to(device)
                                # 运行一次 forward 来触发 input_projection 的创建
                                _ = model(batch_solvent, batch_solute, T)
                                print_report('✓ input_projection 层已创建')
                                break
                    
                    # 检查层是否真的创建了，并立即尝试加载权重
                    model_state_temp = model.state_dict()
                    input_proj_created = [k for k in checkpoint_state.keys() if 'input_projection' in k and k in model_state_temp]
                    if input_proj_created:
                        print_report(f'✓ 检测到 {len(input_proj_created)} 个 input_projection 层已创建，立即加载权重...')
                        for key in input_proj_created:
                            if model_state_temp[key].shape == checkpoint_state[key].shape:
                                model_state_temp[key] = checkpoint_state[key]
                        model.load_state_dict(model_state_temp, strict=False)
                        print_report(f'✓ 已加载 {len(input_proj_created)} 个 input_projection 层的权重')
                    
                    # 恢复原始的 attention_weight
                    set_attention_weight_recursive(model, original_attention_weight)
                    print_report(f'✓ 已恢复 attention_weight={original_attention_weight}')
                    
                except Exception as e:
                    print_report(f'⚠️ 警告: 创建 input_projection 层时出错: {e}')
                    print_report('   将尝试直接加载权重（可能会跳过 input_projection 层）')
                    import traceback
                    traceback.print_exc()
            
            model_state = model.state_dict()
            
            # 过滤掉模型中不存在的键，并详细记录加载情况
            filtered_state = {}
            skipped_keys = []
            input_proj_keys = []  # 单独收集 input_projection 键
            matched_keys = []  # 记录成功加载的键
            shape_mismatch_keys = []  # 记录形状不匹配的键
            
            # 统计加载的层类型
            layer_stats = {
                'graphnet': 0,
                'gnorm': 0,
                'global_conv': 0,
                'mlp': 0,
                'input_projection': 0,
                'ext_attention': 0,
                'other': 0
            }
            
            for key, value in checkpoint_state.items():
                if key in model_state:
                    if model_state[key].shape == value.shape:
                        filtered_state[key] = value
                        matched_keys.append(key)
                        # 统计层类型
                        if 'input_projection' in key:
                            layer_stats['input_projection'] += 1
                        elif 'graphnet' in key:
                            layer_stats['graphnet'] += 1
                        elif 'gnorm' in key:
                            layer_stats['gnorm'] += 1
                        elif 'global_conv' in key:
                            layer_stats['global_conv'] += 1
                        elif 'mlp' in key:
                            layer_stats['mlp'] += 1
                        elif 'ext_attention' in key:
                            layer_stats['ext_attention'] += 1
                        else:
                            layer_stats['other'] += 1
                    else:
                        shape_mismatch_keys.append(key)
                        skipped_keys.append(f"Shape mismatch: {key} (checkpoint: {value.shape}, model: {model_state[key].shape})")
                else:
                    # 检查是否是 input_projection 键
                    if 'input_projection' in key:
                        input_proj_keys.append(key)
                    skipped_keys.append(f"Key not in model: {key}")
            
            # 加载权重（已在上面完成）
            # model.load_state_dict(filtered_state, strict=False)  # 已在上面调用
            
            # 详细报告加载情况
            print_report('')
            print_report('='*60)
            print_report('【检查点权重加载报告】')
            print_report('='*60)
            print_report(f'检查点文件: {checkpoint_to_load}')
            print_report(f'检查点中的总键数: {len(checkpoint_state)}')
            print_report(f'成功加载的层数: {len(matched_keys)}')
            print_report('')
            print_report('按层类型统计:')
            if layer_stats['graphnet'] > 0:
                print_report(f'  ✓ GraphNet层: {layer_stats["graphnet"]} 层')
            if layer_stats['gnorm'] > 0:
                print_report(f'  ✓ GraphNorm层: {layer_stats["gnorm"]} 层')
            if layer_stats['global_conv'] > 0:
                print_report(f'  ✓ GlobalConv层: {layer_stats["global_conv"]} 层')
            if layer_stats['mlp'] > 0:
                print_report(f'  ✓ MLP输出层: {layer_stats["mlp"]} 层')
            if layer_stats['input_projection'] > 0:
                print_report(f'  ✓ InputProjection层: {layer_stats["input_projection"]} 层')
            if layer_stats['ext_attention'] > 0:
                print_report(f'  ✓ ExternalAttention层: {layer_stats["ext_attention"]} 层')
            if layer_stats['other'] > 0:
                print_report(f'  ✓ 其他层: {layer_stats["other"]} 层')
            print_report('')
            
            # 如果检查点中有 input_projection 但当前模型仍然没有，尝试再次创建
            if input_proj_keys:
                print_report('')
                print_report('='*60)
                print_report(f'⚠️ 检测到检查点中包含 {len(input_proj_keys)} 个 input_projection 层键，但当前模型中没有')
                print_report('='*60)
                print_report(f'   当前 attention_weight 配置: {current_attention_weight}')
                print_report(f'   尝试强制创建 input_projection 层以加载这些权重...')
                print_report('='*60)
                print_report('')
                
                try:
                    # 再次尝试创建 input_projection 层
                    def set_attention_weight_recursive(module, value):
                        if hasattr(module, 'attention_weight'):
                            module.attention_weight = value
                        for child in module.children():
                            set_attention_weight_recursive(child, value)
                    
                    # 临时设置 attention_weight = 0.5 以触发 input_projection 创建
                    set_attention_weight_recursive(model, 0.5)
                    
                    # 使用训练数据的一个样本来触发 forward（创建 input_projection 层）
                    temp_loader = get_dataloader_pairs_T(
                        df, 
                        df.index.tolist()[:1],  # 只用第一个样本
                        graphs_solv,
                        graphs_solu,
                        batch_size=1, 
                        shuffle=False, 
                        drop_last=False,
                        num_workers=0,
                        pin_memory=False
                    )
                    
                    model.eval()
                    with torch.no_grad():
                        for batch_data in temp_loader:
                            if len(batch_data) == 3:
                                batch_solvent, batch_solute, T = batch_data
                                batch_solvent = batch_solvent.to(device)
                                batch_solute = batch_solute.to(device)
                                T = T.to(device)
                                # 运行一次 forward 来触发 input_projection 的创建
                                _ = model(batch_solvent, batch_solute, T)
                                break
                    
                    # 恢复原始的 attention_weight
                    set_attention_weight_recursive(model, current_attention_weight)
                    
                    # 重新检查模型状态，尝试加载 input_projection 权重
                    model_state_after = model.state_dict()
                    input_proj_loaded = 0
                    for key in input_proj_keys:
                        if key in model_state_after:
                            if model_state_after[key].shape == checkpoint_state[key].shape:
                                model_state_after[key] = checkpoint_state[key]
                                input_proj_loaded += 1
                    
                    if input_proj_loaded > 0:
                        model.load_state_dict(model_state_after, strict=False)
                        print_report(f'✓ 成功创建并加载了 {input_proj_loaded} 个 input_projection 层')
                        print_report(f'✓ 已恢复 attention_weight={current_attention_weight}')
                        # 更新 filtered_state 和 matched_keys，移除已加载的 input_proj_keys
                        for key in input_proj_keys:
                            if key in model_state_after:
                                filtered_state[key] = checkpoint_state[key]
                                matched_keys.append(key)
                                if key in input_proj_keys:
                                    input_proj_keys.remove(key)
                                layer_stats['input_projection'] += 1
                    else:
                        print_report(f'⚠️ 无法创建或加载 input_projection 层')
                        print_report(f'   检查点中的 input_projection 层键:')
                        if len(input_proj_keys) <= 10:
                            for key in input_proj_keys:
                                print_report(f'     - {key}')
                        else:
                            for key in input_proj_keys[:10]:
                                print_report(f'     - {key}')
                            print_report(f'     ... 还有 {len(input_proj_keys) - 10} 个键未显示')
                    
                except Exception as e:
                    print_report(f'⚠️ 强制创建 input_projection 层时出错: {e}')
                    print_report(f'   检查点中的 input_projection 层键:')
                    if len(input_proj_keys) <= 10:
                        for key in input_proj_keys:
                            print_report(f'     - {key}')
                    else:
                        for key in input_proj_keys[:10]:
                            print_report(f'     - {key}')
                        print_report(f'     ... 还有 {len(input_proj_keys) - 10} 个键未显示')
                    import traceback
                    traceback.print_exc()
                
                print_report('='*60)
                print_report('')
            
            # 报告形状不匹配的键
            if shape_mismatch_keys:
                print_report(f'⚠️ 形状不匹配的层数: {len(shape_mismatch_keys)}')
                if len(shape_mismatch_keys) <= 5:
                    for key in shape_mismatch_keys:
                        print_report(f'    - {key}')
                else:
                    for key in shape_mismatch_keys[:5]:
                        print_report(f'    - {key}')
                    print_report(f'    ... 还有 {len(shape_mismatch_keys) - 5} 个键未显示')
                print_report('')
            
                print_report('='*60)
                print_report('')
            
            # 报告形状不匹配的键
            if shape_mismatch_keys:
                print_report(f'⚠️ 形状不匹配的层数: {len(shape_mismatch_keys)}')
                if len(shape_mismatch_keys) <= 5:
                    for key in shape_mismatch_keys:
                        print_report(f'    - {key}')
                else:
                    for key in shape_mismatch_keys[:5]:
                        print_report(f'    - {key}')
                    print_report(f'    ... 还有 {len(shape_mismatch_keys) - 5} 个键未显示')
                print_report('')
            
            print_report('='*60)
            print_report('')
            
            if skipped_keys:
                # 统计跳过的键的类型（排除 input_projection，因为已经单独处理）
                other_keys = [k for k in skipped_keys if 'input_projection' not in k]
                
                if other_keys:
                    print_report(f'  跳过 {len(other_keys)} 个其他不兼容的键（可能是由于架构变化）')
                    if len(other_keys) <= 5:  # 只显示前5个，避免输出过长
                        for key in other_keys[:5]:
                            print_report(f'    - {key}')
                    if len(other_keys) > 5:
                        print_report(f'    ... 还有 {len(other_keys) - 5} 个键未显示')
                
                # 修复：根据当前的 fine_tune_stage 重新设置参数的 requires_grad
                # 避免从检查点加载时恢复错误的冻结状态
                if fine_tune_stage == 'none' or fine_tune_stage is None:
                    # 不冻结任何层，所有参数都可训练
                    for param in model.parameters():
                        param.requires_grad = True
                    print_report('已重置所有参数为可训练状态（fine_tune_stage=none）')
                elif fine_tune_stage == 'two_stage':
                    # 先解冻所有参数，冻结逻辑会在训练循环中根据 freeze_epochs 处理
                    for param in model.parameters():
                        param.requires_grad = True
                    print_report('已重置所有参数为可训练状态（fine_tune_stage=two_stage，将在训练循环中处理冻结）')
                elif fine_tune_stage == 'output_only':
                    # 冻结共享层
                    frozen_count = 0
                    for name, param in model.named_parameters():
                        if any(layer in name for layer in ['graphnet1', 'graphnet2', 'gnorm1', 'gnorm2', 'global_conv1']):
                            param.requires_grad = False
                            frozen_count += 1
                        else:
                            param.requires_grad = True
                    print_report(f'已根据 fine_tune_stage=output_only 设置冻结状态（冻结 {frozen_count} 层）')
            
            # 如果是.pth文件，不包含优化器状态和训练历史，跳过这些加载
            if is_pth_file:
                print_report('⚠️ 从.pth文件加载：不包含优化器状态和训练历史，训练将从第0轮开始')
                optimizer_state_dict = None
                scheduler_state_dict = None
                start_epoch = 0
                # 不恢复训练历史，使用初始值
            else:
                # 保存优化器状态（将在optimizer创建后加载）
                optimizer_state_dict = checkpoint.get('optimizer_state_dict', None)
                if optimizer_state_dict is not None:
                    print_report('检查点中包含优化器状态，将在optimizer创建后加载')
                    # 提前检查优化器配置，以便在创建优化器时匹配
                    checkpoint_param_groups = len(optimizer_state_dict.get('param_groups', []))
                    print_report(f'检查点中的优化器参数组数量: {checkpoint_param_groups}')
                    # 如果检查点使用了分层学习率，标记以便后续使用
                    if checkpoint_param_groups > 1:
                        checkpoint_uses_layered_lr = True
                        print_report(f'✓ 已标记：检查点使用分层学习率（{checkpoint_param_groups}个参数组），将在创建优化器时匹配此配置')
            
            # 保存调度器状态（将在scheduler创建后加载）
            scheduler_state_dict = checkpoint.get('scheduler_state_dict', None)
            if scheduler_state_dict is not None:
                print_report('检查点中包含调度器状态，将在scheduler创建后加载')
            
            # 加载最佳模型
            if 'best_model_state_dict' in checkpoint:
                best_model = checkpoint['best_model_state_dict']
                print_report('已加载最佳模型状态')
                # 重要：如果从检查点恢复，优先使用最佳模型状态而不是最后一轮的状态
                # 这样可以避免从较差的模型状态继续训练
                # 但需要用户明确指定是否使用最佳模型（通过hyperparameters中的use_best_model_on_resume）
                use_best_model_on_resume = hyperparameters.get('use_best_model_on_resume', False)
                if use_best_model_on_resume:
                    print_report('⚠️ 使用最佳模型状态恢复训练（而不是最后一轮状态）')
                    # 使用最佳模型状态覆盖当前模型状态
                    model.load_state_dict(best_model, strict=False)
                    print_report('已用最佳模型状态更新当前模型')
            
            # 恢复训练历史
            if 'epoch' in checkpoint:
                checkpoint_epoch = checkpoint['epoch']
                # 如果指定了start_epoch_override，使用指定的值；否则使用checkpoint中的epoch+1
                if start_epoch_override is not None:
                    start_epoch = start_epoch_override
                    print_report(f'检查点中的epoch: {checkpoint_epoch}，但将强制从第 {start_epoch} 轮开始训练')
                else:
                    start_epoch = checkpoint_epoch + 1
                    print_report(f'从第 {start_epoch} 轮继续训练（检查点中的epoch: {checkpoint_epoch}）')
            
            # 恢复训练历史
            if 'mae_train' in checkpoint:
                mae_train = checkpoint['mae_train']
                print_report(f'已恢复训练MAE历史 ({len(mae_train)} 轮)')
            
            if 'r2_train' in checkpoint:
                r2_train = checkpoint['r2_train']
                print_report(f'已恢复训练R²历史 ({len(r2_train)} 轮)')
            
            # 🔑 恢复验证集历史
            if 'mae_valid' in checkpoint:
                mae_valid = checkpoint['mae_valid']
                print_report(f'已恢复验证集MAE历史 ({len(mae_valid)} 轮)')
            
            if 'r2_valid' in checkpoint:
                r2_valid = checkpoint['r2_valid']
                print_report(f'已恢复验证集R²历史 ({len(r2_valid)} 轮)')
            
            # 恢复best_MAE
            if 'best_MAE' in checkpoint:
                # 如果提供了验证集且checkpoint中有验证集历史，使用验证集最佳值
                if val_loader is not None and len(mae_valid) > 0:
                    best_MAE = min(mae_valid)
                    print_report(f'从验证集历史恢复最佳MAE: {best_MAE:.6f}')
                # 如果提供了验证集但checkpoint中没有验证集历史，重新计算
                elif val_loader is not None:
                    print_report(f'从checkpoint恢复：检测到验证集，但无验证集历史，将重新寻找验证集最佳MAE')
                    best_MAE = np.inf
                # 如果没有提供验证集，使用checkpoint中的值
                else:
                    best_MAE = checkpoint['best_MAE']
                    print_report(f'当前最佳MAE: {best_MAE:.6f}')
            
            # 记录恢复时的最佳MAE（用于后续判断是否更新）
            resumed_best_mae = best_MAE
            
            # 恢复早停器状态
            if 'early_stopping_state' in checkpoint:
                es_state = checkpoint['early_stopping_state']
                early_stopper.best = es_state.get('best')
                early_stopper.num_bad = es_state.get('num_bad', 0)
                early_stopper.should_stop = es_state.get('should_stop', False)
                
                # 如果patience被设置为非常大的值（表示禁用早停），重置should_stop
                if early_stop_resume_patience >= 99999:  # 阈值：99999或更大表示禁用早停
                    if early_stopper.should_stop:
                        print_report(f'检测到早停已禁用（patience={early_stop_resume_patience}），重置早停器状态')
                        early_stopper.should_stop = False
                        early_stopper.num_bad = 0  # 重置计数，允许继续训练
                
                print_report(f'已恢复早停器状态 (best: {early_stopper.best:.6f}, num_bad: {early_stopper.num_bad}, should_stop: {early_stopper.should_stop})')
            
            # 🔑 从检查点恢复时，先禁用早停，待验证集最佳MAE更新后再恢复
            if resumed_from_checkpoint:
                early_stop_active = False
                early_stopper.should_stop = False
                early_stopper.num_bad = 0
                print_report(f'🔑 从检查点恢复：已禁用早停，等待验证集最佳MAE更新后再恢复')
                if resumed_best_mae is not None and resumed_best_mae != np.inf:
                    print_report(f'   当前最佳MAE: {resumed_best_mae:.6f}，将在找到更好的模型后恢复早停')
                else:
                    print_report(f'   将等待验证集最佳MAE首次更新后恢复早停')
            
            print_report('检查点加载完成！')
        except Exception as e:
            print_report(f'警告: 加载检查点失败: {e}')
            print_report('将从第0轮开始训练')
            start_epoch = 0
    else:
        if resume_checkpoint:
            print_report(f'警告: 指定的检查点文件不存在: {resume_checkpoint}')
            print_report('将从第0轮开始训练')
        else:
            print_report('未找到检查点文件，将从第0轮开始训练')

    # 根据 fine_tune_epochs 调整目标训练轮数（start_epoch 已确定）
    target_total_epochs = n_epochs
    if fine_tune_epochs is not None:
        # 确保至少还能训练1轮
        target_total_epochs = max(start_epoch + fine_tune_epochs, start_epoch + 1)
        print_report(f'Fine-tune模式：将在当前起点（第 {start_epoch} 轮）基础上追加 {fine_tune_epochs} 轮，总训练轮次调整为 {target_total_epochs}')
    total_epochs = target_total_epochs
    
    # 在加载预训练权重后，创建优化器（如果还没有创建）
    # 从超参数中获取weight_decay，默认1e-2
    weight_decay = hyperparameters.get('weight_decay', 1e-2)
    
    if 'optimizer' not in locals():
        # 如果使用冻结策略，只优化可训练参数
        if freeze_shared_layers:
            trainable_params = [p for p in model.parameters() if p.requires_grad]
            optimizer = torch.optim.AdamW(trainable_params, lr=original_lr, weight_decay=weight_decay)
            print_report(f'优化器: 只优化可训练参数')
        else:
            # 全量微调：使用不同学习率（共享层用更小的学习率，注意力层用更大的学习率）
            if use_pretrained and not has_checkpoint and fine_tune_stage == 'full':
                # 为共享层、注意力层和输出层设置不同的学习率
                # 优化策略：根据attention_weight动态调整学习率
                attention_weight = hyperparameters.get('attention_weight', 1.0)
                
                # 根据attention_weight动态调整共享层学习率
                if attention_weight <= 0.1:
                    shared_lr_factor = max(fine_tune_lr_factor, 0.08)  # 注意力影响小，可以稍大的学习率
                elif attention_weight >= 0.9:
                    shared_lr_factor = min(fine_tune_lr_factor, 0.02)  # 注意力影响大，需要很小的学习率
                else:
                    # 线性插值：0.1->0.08, 0.9->0.02
                    base_factor = 0.08 - 0.075 * (attention_weight - 0.1) / 0.8
                    shared_lr_factor = min(fine_tune_lr_factor, base_factor)
                
                # 根据attention_weight动态调整注意力层学习率
                if attention_weight <= 0.1:
                    attention_lr_factor = 2.5
                elif attention_weight >= 0.9:
                    attention_lr_factor = 1.5
                else:
                    attention_lr_factor = 2.5 - 1.25 * (attention_weight - 0.1) / 0.8
                
                shared_lr = original_lr * shared_lr_factor
                attention_lr = original_lr * attention_lr_factor
                shared_params = []
                attention_params = []
                output_params = []
                for name, param in model.named_parameters():
                    if any(layer in name for layer in ['graphnet1', 'graphnet2', 'gnorm1', 'gnorm2', 'global_conv1']):
                        shared_params.append(param)
                    elif 'ext_attention' in name and ('Mk' in name or 'Mv' in name):
                        attention_params.append(param)
                    else:
                        output_params.append(param)
                
                # 构建参数组
                param_groups = [
                    {'params': shared_params, 'lr': shared_lr, 'weight_decay': weight_decay},  # 共享层用更小的学习率
                    {'params': output_params, 'lr': original_lr, 'weight_decay': weight_decay}  # 输出层用正常学习率
                ]
                if attention_params:
                        param_groups.insert(1, {'params': attention_params, 'lr': attention_lr, 'weight_decay': weight_decay})  # 注意力层用更大的学习率
                
                optimizer = torch.optim.AdamW(param_groups)
                if attention_params:
                    print_report(f'优化器: 动态分层学习率（attention_weight={attention_weight:.2f}, fine_tune_stage=full）- 共享层 lr={shared_lr:.6f} ({shared_lr_factor:.3f}x), 注意力层 lr={attention_lr:.6f} ({attention_lr_factor:.3f}x), 输出层 lr={original_lr:.6f}')
                else:
                    print_report(f'优化器: 动态分层学习率（attention_weight={attention_weight:.2f}, fine_tune_stage=full）- 共享层 lr={shared_lr:.6f} ({shared_lr_factor:.3f}x), 输出层 lr={original_lr:.6f}')
            else:
                # 即使fine_tune_stage == 'none'，如果使用预训练权重，也应该使用分层学习率保护预训练权重
                # 优化策略：根据attention_weight动态调整学习率
                # attention_weight越小 → 注意力影响小 → 可以更信任预训练权重 → 共享层学习率可以稍大
                # attention_weight越大 → 注意力影响大 → 需要更小心保护预训练权重 → 共享层学习率应该更小
                
                # 检查检查点中的优化器配置，如果检查点使用了分层学习率，我们也应该使用分层学习率
                # 使用全局变量 checkpoint_uses_layered_lr（在加载检查点时已设置）
                if has_checkpoint and checkpoint_uses_layered_lr:
                    print_report(f'✓ 检测到检查点使用了分层学习率，将匹配此配置以正确加载优化器状态')
                
                # 如果使用预训练权重，应该使用分层学习率保护预训练权重
                # 条件：use_pretrained=True 且 (没有checkpoint 或 checkpoint使用了分层学习率 或 是从.pth文件加载)
                # 从.pth文件加载时，虽然无法检测checkpoint的优化器配置，但如果使用了预训练权重，仍应使用分层学习率
                # 但是，如果明确设置了 force_layered_lr=False，则强制不使用分层学习率
                if force_layered_lr is False:
                    # 明确禁用分层学习率
                    should_use_layered_lr = False
                    print_report('⚙️  force_layered_lr=False，强制禁用分层学习率，使用统一学习率')
                else:
                    should_use_layered_lr = force_layered_lr or (use_pretrained and (not has_checkpoint or checkpoint_uses_layered_lr or is_pth_file))
                
                if force_layered_lr and has_checkpoint and not (checkpoint_uses_layered_lr or is_pth_file):
                    print_report('⚙️  已启用 force_layered_lr，忽略 checkpoint 中的学习率策略，强制使用分层学习率')
                
                if should_use_layered_lr:
                    # 获取attention_weight（默认1.0）
                    attention_weight = hyperparameters.get('attention_weight', 1.0)
                    
                    # 根据attention_weight动态调整共享层学习率
                    # attention_weight从0.1到0.9，共享层学习率从0.15线性降低到0.05（提高学习率）
                    # 公式：shared_lr_factor = 0.15 - 0.125 * (attention_weight - 0.1) / 0.8
                    # 当attention_weight=0.1时，shared_lr_factor=0.15
                    # 当attention_weight=0.9时，shared_lr_factor=0.05
                    if attention_weight <= 0.1:
                        shared_lr_factor = 0.15  # 注意力影响很小，可以使用较大的学习率
                    elif attention_weight >= 0.9:
                        shared_lr_factor = 0.05  # 注意力影响很大，使用较小的学习率（但仍比之前高）
                    else:
                        # 线性插值：0.1->0.15, 0.9->0.05
                        shared_lr_factor = 0.15 - 0.125 * (attention_weight - 0.1) / 0.8
                    
                    # 根据attention_weight动态调整注意力层学习率
                    # attention_weight越小，注意力层需要更快学习（因为影响小，需要快速适应）
                    # attention_weight越大，注意力层可以稍慢学习（因为影响大，已经占主导）
                    if attention_weight <= 0.1:
                        attention_lr_factor = 2.5  # 注意力影响小，需要快速学习适应
                    elif attention_weight >= 0.9:
                        attention_lr_factor = 1.5  # 注意力影响大，已经占主导，可以稍慢
                    else:
                        # 线性插值：0.1->2.5, 0.9->1.5
                        attention_lr_factor = 2.5 - 1.25 * (attention_weight - 0.1) / 0.8
                    
                    # 根据attention_weight动态调整输出层学习率
                    # attention_weight越小，输出层可以稍大学习率（预训练权重更可靠）
                    # attention_weight越大，输出层需要更小学习率（保护预训练权重）
                    # 提高学习率：从0.5-0.2改为0.8-0.4
                    if attention_weight <= 0.1:
                        output_lr_factor = 0.8  # 注意力影响小，预训练权重更可靠，可以使用较大学习率
                    elif attention_weight >= 0.9:
                        output_lr_factor = 0.4  # 注意力影响大，需要保护，但仍比之前高
                    else:
                        # 线性插值：0.1->0.8, 0.9->0.4
                        output_lr_factor = 0.8 - 0.5 * (attention_weight - 0.1) / 0.8
                    
                    shared_lr = original_lr * shared_lr_factor
                    attention_lr = original_lr * attention_lr_factor
                    output_lr = original_lr * output_lr_factor
                    
                    shared_params = []
                    attention_params = []
                    output_params = []
                    
                    for name, param in model.named_parameters():
                        if any(layer in name for layer in ['graphnet1', 'graphnet2', 'gnorm1', 'gnorm2', 'global_conv1']):
                            shared_params.append(param)
                        elif 'ext_attention' in name and ('Mk' in name or 'Mv' in name):
                            attention_params.append(param)
                        else:
                            output_params.append(param)
                    
                    # 构建参数组
                    param_groups = [
                        {'params': shared_params, 'lr': shared_lr, 'weight_decay': weight_decay},  # 共享层：保护预训练权重
                        {'params': output_params, 'lr': output_lr, 'weight_decay': weight_decay}  # 输出层：正常学习率
                    ]
                    if attention_params:
                        param_groups.insert(1, {'params': attention_params, 'lr': attention_lr, 'weight_decay': weight_decay})  # 注意力层：快速学习
                    
                    optimizer = torch.optim.AdamW(param_groups)
                    if attention_params:
                        print_report(f'优化器: 动态分层学习率（attention_weight={attention_weight:.2f}）- 共享层 lr={shared_lr:.6f} ({shared_lr_factor:.3f}x), 注意力层 lr={attention_lr:.6f} ({attention_lr_factor:.3f}x), 输出层 lr={output_lr:.6f} ({output_lr_factor:.3f}x)')
                    else:
                        print_report(f'优化器: 动态分层学习率（attention_weight={attention_weight:.2f}）- 共享层 lr={shared_lr:.6f} ({shared_lr_factor:.3f}x), 输出层 lr={output_lr:.6f} ({output_lr_factor:.3f}x)')
                else:
                    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        
        # 重新创建调度器（支持周期策略或自适应策略）
        scheduler = _create_scheduler(optimizer)
        
        # 如果有检查点的优化器状态，现在加载它
        if optimizer_state_dict is not None:
            try:
                # 检查参数组数量是否匹配
                checkpoint_param_groups = len(optimizer_state_dict.get('param_groups', []))
                current_param_groups = len(optimizer.param_groups)
                
                if checkpoint_param_groups != current_param_groups:
                    print_report(f'警告: 优化器参数组数量不匹配 (检查点: {checkpoint_param_groups}, 当前: {current_param_groups})')
                    print_report('  这通常发生在使用了不同的学习率策略时（例如，分层学习率 vs 单一学习率）')
                    print_report('  将跳过优化器状态加载，使用新初始化的优化器')
                else:
                    # 参数组数量匹配，尝试加载
                    optimizer.load_state_dict(optimizer_state_dict)
                    print_report('已加载优化器状态')
            except Exception as e:
                print_report(f'警告: 加载优化器状态失败: {e}')
                print_report('  可能原因: 参数组配置不匹配（例如，使用了不同的分层学习率策略）')
                print_report('  将使用新初始化的优化器')
        
        # 如果有检查点的调度器状态，现在加载它
        if scheduler_state_dict is not None:
            try:
                scheduler.load_state_dict(scheduler_state_dict)
                print_report('已加载调度器状态')
            except Exception as e:
                print_report(f'警告: 加载调度器状态失败: {e}')
                print_report('将使用新初始化的调度器')

    # 如果模型被重新创建（由于Triton错误），需要重新创建优化器和scaler
    if 'need_recreate_optimizer' in locals() and need_recreate_optimizer:
        print_report('')
        print_report('='*60)
        print_report('【重新创建优化器和Scaler】')
        print_report('='*60)
        print_report('由于模型已重新创建，需要重新创建优化器和scaler以匹配新的模型参数')
        
        # 重新创建优化器（使用与之前相同的逻辑）
        if freeze_shared_layers:
            trainable_params = [p for p in model.parameters() if p.requires_grad]
            optimizer = torch.optim.AdamW(trainable_params, lr=original_lr, weight_decay=weight_decay)
            print_report('✓ 优化器已重新创建（只优化可训练参数）')
        else:
            optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
            print_report('✓ 优化器已重新创建（优化所有参数）')
        
        # 重新创建调度器
        scheduler = _create_scheduler(optimizer)
        print_report('✓ 调度器已重新创建')
        
        # 重新创建 GradScaler（如果使用混合精度）
        if use_mixed_precision and torch.cuda.is_available():
            scaler = GradScaler('cuda')
            print_report('✓ GradScaler已重新创建（混合精度训练）')
        
        print_report('='*60)
        print_report('')
    
    # 训练开始提示
    print_report('')
    print_report('='*60)
    # 训练配置信息输出已禁用（用户要求不再输出）
    # print_report('【开始训练】')
    # print_report('='*60)
    # print_report(f'总训练轮数: {total_epochs}')
    # print_report(f'起始轮次: {start_epoch}')
    # print_report(f'批次大小: {batch_size}')
    # print_report(f'训练集大小: {len(df)}')
    # if val_loader is not None:
    #     print_report(f'验证集大小: {len(val_df)}')
    # print_report(f'数据加载进程数: {num_workers} (单进程模式，已禁用多进程加速)')
    # print_report(f'数据预取因子: {prefetch_factor}')
    # print_report(f'内存固定: {"启用" if pin_memory else "禁用"} (已禁用)')
    # print_report(f'持久化工作进程: {"启用" if persistent_workers else "禁用"} (已禁用)')
    # print_report(f'混合精度训练: {"启用" if scaler is not None else "禁用"} (已禁用)')
    # print_report(f'模型编译: {"启用" if use_torch_compile else "禁用"} (已禁用)')
    # print_report(f'模型预热: 禁用 (已取消所有训练加速方法)')
    # 
    # # 计算预期的batch数量和性能指标
    # try:
    #     expected_batches = len(train_loader)
    #     samples_per_epoch = batch_size * expected_batches
    #     print_report(f'每个epoch的batch数: {expected_batches}')
    #     print_report(f'每个epoch的样本数: {samples_per_epoch}')
    #     if expected_batches > 0:
    #         # 估算每个batch的理想时间（假设GPU利用率100%）
    #         # 对于图神经网络，每个batch通常需要0.1-0.3秒（取决于模型复杂度和batch大小）
    #         estimated_batch_time = 0.15  # 保守估计
    #         estimated_epoch_time = expected_batches * estimated_batch_time
    #         print_report(f'预估每个epoch时间: {estimated_epoch_time:.1f}秒（理想情况，实际可能更慢）')
    # except Exception:
    #     pass
    # 
    # print_report('='*60)
    # print_report('')
    
    # 模型预热：已禁用（取消所有训练加速方法）
    # 如果 start_epoch == 0:  # 只在从头训练时预热
    #     print_report('【模型预热】预热模型和数据加载器（加速首次迭代）...')
    if False:  # 禁用模型预热
        print_report('【模型预热】预热模型和数据加载器（加速首次迭代）...')
        try:
            model.eval()
            warmup_start = time.time()
            with torch.no_grad():
                # 获取第一个batch进行预热
                warmup_iter = iter(train_loader)
                warmup_batch = next(warmup_iter)
                
                if len(warmup_batch) == 3:
                    warmup_solvent, warmup_solute, warmup_T = warmup_batch
                    warmup_T = warmup_T.to(device, non_blocking=True)
                    warmup_solvent = warmup_solvent.to(device, non_blocking=True)
                    warmup_solute = warmup_solute.to(device, non_blocking=True)
                    # 运行一次forward预热
                    _ = model(warmup_solvent, warmup_solute, warmup_T)
                elif len(warmup_batch) == 2:
                    warmup_solvent, warmup_solute = warmup_batch
                    warmup_solvent = warmup_solvent.to(device, non_blocking=True)
                    warmup_solute = warmup_solute.to(device, non_blocking=True)
                    # 运行一次forward预热
                    _ = model(warmup_solvent, warmup_solute)
            
            warmup_time = time.time() - warmup_start
            print_report(f'✓ 模型预热完成（耗时: {warmup_time:.2f}秒）')
            print_report('   已预热：CUDA初始化、模型编译、数据加载器、GPU内存分配')
            print_report('')
        except Exception as e:
            error_msg = str(e)
            error_type = str(type(e).__name__)
            
            # 检查是否是 Triton 相关错误
            is_triton_error = (
                'triton' in error_msg.lower() or 
                'TritonMissing' in error_type or
                'Cannot find a working triton' in error_msg
            )
            
            if is_triton_error and model_compiled:
                print_report('')
                print_report('='*60)
                print_report('⚠️ 检测到 Triton 相关错误')
                print_report('='*60)
                print_report('模型编译需要 Triton，但系统未安装或无法使用 Triton')
                print_report('将禁用模型编译，使用未编译的模型继续训练')
                print_report('='*60)
                print_report('')
                
                # 重新创建模型（不使用编译）
                print_report('重新创建模型（不使用编译）...')
                model = GHGEAT(v_in, e_in, u_in, hidden_dim, attention_weight=attention_weight)
                model = model.to(device)
                
                # 重新加载权重（如果有）
                if has_checkpoint:
                    # 重新加载检查点权重
                    checkpoint = torch.load(checkpoint_to_load, map_location=torch.device(available_device), weights_only=False)
                    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                        checkpoint_state = checkpoint['model_state_dict']
                    else:
                        checkpoint_state = checkpoint
                    model.load_state_dict(checkpoint_state, strict=False)
                    print_report('✓ 已重新加载检查点权重')
                elif use_pretrained and os.path.exists(pretrained_path):
                    # 重新加载预训练权重
                    checkpoint = torch.load(pretrained_path, map_location=torch.device(available_device), weights_only=False)
                    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                        pretrained_dict = checkpoint['model_state_dict']
                    else:
                        pretrained_dict = checkpoint
                    model_dict = model.state_dict()
                    # 简化的权重加载（复用之前的逻辑）
                    for key in pretrained_dict.keys():
                        model_key = None
                        if 'shared_layer.' in key:
                            model_key = key.replace('shared_layer.', '')
                        elif 'task_A.' in key:
                            model_key = key.replace('task_A.', '')
                        elif 'task_B.' in key:
                            model_key = key.replace('task_B.', '')
                        else:
                            model_key = key
                        
                        if model_key and model_key in model_dict:
                            if model_dict[model_key].shape == pretrained_dict[key].shape:
                                model_dict[model_key] = pretrained_dict[key]
                    model.load_state_dict(model_dict, strict=False)
                    print_report('✓ 已重新加载预训练权重')
                
                model_compiled = False
                print_report('✓ 模型已重新创建（未编译）')
                
                # 标记需要重新创建优化器（因为模型参数已改变）
                need_recreate_optimizer = True
                print_report('   注意：将在预热后重新创建优化器（因为模型已重新创建）')
                print_report('')
                
                # 重新尝试预热
                try:
                    model.eval()
                    warmup_start = time.time()
                    with torch.no_grad():
                        if len(warmup_batch) == 3:
                            _ = model(warmup_solvent, warmup_solute, warmup_T)
                        elif len(warmup_batch) == 2:
                            _ = model(warmup_solvent, warmup_solute)
                    warmup_time = time.time() - warmup_start
                    print_report(f'✓ 模型预热完成（耗时: {warmup_time:.2f}秒）')
                    print_report('')
                except Exception as e2:
                    print_report(f'⚠️ 模型预热失败: {e2}')
                    print_report('   将跳过预热，首次迭代可能较慢')
                    print_report('')
            else:
                print_report(f'⚠️ 模型预热失败: {e}')
                print_report('   将跳过预热，首次迭代可能较慢')
                print_report('')

    # 🔑 在训练开始前验证输入数据（仅在第一个epoch和start_epoch==0时检查）
    if start_epoch == 0:
        print_report('【数据验证】检查训练数据质量...')
        try:
            # 检查前几个样本
            sample_count = 0
            max_samples_to_check = min(10, len(df))
            for idx in range(max_samples_to_check):
                try:
                    sample_solv = graphs_solv[idx]
                    sample_solu = graphs_solu[idx]
                    # 检查图数据中的特征
                    if hasattr(sample_solv, 'x') and sample_solv.x is not None:
                        if torch.any(~torch.isfinite(sample_solv.x)):
                            nan_count = torch.sum(~torch.isfinite(sample_solv.x)).item()
                            print_report(f'  ⚠️ 样本 {idx} 溶剂节点特征包含NaN/Inf: {nan_count} 个值')
                    if hasattr(sample_solv, 'y') and sample_solv.y is not None:
                        if torch.any(~torch.isfinite(sample_solv.y)):
                            print_report(f'  ⚠️ 样本 {idx} 溶剂标签包含NaN/Inf')
                    if hasattr(sample_solu, 'x') and sample_solu.x is not None:
                        if torch.any(~torch.isfinite(sample_solu.x)):
                            nan_count = torch.sum(~torch.isfinite(sample_solu.x)).item()
                            print_report(f'  ⚠️ 样本 {idx} 溶质节点特征包含NaN/Inf: {nan_count} 个值')
                    sample_count += 1
                except Exception as e:
                    print_report(f'  ⚠️ 检查样本 {idx} 时出错: {e}')
            
            if sample_count == max_samples_to_check:
                print_report(f'✓ 已检查 {sample_count} 个样本，未发现明显的NaN/Inf（在节点特征中）')
        except Exception as e:
            print_report(f'  ⚠️ 数据验证失败: {e}')
        print_report('')
    
    for epoch in range(start_epoch, total_epochs):
        epoch_start_time = time.time()
        relative_epoch = epoch - start_epoch + 1
        stats = OrderedDict()
        relative_epoch = epoch - start_epoch + 1
        
        # 第一个epoch开始时输出提示
        if epoch == start_epoch:
            print_report(f'开始第 {epoch+1} 轮训练...')
        
        # 两阶段微调：在指定epoch解冻共享层
        if use_pretrained and not has_checkpoint and fine_tune_stage == 'two_stage' and freeze_shared_layers:
            if epoch == freeze_epochs:
                # 解冻所有层，切换到全量微调
                # print_report(f'\n{"="*60}')
                # print_report(f'【阶段切换】第 {epoch+1} 轮：解冻所有层，开始全量微调')
                # print_report(f'{"="*60}')
                
                # 解冻所有参数
                for name, param in model.named_parameters():
                    param.requires_grad = True
                
                # 重新创建优化器，使用更小的学习率进行全量微调
                fine_tune_lr = original_lr * fine_tune_lr_factor
                reduction_factor_str = f"{1/fine_tune_lr_factor:.1f}倍" if fine_tune_lr_factor < 1 else "不变"
                optimizer = torch.optim.AdamW(model.parameters(), lr=fine_tune_lr, weight_decay=weight_decay)
                
                # 重新创建 GradScaler（如果使用混合精度），确保与新的 optimizer 匹配
                if use_mixed_precision and torch.cuda.is_available():
                    scaler = GradScaler('cuda')
                    print_report('✓ GradScaler已重新创建（匹配新的优化器）')
                
                # 重新创建调度器（支持配置的周期策略）
                remaining_epochs = total_epochs - epoch
                scheduler = _create_scheduler(optimizer)
                # print_report(f'全量微调学习率: {fine_tune_lr:.6f} (降低{reduction_factor_str})')
                # print_report(f'剩余训练轮数: {remaining_epochs}')
                # print_report(f'阶段2 Warmup: {warmup_remaining} epochs ({warmup_remaining/remaining_epochs*100:.1f}%)')
                # print_report(f'{"="*60}\n')
                
                freeze_shared_layers = False  # 标记已解冻

        # Train（使用优化的训练函数，支持混合精度）
        # 验证设备使用情况（仅在第一个epoch检查，已禁用输出）
        # if epoch == 0:
        #     # 验证设备（已禁用输出）
        #     # sample_param = next(model.parameters())
        #     # actual_device = sample_param.device
        #     # if torch.cuda.is_available() and actual_device.type != 'cuda':
        #     #     print_report(f'⚠️ 警告: 模型参数不在CUDA上！实际设备: {actual_device}')
        #     # elif torch.cuda.is_available():
        #     #     print_report(f'✓ 训练使用CUDA设备: {actual_device}')
        #     # else:
        #     #     print_report(f'⚠️ 警告: 使用CPU训练，速度会很慢')
        #     pass
        
        train_stats = train(model, device, train_loader, optimizer, task_type, stats, scaler=scaler)
        stats.update(train_stats)
        
        # 如果 train 函数重新创建了 scaler，更新本地的 scaler 引用
        if '_updated_scaler' in train_stats:
            scaler = train_stats['_updated_scaler']
            del train_stats['_updated_scaler']  # 清理临时键
        
        # 验证频率优化：可以设置每N个epoch验证一次（默认每个epoch都验证）
        eval_interval = hyperparameters.get('eval_interval', 1)  # 1表示每个epoch都验证
        should_eval = (epoch % eval_interval == 0) or (epoch == total_epochs - 1)  # 最后一个epoch总是验证
        
        if should_eval:
            # Evaluation on training set
            stats.update(eval(model, device, train_loader, MAE, stats, 'Train', task_type))
            stats.update(eval(model, device, train_loader, R2, stats, 'Train', task_type))
        
            # Evaluation on validation set (if provided)
            eval_loader = val_loader if val_loader is not None else train_loader
            eval_split_label = 'Valid' if val_loader is not None else 'Train'
            stats.update(eval(model, device, eval_loader, MAE, stats, eval_split_label, task_type))
            stats.update(eval(model, device, eval_loader, R2, stats, eval_split_label, task_type))
        else:
            # 即使不进行完整验证，也需要计算训练集指标用于监控
            stats.update(eval(model, device, train_loader, MAE, stats, 'Train', task_type))
            stats.update(eval(model, device, train_loader, R2, stats, 'Train', task_type))
            # 对于验证集，使用上一次的值（如果存在）
            if val_loader is not None and len(mae_valid) > 0:
                # 使用上一次的验证集指标
                stats[f'MAE_Valid'] = mae_valid[-1]
                stats[f'R2_Valid'] = r2_valid[-1] if len(r2_valid) > 0 else 0.0
            eval_loader = val_loader if val_loader is not None else train_loader
            eval_split_label = 'Valid' if val_loader is not None else 'Train'
        
        # 🔑 NaN/Inf检测：检查训练指标是否有效
        train_mae = stats.get('MAE_Train', np.inf)
        train_r2 = stats.get('R2_Train', -np.inf)
        valid_mae = stats.get(f'MAE_{eval_split_label}', np.inf)
        valid_r2 = stats.get(f'R2_{eval_split_label}', -np.inf)
        
        # 检测异常值
        has_nan_or_inf = (
            not np.isfinite(train_mae) or 
            not np.isfinite(train_r2) or 
            not np.isfinite(valid_mae) or 
            not np.isfinite(valid_r2)
        )
        
        if has_nan_or_inf:
            print_report('')
            print_report('='*60)
            print_report(f'❌ 训练失败：检测到NaN或Inf值 (Epoch {epoch+1})')
            print_report(f'   Train MAE: {train_mae:.6f}, R²: {train_r2:.6f}')
            print_report(f'   {eval_split_label} MAE: {valid_mae:.6f}, R²: {valid_r2:.6f}')
            
            # 🔑 添加详细的诊断信息
            print_report('')
            print_report('【详细诊断信息】')
            print_report(f'   训练MAE是否有限: {np.isfinite(train_mae)}')
            print_report(f'   训练R²是否有限: {np.isfinite(train_r2)}')
            print_report(f'   验证MAE是否有限: {np.isfinite(valid_mae)}')
            print_report(f'   验证R²是否有限: {np.isfinite(valid_r2)}')
            
            # 检查模型参数的当前范围
            try:
                max_weight = 0.0
                max_weight_name = ''
                for name, param in model.named_parameters():
                    if param.requires_grad:
                        param_max = param.data.abs().max().item()
                        if param_max > max_weight:
                            max_weight = param_max
                            max_weight_name = name
                print_report(f'   最大权重绝对值: {max_weight:.2f} (在 {max_weight_name})')
                if max_weight > 1000:
                    print_report(f'   ⚠️ 检测到异常大的权重值，可能导致数值不稳定')
            except Exception as e:
                print_report(f'   无法检查模型参数: {e}')
            
            print_report('')
            print_report('   可能原因：')
            print_report('   1. 模型输出包含NaN/Inf（已在模型内部和训练循环中处理）')
            print_report('   2. R²计算时遇到问题（真实值方差为0或数据问题）')
            print_report('   3. 模型权重过大导致输出溢出')
            print_report('   4. 输入数据本身包含异常值')
            print_report('')
            print_report('   建议：')
            print_report('   1. 检查输入数据是否包含NaN或异常值')
            print_report('   2. 检查模型权重初始化')
            print_report('   3. 尝试更小的学习率（虽然已尝试但可能仍需更小）')
            print_report('   4. 检查数据预处理和归一化')
            print_report('='*60)
            print_report('')
            # 对于超参搜索：标记为TrialPruned以跳过此组合；否则回退为ValueError
            try:
                import optuna
                raise optuna.exceptions.TrialPruned(f'训练在epoch {epoch+1}出现NaN/Inf，学习率可能过大或模型不稳定')
            except ImportError:
                raise ValueError(f'训练在epoch {epoch+1}出现NaN/Inf，学习率可能过大或模型不稳定')
        
        # 用于早停和调度的MAE（优先使用验证集）
        # 如果当前epoch没有验证，使用上一次的验证集MAE
        if should_eval:
            mae_for_scheduler = stats[f'MAE_{eval_split_label}']
        else:
            # 使用上一次的验证集MAE（如果存在），否则使用训练集MAE
            if val_loader is not None and len(mae_valid) > 0:
                mae_for_scheduler = mae_valid[-1]
            else:
                mae_for_scheduler = stats['MAE_Train']
        
        # Scheduler（根据配置可能是周期调度器或ReduceLROnPlateau）
        if scheduler_type == 'plateau':
            scheduler.step(mae_for_scheduler)
        else:
            scheduler.step()
        # Early stopping（使用验证集MAE，如果提供）
        if early_stop_active:
            early_stopper.step(mae_for_scheduler)
            if early_stopper.should_stop:
                # 使用完整的验证集历史来报告真实最佳轮次，避免与早停计数不一致
                if val_loader is not None and len(mae_valid) > 0:
                    best_valid_mae = min(mae_valid)
                    best_valid_epoch = mae_valid.index(best_valid_mae) + 1
                else:
                    best_valid_mae = early_stopper.best
                    best_valid_epoch = epoch + 1 - early_stopper.num_bad
                print_report(f'早停触发：{eval_split_label} MAE在{early_stopper.num_bad}轮内未显著改善（patience={early_stop_resume_patience}, min_delta={early_stop_min_delta}）')
                print_report(f'全程最佳{eval_split_label} MAE: {best_valid_mae:.6f} (epoch {best_valid_epoch})')
                break
        # Save info
        mae_train.append(stats['MAE_Train'])
        r2_train.append(stats['R2_Train'])
        if val_loader is not None:
            if should_eval:
                mae_valid.append(stats['MAE_Valid'])
                r2_valid.append(stats['R2_Valid'])
            else:
                # 如果当前epoch没有验证，使用上一次的值
                if len(mae_valid) > 0:
                    mae_valid.append(mae_valid[-1])
                    r2_valid.append(r2_valid[-1])
                else:
                    # 如果没有历史值，使用训练集指标（临时）
                    mae_valid.append(stats['MAE_Train'])
                    r2_valid.append(stats['R2_Train'])
        
        # 从.pth文件加载时，当MAE达到阈值后自动降低学习率
        if is_pth_file and not lr_reduced:
            mae_threshold = hyperparameters.get('mae_threshold_for_lr_reduction', None)
            fine_tune_lr = hyperparameters.get('fine_tune_lr', None)
            if mae_threshold is not None and fine_tune_lr is not None:
                current_mae = mae_for_scheduler  # 使用验证集MAE（如果提供）
                if current_mae <= mae_threshold:
                    print_report(f'\n{"="*60}')
                    print_report(f'【学习率调整】{eval_split_label} MAE达到阈值 {mae_threshold:.6f}（当前MAE: {current_mae:.6f}）')
                    print_report(f'将学习率从 {original_lr:.6f} 降低至 {fine_tune_lr:.6f} 进行精细微调')
                    print_report(f'{"="*60}\n')
                    
                    # 更新所有参数组的学习率
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = fine_tune_lr
                    
                    lr_reduced = True
                    # 重新创建调度器以适应新的学习率
                    scheduler = _create_scheduler(optimizer)
                    print_report(f'已切换到精细微调模式，学习率: {fine_tune_lr:.6f}')
        
        # Save best model（使用验证集MAE，如果提供）
        if mae_for_scheduler < best_MAE:
            best_model = copy.deepcopy(model.state_dict())
            best_MAE = mae_for_scheduler
            
            # 🔑 如果从检查点恢复，且找到了比恢复时更好的模型，则恢复早停
            if resumed_from_checkpoint and resumed_best_mae is not None:
                # 如果恢复时的最佳MAE是inf（表示没有验证集历史），或者找到了更好的模型，则恢复早停
                if resumed_best_mae == np.inf or mae_for_scheduler < resumed_best_mae:
                    # 找到了比恢复时更好的模型（或首次更新），恢复早停
                    early_stop_active = True
                    early_stopper = EarlyStopping(patience=early_stop_resume_patience, min_delta=early_stop_min_delta)
                    # 重置早停器状态，以新的最佳MAE为基准
                    early_stopper.best = mae_for_scheduler
                    early_stopper.num_bad = 0
                    early_stopper.should_stop = False
                    if resumed_best_mae == np.inf:
                        print_report(f'✓  Best {eval_split_label} MAE 首次更新: {best_MAE:.6f} → 已恢复早停 (patience={early_stop_resume_patience})')
                    else:
                        print_report(f'✓  Best {eval_split_label} MAE 更新: {resumed_best_mae:.6f} → {best_MAE:.6f} → 已恢复早停 (patience={early_stop_resume_patience})')
                    # 清除恢复标志，后续正常使用早停
                    resumed_from_checkpoint = False
                    resumed_best_mae = None
                else:
                    # 虽然更新了best_MAE，但还没有超过恢复时的最佳值，继续等待
                    print_report(f'✓  Best {eval_split_label} MAE 更新: {best_MAE:.6f} (仍需等待超过恢复时的最佳值 {resumed_best_mae:.6f})')
            elif not early_stop_active:
                # 非检查点恢复情况，正常启用早停
                early_stop_active = True
                early_stopper = EarlyStopping(patience=early_stop_resume_patience, min_delta=early_stop_min_delta)
                print_report(f'✓  Best {eval_split_label} MAE 更新: {best_MAE:.6f} → 重新启用早停 (patience={early_stop_resume_patience})')
        
        # 计算本轮训练时间
        epoch_time = time.time() - epoch_start_time
        
        # 在一行中显示每轮训练信息（包含训练时间）
        if val_loader is not None:
            train_mae = stats["MAE_Train"]
            # 如果当前epoch进行了验证，使用新的值；否则使用上一次的值
            if should_eval:
                valid_mae = stats["MAE_Valid"]
                valid_r2 = stats["R2_Valid"]
            else:
                valid_mae = mae_valid[-1] if len(mae_valid) > 0 else train_mae
                valid_r2 = r2_valid[-1] if len(r2_valid) > 0 else stats["R2_Train"]
            
            mae_gap = train_mae - valid_mae  # 训练集MAE - 验证集MAE，负值表示过拟合
            gap_warning = ""
            if mae_gap < -0.01:  # 如果训练集MAE比验证集MAE小超过0.01，可能过拟合
                gap_warning = f" ⚠️过拟合风险(差距:{mae_gap:.6f})"
            elif mae_gap > 0.01:  # 如果训练集MAE比验证集MAE大，可能欠拟合
                gap_warning = f" ⚠️欠拟合(差距:{mae_gap:.6f})"
            
            # 确保best_MAE显示的是验证集最佳MAE（从验证集历史中获取）
            if len(mae_valid) > 0:
                actual_best_valid_mae = min(mae_valid)
                # 如果best_MAE与验证集历史不一致，使用验证集历史最佳值
                if abs(best_MAE - actual_best_valid_mae) > 1e-6:
                    best_MAE = actual_best_valid_mae
            
            eval_marker = "" if should_eval else " (跳过验证)"
            # 计算性能指标
            try:
                expected_batches = len(train_loader)
                if expected_batches > 0:
                    time_per_batch = epoch_time / expected_batches
                    samples_per_second = (batch_size * expected_batches) / epoch_time
                    print_report(f'Epoch {epoch+1}/{total_epochs} - Time: {epoch_time:.2f}s ({time_per_batch:.3f}s/batch, {samples_per_second:.1f} samples/s) - Train MAE: {train_mae:.6f}, R²: {stats["R2_Train"]:.6f} | Valid MAE: {valid_mae:.6f}, R²: {valid_r2:.6f} | Best Valid MAE: {best_MAE:.6f}{gap_warning}{eval_marker}')
                else:
                    print_report(f'Epoch {epoch+1}/{total_epochs} - Time: {epoch_time:.2f}s - Train MAE: {train_mae:.6f}, R²: {stats["R2_Train"]:.6f} | Valid MAE: {valid_mae:.6f}, R²: {valid_r2:.6f} | Best Valid MAE: {best_MAE:.6f}{gap_warning}{eval_marker}')
            except Exception:
                print_report(f'Epoch {epoch+1}/{total_epochs} - Time: {epoch_time:.2f}s - Train MAE: {train_mae:.6f}, R²: {stats["R2_Train"]:.6f} | Valid MAE: {valid_mae:.6f}, R²: {valid_r2:.6f} | Best Valid MAE: {best_MAE:.6f}{gap_warning}{eval_marker}')
        else:
            print_report(f'Epoch {epoch+1}/{total_epochs} - Time: {epoch_time:.2f}s - MAE: {stats["MAE_Train"]:.6f}, R²: {stats["R2_Train"]:.6f} | Best MAE: {best_MAE:.6f}')
        
        # 第一轮MAE阈值检查已禁用（默认阈值为无穷大，不会触发提前终止）
        # 如果需要启用，可以在hyperparameters中设置 'first_epoch_mae_threshold' 参数
        if epoch == 0:  # 第一轮epoch完成后
            first_epoch_mae_threshold = hyperparameters.get('first_epoch_mae_threshold', float('inf'))  # 默认禁用：设置为无穷大
            first_epoch_mae = mae_for_scheduler  # 使用验证集MAE（如果提供）
            
            if first_epoch_mae > first_epoch_mae_threshold:
                print_report(f'⚠️ 第一轮{eval_split_label} MAE ({first_epoch_mae:.6f}) 超过阈值 ({first_epoch_mae_threshold:.6f})，提前终止试验')
                print_report(f'   基于历史数据分析，第一轮MAE > {first_epoch_mae_threshold:.6f} 的试验通常无法达到好的结果')
                raise FirstEpochMAEThresholdExceeded(first_epoch_mae, first_epoch_mae_threshold)
        
        # Save checkpoint
        try:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_model_state_dict': best_model,
                'best_MAE': best_MAE,
                'mae_train': mae_train,
                'r2_train': r2_train,
                'mae_valid': mae_valid,  # 🔑 添加验证集MAE历史
                'r2_valid': r2_valid,    # 🔑 添加验证集R²历史
                'early_stopping_state': {
                    'best': early_stopper.best,
                    'num_bad': early_stopper.num_bad,
                    'should_stop': early_stopper.should_stop
                },
                'hyperparameters': hyperparameters
            }
            if (epoch + 1) % checkpoint_interval == 0:
                torch.save(checkpoint, checkpoint_path)
                print_report(f'✓  轮次检查点已保存: {checkpoint_path}')
                epoch_checkpoint_path = os.path.join(checkpoint_dir, f'{model_name}_checkpoint_epoch{epoch+1:04d}.pth')
                torch.save(checkpoint, epoch_checkpoint_path)
                print_report(f'✓  轮次编号检查点已保存: {epoch_checkpoint_path}')
                # 每隔 checkpoint_interval 轮进行一次测试评估并保存预测结果
                if test_eval_path:
                    try:
                        if test_df_cache is None:
                            test_df_cache = pd.read_csv(test_eval_path)
                        test_mae, test_r2, test_predictions, test_targets = evaluate_on_testset(
                            model,
                            test_df=test_df_cache,
                            device=device,
                            batch_size=test_eval_batch,
                            subset_size=test_eval_subset,
                        )
                        print_report(f'【测试集评估】Epoch {epoch+1}: MAE={test_mae:.6f}, R²={test_r2:.6f}')
                    except Exception as e:
                        print_report(f'警告: 轮次测试集评估失败 (Epoch {epoch+1}): {e}')
        except Exception as e:
            print_report(f'警告: 保存检查点失败: {e}')

        # 每 checkpoint_interval 轮sleep 0.1 秒
        should_restart_scheduler = False
        restart_labels = []
        if scheduler_restart_epochs_relative and relative_epoch in scheduler_restart_epochs_relative:
            should_restart_scheduler = True
            restart_labels.append('relative')
        if scheduler_restart_epochs_absolute and (epoch + 1) in scheduler_restart_epochs_absolute:
            should_restart_scheduler = True
            restart_labels.append('absolute')
        if should_restart_scheduler:
            suffix = '_'.join(sorted(set(restart_labels)))
            print_report(f'>>> 触发调度器重启（{suffix}）以恢复周期扰动')
            base_lr = hyperparameters.get('lr', lr)
            for param_group in optimizer.param_groups:
                param_group['lr'] = base_lr
            scheduler = _create_scheduler(optimizer)
            _reset_optimizer_momentum(optimizer)
            print_report('✓ 调度器已重建，准备进入下一轮周期')
            early_stop_active = False
            early_stopper = EarlyStopping(patience=early_stop_resume_patience, min_delta=early_stop_min_delta)
            print_report(f'>>> 早停已禁用，待更新 Best MAE 后再恢复 (patience={early_stop_resume_patience})')
        if checkpoint_interval > 0 and (epoch + 1) % checkpoint_interval == 0:
            time.sleep(0.1)

    print_report('-' * 30)
    if val_loader is not None and len(mae_valid) > 0:
        # 如果有验证集，显示验证集最佳结果（从验证集历史中获取真实最佳值）
        actual_best_valid_mae = min(mae_valid)
        best_valid_epoch = mae_valid.index(actual_best_valid_mae) + 1
        best_valid_epoch_idx = best_valid_epoch - 1
        # 确保best_MAE与验证集历史最佳值一致
        if abs(best_MAE - actual_best_valid_mae) > 1e-6:
            print_report(f'⚠️  注意: best_MAE变量({best_MAE:.6f})与验证集历史最佳({actual_best_valid_mae:.6f})不一致，使用验证集历史最佳值')
            best_MAE = actual_best_valid_mae
        print_report(f'最佳验证集MAE: {best_MAE:.6f} (epoch {best_valid_epoch})')
        print_report(f'验证集R²      : {r2_valid[best_valid_epoch_idx]:.6f}')
        print_report('-' * 30)
        if len(mae_train) > 0:
            best_train_epoch = mae_train.index(min(mae_train)) + 1
            best_train_epoch_idx = best_train_epoch - 1
            print_report('最佳训练集MAE: ' + str(mae_train[best_train_epoch_idx]) + f' (epoch {best_train_epoch})')
            print_report('训练集R²      : ' + str(r2_train[best_train_epoch_idx]))
            best_epoch_idx = best_valid_epoch_idx  # 用于后续保存
        else:
            print_report('警告: 没有训练数据')
            best_epoch_idx = best_valid_epoch_idx
    else:
        # 如果没有验证集，显示训练集最佳结果
        if len(mae_train) > 0:
            best_epoch = mae_train.index(min(mae_train)) + 1
            best_epoch_idx = best_epoch - 1
            print_report('Best Epoch     : ' + str(best_epoch))
            print_report('Training MAE   : ' + str(mae_train[best_epoch_idx]))
            print_report('Training R²    : ' + str(r2_train[best_epoch_idx]))
        else:
            print_report('警告: 没有训练数据，无法确定最佳epoch')
            best_epoch_idx = -1

    # Save training trajectory
    try:
        df_model_training = pd.DataFrame()
        if len(mae_train) > 0:
            df_model_training['MAE_Train'] = mae_train
            df_model_training['R2_Train'] = r2_train
        else:
            # 如果没有训练数据，创建空的DataFrame
            df_model_training['MAE_Train'] = []
            df_model_training['R2_Train'] = []
        
        # 确保路径存在（保存到训练文件目录）
        save_path = training_files_dir if custom_save_path else path
        if not os.path.exists(save_path):
            os.makedirs(save_path, exist_ok=True)
        
        save_train_traj(save_path, df_model_training, valid=False)
        print_report('✓ 训练轨迹已保存: Training.csv')
    except Exception as e:
        print_report(f'警告: 保存训练轨迹失败: {e}')
        import traceback
        traceback.print_exc()

    # Save best model（保存到训练文件目录）
    try:
        # 确保路径存在
        save_path = training_files_dir if custom_save_path else path
        if not os.path.exists(save_path):
            os.makedirs(save_path, exist_ok=True)
        
        # 如果best_model为None，使用当前模型状态
        if best_model is None:
            print_report('警告: best_model为None，使用当前模型状态保存')
            best_model = model.state_dict()
        
        # 使用os.path.join确保路径正确
        model_pth_path = os.path.join(save_path, f'{model_name}.pth')
        torch.save(best_model, model_pth_path)
        print_report(f'✓ 最佳模型已保存: {model_pth_path}')
    except Exception as e:
        print_report(f'错误: 保存最佳模型失败: {e}')
        import traceback
        traceback.print_exc()
    
    # 训练完成后评估测试集（如果配置了）
    if test_eval_path:
        print_report('-' * 60)
        print_report('【测试集评估】训练完成，开始评估测试集...')
        try:
            if test_df_cache is None:
                test_df_cache = pd.read_csv(test_eval_path)
            test_mae, test_r2, test_predictions, test_targets = evaluate_on_testset(
                model,
                test_df=test_df_cache,
                device=device,
                batch_size=test_eval_batch,
                subset_size=test_eval_subset
            )
            print_report(f'测试集评估结果:')
            print_report(f'  - 测试集MAE: {test_mae:.6f}')
            print_report(f'  - 测试集R²: {test_r2:.6f}')
            print_report(f'  - 测试集样本数: {len(test_predictions)}')
            try:
                save_path = training_files_dir if custom_save_path else path
                os.makedirs(save_path, exist_ok=True)
                test_out = os.path.join(save_path, 'test_results_best.json')
                import json
                with open(test_out, 'w', encoding='utf-8') as fp:
                    json.dump(
                        {
                            'model': model_name,
                            'mae': float(test_mae),
                            'r2': float(test_r2),
                            'n_samples': int(len(test_predictions)),
                        },
                        fp,
                        indent=2,
                        ensure_ascii=False,
                    )
                print_report(f'  - 已保存测试集结果JSON: {test_out}')
            except Exception as e_save:
                print_report(f'警告: 保存测试集预测失败: {e_save}')
            print_report('-' * 60)
        except Exception as e:
            print_report(f'警告: 测试集评估失败: {e}')
            import traceback
            traceback.print_exc()
        # 尝试保存当前模型状态作为备选
        try:
            model_pth_path = os.path.join(path, f'{model_name}_fallback.pth')
            torch.save(model.state_dict(), model_pth_path)
            print_report(f'已保存当前模型状态作为备选: {model_pth_path}')
        except Exception as e2:
            print_report(f'错误: 保存备选模型也失败: {e2}')

    end = time.time()

    print_report('\nTraining time (min): ' + str((end - start) / 60))
    report.close()
if __name__ == '__main__':
    # 所有训练产物写入本文件同目录下的 outputs/（含 checkpoint、Report、Training.csv、*.pth）
    hyperparameters_dict = {'hidden_dim'  : 38,
                            'lr'          : 0.0012947540158123575,
                            'n_epochs'    : 500,
                            'batch_size'  : 64,
                            'fine_tune_stage': 'none',
                            # 无预训练权重时关闭，避免加载失败；有权重文件时可改为 True
                            'use_pretrained': True,
                            'checkpoint_interval': 30,
                            'test_eval_path': r'dataset/all/all_merged_test.csv',
                            'custom_save_path': r'src/models/GHGEAT/outputs',
                            'training_files_save_dir': r'src/models/GHGEAT/outputs',
                            }

    df = pd.read_csv(r'dataset/all/all_merged_train.csv')
    val_df = pd.read_csv(r'dataset/all/all_merged_valid.csv')
    # 混合Brouwer_2021数据（10%-20%，默认15%）
    # 设置 mix_brouwer_ratio=0.15 表示混合15%的Brouwer_2021数据
    # 可以设置为 0.1 (10%), 0.15 (15%), 0.2 (20%) 或 None (不混合)
    # mix_ratio = 0.15  # 混合比例：0.1=10%, 0.15=15%, 0.2=20%
    # df = mix_brouwer_data(df, brouwer_path='data/raw/Brouwer_2021.csv', mix_ratio=mix_ratio, random_seed=42)

    train_GNNGH_T(df, 'GHGEAT', hyperparameters_dict, val_df=val_df)