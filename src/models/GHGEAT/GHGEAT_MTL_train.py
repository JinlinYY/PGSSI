import numpy as np
# Scientific computing
import pandas as pd

# RDKiT
from rdkit import Chem

# Internal utilities
from scr.models.GH_pyGEAT_MTL_architecture_0615 import GHGEAT_MTL, count_parameters
from scr.models.utilities_v2.mol2graph import get_dataloader_pairs, sys2graph_MTL, n_atom_features, n_bond_features
from scr.models.utilities_v2.Train_eval_MTL import train, eval, MAE, R2
from scr.models.utilities_v2.save_info import save_train_traj
# External utilities
from tqdm import tqdm
#tqdm.pandas()
from collections import OrderedDict
import copy
import time
import os

# Pytorch
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau as reduce_lr
#from torch.cuda.amp import GradScaler
from sklearn.preprocessing import MinMaxScaler




def train_GNNGH_MTL(df, model_name, hyperparameters, resume=False):
    """
    训练GHGEAT多任务模型
    
    Parameters:
    -----------
    df : pd.DataFrame
        训练数据
    model_name : str
        模型名称
    hyperparameters : dict
        超参数字典
    resume : bool
        是否从检查点恢复训练（默认False）
    """
    # 设置随机种子以确保可重复性
    import random
    random_seed = 42
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(random_seed)
        torch.cuda.manual_seed_all(random_seed)
        # 为了可重复性，需要设置 deterministic=True 和 benchmark=False
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    # 创建用于DataLoader的随机数生成器（确保数据shuffle的可重复性）
    generator = torch.Generator()
    generator.manual_seed(random_seed)
    
    path = os.getcwd()
    path = path + '\\' + model_name

    if not os.path.exists(path):
        os.makedirs(path)
    
    # 检查点文件路径
    checkpoint_path = os.path.join(path, model_name + '_checkpoint.pth')

    # Open report file
    report = open(path+'/Report_training_' + model_name + '.txt', 'w')

    def print_report(string, file=report):
        print(string)
        file.write('\n' + string)

    print_report(' Report for ' + model_name)
    print_report('-' * 50)

    # Build molecule from SMILES
    mol_column_solvent = 'Molecule_Solvent'
    df[mol_column_solvent] = df['Solvent_SMILES'].apply(Chem.MolFromSmiles)

    mol_column_solute = 'Molecule_Solute'
    df[mol_column_solute] = df['Solute_SMILES'].apply(Chem.MolFromSmiles)

    train_index = df.index.tolist()

    # target = 'log-gamma'
    targets = ['K1', 'K2']
    
    # 检查目标列是否存在
    missing_columns = [col for col in targets if col not in df.columns]
    if missing_columns:
        available_columns = list(df.columns)
        error_msg = (
            f"\n{'='*80}\n"
            f"错误: 数据文件中缺少必需的列！\n"
            f"{'='*80}\n"
            f"缺少的列: {missing_columns}\n"
            f"数据文件应包含列: {targets}\n"
            f"\n当前数据文件包含的列:\n"
            f"{', '.join(available_columns)}\n"
            f"\n请检查:\n"
            f"1. 数据文件路径是否正确\n"
            f"2. 数据文件是否包含K1和K2列\n"
            f"3. 列名是否正确（区分大小写）\n"
            f"{'='*80}\n"
        )
        print_report(error_msg)
        raise KeyError(f"数据文件中缺少必需的列: {missing_columns}")
    
    scaler = MinMaxScaler()
    scaler = scaler.fit(df[targets].to_numpy())
    
    # 输出scaler信息用于调试
    print_report(f'目标值归一化信息:')
    print_report(f'  K1范围: [{df[targets[0]].min():.6f}, {df[targets[0]].max():.6f}]')
    print_report(f'  K2范围: [{df[targets[1]].min():.6f}, {df[targets[1]].max():.6f}]')
    print_report(f'  归一化后范围: [0, 1] (MinMaxScaler)')

    graphs_solv, graphs_solu = 'g_solv', 'g_solu'
    # ⚠️ 重要：传递scaler参数以对目标值进行归一化
    df[graphs_solv], df[graphs_solu] = sys2graph_MTL(df, mol_column_solvent, mol_column_solute, targets, y_scaler=scaler)

    # Hyperparameters
    hidden_dim = hyperparameters['hidden_dim']
    lr = hyperparameters['lr']
    n_epochs = hyperparameters['n_epochs']
    batch_size = hyperparameters['batch_size']
    loss_balance_mode = hyperparameters.get('loss_balance_mode', 'fixed')
    loss_weights = hyperparameters.get('loss_weights', [1.0, 1.0])

    start = time.time()

    # Data loaders
    train_loader = get_dataloader_pairs(df,
                                         train_index,
                                         graphs_solv,
                                         graphs_solu,
                                         batch_size,
                                         shuffle=True,
                                         drop_last=True,
                                         generator=generator)  # 使用固定种子的生成器确保shuffle可重复

    # Model
    v_in = n_atom_features()
    e_in = n_bond_features()
    u_in = 3  # ap, bp, topopsa
    model = GHGEAT_MTL(v_in, e_in, u_in, hidden_dim)
    # 智能选择GPU设备：使用GPU 0
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        print(f'    检测到 {gpu_count} 个CUDA设备（PyTorch可用的NVIDIA GPU）')
        # 显示所有可用的GPU信息
        for i in range(gpu_count):
            gpu_name = torch.cuda.get_device_name(i)
            print(f'      GPU {i}: {gpu_name}')
        
        device = torch.device('cuda:0')
        print(f'    使用 GPU 0 进行训练')
    else:
        device = torch.device('cpu')
        print('    CUDA不可用，使用 CPU 进行训练')
    model = model.to(device)

    print('    Number of model parameters: ', count_parameters(model))

    # Optimizer
    loss_params = []
    param_groups = [{'params': model.parameters()}]
    if loss_balance_mode == 'uncertainty':
        log_vars = torch.nn.Parameter(torch.zeros(2, device=device))
        param_groups.append({'params': [log_vars], 'lr': hyperparameters.get('loss_lr', lr)})
    else:
        log_vars = None
    optimizer = torch.optim.SGD(param_groups, lr=lr, momentum=0.9)
    loss_config = {
        'mode': loss_balance_mode,
        'weights': loss_weights,
        'log_vars': log_vars
    }
    scheduler = reduce_lr(optimizer, mode='min', factor=0.8, patience=3, min_lr=1e-7)

    # To save trajectory(由于有两个任务，故两个列表均为为二维)
    mae_train = []
    r2_train = []  # 添加R²记录
    train_loss = []
    best_model = None
    best_metric = np.inf
    start_epoch = 0
    
    # 尝试从检查点恢复训练
    if resume and os.path.exists(checkpoint_path):
        try:
            print_report(f'检测到检查点文件: {checkpoint_path}')
            print_report('正在加载检查点...')
            checkpoint = torch.load(checkpoint_path, map_location=device)
            
            # 加载模型状态
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            if loss_balance_mode == 'uncertainty' and 'log_vars' in checkpoint:
                log_vars.data = checkpoint['log_vars']
            
            # 恢复训练状态
            start_epoch = checkpoint['epoch'] + 1
            mae_train = checkpoint.get('mae_train', [])
            r2_train = checkpoint.get('r2_train', [])  # 恢复R²记录
            train_loss = checkpoint.get('train_loss', [])
            best_metric = checkpoint.get('best_metric', np.inf)
            best_model = checkpoint.get('best_model_state_dict', None)
            
            print_report(f'成功加载检查点！')
            print_report(f'将从第 {start_epoch + 1} 轮继续训练（共 {n_epochs} 轮）')
            print_report(f'已训练轮数: {len(mae_train)}')
            if len(mae_train) > 0:
                print_report(f'当前最佳MAE: {best_metric:.6f}')
        except Exception as e:
            print_report(f'警告: 加载检查点失败: {e}')
            print_report('将从第1轮开始训练')
            start_epoch = 0
    elif resume:
        print_report(f'未找到检查点文件: {checkpoint_path}')
        print_report('将从第1轮开始训练')
    else:
        print_report('断点续训已禁用，将从第1轮开始训练')

    # Mixed precision training with autocast
    # 根据是否使用CUDA和是否恢复训练来决定进度条
    if torch.cuda.is_available():
        pbar = range(start_epoch, n_epochs)
    else:
        pbar = tqdm(range(start_epoch, n_epochs))

    for epoch in pbar:
        epoch_start = time.time()
        stats = OrderedDict()
        # Train
        train_stats = train(model, device, train_loader, optimizer, stats, loss_config=loss_config)
        stats.update(train_stats)  # 更新统计信息，包括所有任务的损失

        # Evaluation
        eval_stats = eval(model, device, train_loader, MAE, stats, split_label='Train')
        stats.update(eval_stats)  # 更新统计信息，包括所有任务的性能指标
        
        # 计算R²
        r2_stats = eval(model, device, train_loader, R2, OrderedDict(), split_label='Train')
        total_r2 = r2_stats.get('total_R2_Train', 0.0)
        r2_K1 = r2_stats.get('R2_K1_Train', None)
        r2_K2 = r2_stats.get('R2_K2_Train', None)

        # Scheduler
        scheduler.step(stats['total_MAE_Train'])#参数K1和K2的MAE
        # Save info
        train_loss.append(stats['total_train_loss'])  # 假设有两个任务的损失
        mae_train.append(stats['total_MAE_Train'])
        r2_train.append(total_r2)  # 保存R²

        # 输出每轮信息（仿照单任务版本的格式）
        epoch_time = time.time() - epoch_start
        # 如果有单独的K1和K2的R²，也显示出来
        if r2_K1 is not None and r2_K2 is not None:
            print_report(f'Epoch {epoch+1}/{n_epochs} - MAE: {stats["total_MAE_Train"]:.6f}, R^2: {total_r2:.6f} (K1: {r2_K1:.6f}, K2: {r2_K2:.6f}), Time: {epoch_time:.2f}s')
        else:
            print_report(f'Epoch {epoch+1}/{n_epochs} - MAE: {stats["total_MAE_Train"]:.6f}, R^2: {total_r2:.6f}, Time: {epoch_time:.2f}s')

        # Save best model
        if stats['total_MAE_Train'] < best_metric:
            best_model = copy.deepcopy(model.state_dict())
            best_metric = stats['total_MAE_Train']
        
        # 保存检查点（每5个epoch保存一次，或者是最佳模型时）
        if (epoch + 1) % 5 == 0 or stats['total_MAE_Train'] < best_metric:
            try:
                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'mae_train': mae_train,
                    'r2_train': r2_train,  # 保存R²记录
                    'train_loss': train_loss,
                    'best_metric': best_metric,
                    'best_model_state_dict': best_model,
                    'hyperparameters': hyperparameters,
                    'random_seed': random_seed  # 保存随机种子以便后续查询
                }
                if loss_balance_mode == 'uncertainty' and log_vars is not None:
                    checkpoint['log_vars'] = log_vars.data
                torch.save(checkpoint, checkpoint_path)
                if (epoch + 1) % 10 == 0:  # 每10个epoch打印一次保存信息
                    print_report(f'检查点已保存 (Epoch {epoch+1})')
            except Exception as e:
                print_report(f'警告: 保存检查点失败: {e}')

    print_report('-' * 30)
    best_epoch = mae_train.index(min(mae_train)) + 1
    print_report('Best Epoch     : ' + str(best_epoch))
    print_report('Training MAE   : ' + str(mae_train[best_epoch-1]))
    if len(r2_train) > 0:
        print_report('Training R^2   : ' + str(r2_train[best_epoch-1]))
    print_report('Training Loss   : ' + str(train_loss[best_epoch-1]))

    # Save training trajectory
    df_model_training = pd.DataFrame(train_loss, columns=['Train_loss'])
    df_model_training['total_MAE_Train'] = mae_train
    if len(r2_train) > 0:
        df_model_training['total_R2_Train'] = r2_train
    save_train_traj(path, df_model_training, valid=False)

    # Save best model
    if best_model is None:
        best_model = model.state_dict()
    
    # 保存最佳模型权重和元数据（包括随机种子）
    best_model_dict = {
        'model_state_dict': best_model,
        'random_seed': random_seed,
        'hyperparameters': hyperparameters,
        'best_metric': best_metric,
        'best_epoch': mae_train.index(min(mae_train)) + 1 if len(mae_train) > 0 else len(mae_train)
    }
    torch.save(best_model_dict, path + '/' + model_name + '.pth')

    end = time.time()

    print_report('\nTraining time (min): ' + str((end - start) / 60))
    report.close()

if __name__ == '__main__':
    hyperparameters_dict = {'hidden_dim'  : 38,
                            'lr'          : 0.0002532501358651798,
                            'n_epochs'    : 300,
                            'batch_size'  : 64,
                            'loss_balance_mode': 'uncertainty',
                            'loss_lr': 0.01
                            }
    df = pd.read_csv('data\\processed\\new_dataset\\Ki\\v1\\new_Ki_train.csv')
    train_GNNGH_MTL(df, '0616-GH_pyGEAT_epochs_v3_'+str(300),hyperparameters_dict)