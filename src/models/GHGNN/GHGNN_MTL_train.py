import numpy as np
# Scientific computing
import pandas as pd

# RDKiT
from rdkit import Chem

# Internal utilities
from GHGNN_MTL_architecture import GHGNN_MTL, count_parameters
from utilities.mol2graph import get_dataloader_pairs, sys2graph_MTL, n_atom_features, n_bond_features
from utilities.Train_eval_MTL import train, eval, MAE
from utilities.save_info import save_train_traj
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




def train_GNNGH_MTL(df, model_name, hyperparameters):
    path= 'F:\\化工预测\\论文复现结果\\GH-GEAT\\scr\\models\\MTL_train\\Gelu'
    path = path + '\\' + model_name

    if not os.path.exists(path):
        os.makedirs(path)

    # Open report file
    report = open(path + '\\Report_training_' + model_name + '.txt', 'w')

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
    scaler = MinMaxScaler()
    scaler = scaler.fit(df[targets].to_numpy())

    graphs_solv, graphs_solu = 'g_solv', 'g_solu'
    df[graphs_solv], df[graphs_solu] = sys2graph_MTL(df, mol_column_solvent, mol_column_solute, targets)

    # Hyperparameters
    hidden_dim = hyperparameters['hidden_dim']
    lr = hyperparameters['lr']
    n_epochs = hyperparameters['n_epochs']
    batch_size = hyperparameters['batch_size']

    start = time.time()

    # Data loaders
    train_loader = get_dataloader_pairs(df,
                                         train_index,
                                         graphs_solv,
                                         graphs_solu,
                                         batch_size,
                                         shuffle=True,
                                         drop_last=True)

    # Model
    v_in = n_atom_features()
    e_in = n_bond_features()
    u_in = 3  # ap, bp, topopsa
    model = GHGNN_MTL(v_in, e_in, u_in, hidden_dim)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    print('    Number of model parameters: ', count_parameters(model))

    # Optimizer
    # optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr,momentum=0.9)
    scheduler = reduce_lr(optimizer, mode='min', factor=0.8, patience=3, min_lr=1e-7)

    # Mixed precision training with autocast
    if torch.cuda.is_available():
        pbar = range(n_epochs)
        # scaler = GradScaler()
    else:
        pbar = tqdm(range(n_epochs))
        # scaler = None

    # To save trajectory(由于有两个任务，故两个列表均为为二维)
    mae_train = []
    train_loss = []

    for epoch in pbar:
        stats = OrderedDict()
        # Train
        train_stats = train(model, device, train_loader, optimizer, stats)
        stats.update(train_stats)  # 更新统计信息，包括所有任务的损失

        # Evaluation
        eval_stats = eval(model, device, train_loader, MAE, stats, split_label='Train')
        stats.update(eval_stats)  # 更新统计信息，包括所有任务的性能指标

        # Scheduler
        scheduler.step(stats['total_MAE_Train'])#参数K1和K2的MAE
        total_best_MAE = np.inf
        # Save info
        train_loss.append(stats['total_train_loss'])  # 假设有两个任务的损失
        mae_train.append(stats['total_MAE_Train'])

        # Save best model
        if stats['total_train_loss']<total_best_MAE:
            best_model = copy.deepcopy(model.state_dict())
            total_best_MAE = stats['total_train_loss']

    print_report('-' * 30)
    best_epoch = mae_train.index(min(mae_train)) + 1
    print_report('Best Epoch     : ' + str(best_epoch))
    print_report('Training MAE   : ' + str(mae_train[best_epoch-1]))
    print_report('Training Loss   : ' + str(train_loss[best_epoch-1]))

    # Save training trajectory
    df_model_training = pd.DataFrame(train_loss, columns=['total_Train_loss'])
    df_model_training['total_MAE_Train'] = mae_train
    save_train_traj(path, df_model_training, valid=False)

    # Save best model

    torch.save(best_model, path + '/' + model_name + '.pth')

    end = time.time()

    print_report('\nTraining time (min): ' + str((end - start) / 60))
    report.close()
hyperparameters_dict = {'hidden_dim'  : 113,
                        'lr'          : 0.0002532501358651798,
                        'n_epochs'    : 250,
                        'batch_size'  : 32
                        }

df = pd.read_csv('F:\\化工预测\\论文复现结果\\GH-GEAT\\data\\processed\\new_dataset\\Ki_dataset3\\Ki_dataset3_train.csv')
train_GNNGH_MTL(df, 'GHGEAT_MTL_epochs'+str(250),hyperparameters_dict)