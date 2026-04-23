"""
GNN-Gibbs-Helmholtz温度预测
"""
# train_pred_loss:0.0055
# test_pred_loss:0.0347
import pandas as pd
from rdkit import Chem
from utilities.mol2graph import get_dataloader_pairs_T, sys2graph, n_atom_features, n_bond_features
from GHGNN_architecture import GHGNN
import torch
import numpy as np
import os

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, r2_score

# df = pd.read_csv('train_norm.csv')
# targets = ['K1', 'K2']
# scaler = MinMaxScaler()
# scaler = scaler.fit(df[targets].to_numpy())
scaler = None

def pred_GNNGH_T(df, model_name, hyperparameters, T=None):
    """
    预测函数，支持使用数据集中的温度列或指定固定温度
    
    Parameters:
    -----------
    df : pd.DataFrame
        包含 Solute_SMILES, Solvent_SMILES, T (可选), log-gamma 的数据框
    model_name : str
        模型名称，用于保存预测结果列名
    hyperparameters : dict
        超参数字典，包含 hidden_dim
    T : float or None
        如果为None，使用数据集中已有的T列；如果指定值，则所有数据使用该温度
    """
    target = 'log-gamma'
    
    # Build molecule from SMILES
    mol_column_solvent     = 'Molecule_Solvent'
    df[mol_column_solvent] = df['Solvent_SMILES'].apply(Chem.MolFromSmiles)

    mol_column_solute      = 'Molecule_Solute'
    df[mol_column_solute]  = df['Solute_SMILES'].apply(Chem.MolFromSmiles)
    
    graphs_solv, graphs_solu = 'g_solv', 'g_solu'
    df[graphs_solv], df[graphs_solu] = sys2graph(df, mol_column_solvent, mol_column_solute, target)
    
    # Hyperparameters
    hidden_dim  = hyperparameters['hidden_dim']    
    
    # Dataloader
    indices = df.index.tolist()
    # 如果T为None，使用数据集中已有的T列；否则使用指定的T值
    if T is None:
        if 'T' not in df.columns:
            raise ValueError("数据集中没有T列，且未指定T参数")
        # 使用数据集中已有的T列
        print(f"使用数据集中已有的温度列，温度范围: {df['T'].min():.1f} - {df['T'].max():.1f}")
    else:
        df["T"] = T
        print(f"使用指定温度: {T}")
    
    predict_loader = get_dataloader_pairs_T(df, 
                                          indices, 
                                          graphs_solv,
                                          graphs_solu,
                                          batch_size=32, 
                                          shuffle=False, 
                                          drop_last=False)
    
    
    ######################
    # --- Prediction --- #
    ######################
    
    available_device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Model
    v_in = n_atom_features()
    print("v_in:{}".format(v_in))
    e_in = n_bond_features()
    print("e_in:{}".format(e_in))
    u_in = 3 # ap, bp, topopsa
    model = GHGNN(v_in, e_in, u_in, hidden_dim)
    
    # 尝试加载模型权重（如果存在）
    model_path = hyperparameters.get('model_path', None)
    if model_path and os.path.exists(model_path):
        print(f"Loading model weights from: {model_path}")
        checkpoint = torch.load(model_path, map_location=torch.device(available_device))
        # 处理不同的保存格式
        if isinstance(checkpoint, dict):
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'], strict=False)
            elif 'best_model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['best_model_state_dict'], strict=False)
            else:
                model.load_state_dict(checkpoint, strict=False)
        else:
            model.load_state_dict(checkpoint, strict=False)
        print("Model weights loaded successfully")
    else:
        print("Warning: No model weights specified or file not found. Using randomly initialized model.")
    
    device = torch.device(available_device)
    model = model.to(device)
    
    y_pred_final = np.array([])
    model.eval()
    with torch.no_grad():
        for batch_solvent, batch_solute, batch_T in predict_loader:
            batch_solvent = batch_solvent.to(device)
            batch_solute  = batch_solute.to(device)
            batch_T = batch_T.to(device)
            with torch.no_grad():
                if torch.cuda.is_available():
                    y_pred  = model(batch_solvent.cuda(), batch_solute.cuda(), batch_T.cuda(), scaler=scaler, ln_gamma=True)
                    y_pred = y_pred.cpu().numpy().reshape(-1,)
            
                    
                else:
                    y_pred  = model(batch_solvent, batch_solute, batch_T, scaler=scaler, ln_gamma=True).reshape(-1,)
                y_pred_final = np.concatenate((y_pred_final, y_pred))
            
    df[model_name] = y_pred_final
    
    return df


epochs = [250]


hyperparameters_dict = {'hidden_dim'  : 113,
                        'lr'          : 0.0002532501358651798,
                        'batch_size'  : 32,
                        'model_path'  : 'D:\\化工预测\\论文复现结果\\GH-GAT\\scr\\models\\GHGNN\\GHGNN_best.pth'  # 可以指定模型权重路径
                        }

Ts = [75,90,120]

# for e in epochs:
#     print('-'*50)
#     print('Epochs: ', e)
#
#     model_name = 'GHGNN'
    
    # Models trained on the complete train/validation set
    
    # print('Predicting with GHGNN')
    # df = pd.read_csv('F:\\化工预测\\论文复现结果\\GH-GEAT\\data\\processed\\new_dataset\\train_dataset\\v2\\molecule_test.csv')
    # df_pred = pred_GNNGH_T(df, model_name=model_name,
    #                   hyperparameters=hyperparameters_dict)
    # df_pred.to_csv('F:\\化工预测\\论文复现结果\\GH-GEAT\\scr\\pred\\GHGNN'+'\\test_pred.csv')

    # df = pd.read_csv('F:\\化工预测\\论文复现结果\\GH-GNN\\data\\processed\\molecule_test.csv')
    # df_pred = pred_GNNGH_T(df, model_name=model_name,
    #                   hyperparameters=hyperparameters_dict)
    # df_pred.to_csv(model_name+'/test_pred.csv')
    # print('Done!')
    
    ###################################
    # --- Predict Brouwer dataset --- #
    ###################################
    
    # Models trained on the complete train/validation set
    # print('Predicting with GHGNN')
    # df = pd.read_csv("F:\\化工预测\\论文复现结果\\GH-GEAT\\data\\processed\\filtered_processed_data.csv")
    # df_pred = pred_GNNGH_T(df, model_name=model_name,
    #                        hyperparameters=hyperparameters_dict)
    # df_pred.to_csv("F:\\化工预测\\论文复现结果\\GH-GEAT\\scr\\pred\\GHGNN\\processed_data_pred.csv")
    # print('Done!')

    # df = pd.read_csv('F:\\化工预测\\论文复现结果\\GH-GNN\\data\\processed\\brouwer_extrapolation_test.csv')
    # df_pred = pred_GNNGH_T(df, model_name=model_name,
    #                   hyperparameters=hyperparameters_dict)
    # df_pred.to_csv(model_name+'/brouwer_extrapolation_pred.csv')
    #  print('Done!')
# 预测 filtered_Brouwer_2021.csv 数据集，不区分温度
print('=' * 80)
print('Predicting filtered_Brouwer_2021.csv dataset (all temperatures)')
print('=' * 80)

# 获取项目根目录
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(script_dir))

# 数据路径（优先使用绝对路径，如果不存在则使用相对路径）
data_path_abs = r'D:\化工预测\论文复现结果\GH-GAT\data\processed\filtered_Brouwer_2021.csv'
data_path_rel = os.path.join(project_root, 'data', 'processed', 'filtered_Brouwer_2021.csv')

# 检查文件是否存在
if os.path.exists(data_path_abs):
    data_path = data_path_abs
elif os.path.exists(data_path_rel):
    data_path = data_path_rel
else:
    raise FileNotFoundError(f"找不到数据文件。尝试的路径:\n  - {data_path_abs}\n  - {data_path_rel}")

print(f'Loading data from: {data_path}')
df = pd.read_csv(data_path)

print(f'Dataset size: {len(df)} rows')
if 'T' in df.columns:
    print(f'Temperature range: {df["T"].min():.1f} - {df["T"].max():.1f} °C')
    print(f'Unique temperatures: {df["T"].nunique()}')
    print(f'Temperature values: {sorted(df["T"].unique())}')

print('\nPredicting with GHGNN (using temperatures from dataset)...')
df_pred = pred_GNNGH_T(df, 
                       model_name='GHGNN_prediction',
                       hyperparameters=hyperparameters_dict,
                       T=None)  # T=None 表示使用数据集中的温度列

# 保存预测结果
output_dir = os.path.join(project_root, 'scr', 'pred', 'GHGNN')
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, 'filtered_Brouwer_2021_pred.csv')
df_pred.to_csv(output_path, index=False)
print(f'\nPrediction completed!')
print(f'Results saved to: {output_path}')
print(f'Total predictions: {len(df_pred)}')

# 计算并输出MAE和R²（如果数据集中有真实值）
if 'log-gamma' in df_pred.columns and 'GHGNN_prediction' in df_pred.columns:
    y_true = df_pred['log-gamma'].values
    y_pred = df_pred['GHGNN_prediction'].values
    
    # 过滤掉NaN值
    valid_mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    if np.sum(valid_mask) > 0:
        y_true_valid = y_true[valid_mask]
        y_pred_valid = y_pred[valid_mask]
        
        mae = mean_absolute_error(y_true_valid, y_pred_valid)
        r2 = r2_score(y_true_valid, y_pred_valid)
        
        print('\n' + '=' * 80)
        print('Evaluation Metrics:')
        print('=' * 80)
        print(f'MAE (Mean Absolute Error): {mae:.6f}')
        print(f'R²  (Coefficient of Determination): {r2:.6f}')
        print(f'Valid samples: {np.sum(valid_mask)} / {len(df_pred)}')
        print('=' * 80)
    else:
        print('\nWarning: No valid samples for evaluation (all values are NaN)')
else:
    if 'log-gamma' not in df_pred.columns:
        print('\nWarning: Dataset does not contain "log-gamma" column (ground truth).')
        print('         Cannot calculate MAE and R².')
    if 'GHGNN_prediction' not in df_pred.columns:
        print('\nWarning: Prediction column "GHGNN_prediction" not found.')

print('=' * 80)
    


