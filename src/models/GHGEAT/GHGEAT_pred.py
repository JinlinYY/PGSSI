"""
GNN-Gibbs-Helmholtz温度预测
"""
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from rdkit import Chem
from sklearn.preprocessing import MinMaxScaler

# 与 GHGEAT_train 一致：模型在本目录，工具在 src/models/utilities_v2/
_MODELS_ROOT = Path(__file__).resolve().parents[1]
if str(_MODELS_ROOT) not in sys.path:
    sys.path.insert(0, str(_MODELS_ROOT))

from GHGEAT_architecture import GHGEAT
from utilities_v2.mol2graph import (
    get_dataloader_pairs_T,
    n_atom_features,
    n_bond_features,
    sys2graph,
)

# df = pd.read_csv('train_norm.csv')
# targets = ['K1', 'K2']
# scaler = MinMaxScaler()
# scaler = scaler.fit(df[targets].to_numpy())
scaler = None

def pred_GNNGH_T(df, model_name, hyperparameters):
    
    target = 'log-gamma'
    
    # Build molecule from SMILES
    mol_column_solvent = 'Molecule_Solvent'
    df[mol_column_solvent] = df['Solvent_SMILES'].apply(Chem.MolFromSmiles)

    mol_column_solute = 'Molecule_Solute'
    df[mol_column_solute] = df['Solute_SMILES'].apply(Chem.MolFromSmiles)
    
    graphs_solv, graphs_solu = 'g_solv', 'g_solu'
    df[graphs_solv], df[graphs_solu] = sys2graph(df, mol_column_solvent, mol_column_solute, target)
    
    # Hyperparameters
    hidden_dim = hyperparameters['hidden_dim']

    # Dataloader
    indices = df.index.tolist()
    predict_loader = get_dataloader_pairs_T(
        df,
        indices,
        graphs_solv,
        graphs_solu,
        batch_size=32,
        shuffle=False,
        drop_last=False
    )

    # Prediction
    available_device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Model
    v_in = n_atom_features()
    print(f"v_in: {v_in}")
    e_in = n_bond_features()
    print(f"e_in: {e_in}")
    u_in = 3  # ap, bp, topopsa
    
    # 获取attention_weight参数（与训练时保持一致）
    attention_weight = hyperparameters.get('attention_weight', 1.0)
    model = GHGEAT(v_in, e_in, u_in, hidden_dim,
                   attention_weight=attention_weight)

    # 加载模型权重
    model_path = ('D:\\化工预测\\论文复现结果\\GH-GAT - 副本 (2)\\'
                  'GHGEAT_full_search_lr8.00e-04_hd38_bs104_ep434_'
                  'pretrained_attn0.8\\GHGEAT.pth')
    checkpoint = torch.load(model_path,
                            map_location=torch.device(available_device))

    print("\n加载权重文件...")

    # 检查checkpoint中是否有input_projection层
    has_input_proj_in_ckpt = any('input_projection' in k
                                 for k in checkpoint.keys())
    print(f"权重文件包含input_projection: {has_input_proj_in_ckpt}")

    if has_input_proj_in_ckpt and attention_weight < 1.0:
        # 需要先触发模型创建input_projection层
        print("触发模型input_projection层初始化...")

        # 获取第一个batch来初始化input_projection
        init_loader = get_dataloader_pairs_T(
            df,
            df.index.tolist()[:1],  # 只用第一个样本
            graphs_solv,
            graphs_solu,
            batch_size=1,
            shuffle=False,
            drop_last=False
        )
        
        model.eval()
        device_init = torch.device(available_device)
        model = model.to(device_init)
        
        with torch.no_grad():
            for batch_solvent, batch_solute, batch_T in init_loader:
                batch_solvent = batch_solvent.to(device_init)
                batch_solute = batch_solute.to(device_init)
                batch_T = batch_T.to(device_init)
                
                # 运行一次forward来触发input_projection的创建
                _ = model(batch_solvent, batch_solute, batch_T)
                print("✓ input_projection层已初始化")
                break

        # 现在加载完整权重（包括input_projection）
        print("加载完整权重（包括input_projection）...")
        load_result = model.load_state_dict(checkpoint, strict=False)

        if load_result.missing_keys:
            print(f"  缺失的键: {len(load_result.missing_keys)}")
        if load_result.unexpected_keys:
            print(f"  意外的键: {len(load_result.unexpected_keys)}")
        print("✓ 权重加载完成")
    else:
        # 标准加载流程
        print("使用标准加载流程...")
        load_result = model.load_state_dict(checkpoint, strict=False)
        print("✓ 权重加载完成")

    device = torch.device(available_device)
    model = model.to(device)

    # 模型状态检查
    print("\n" + "=" * 60)
    print("模型状态检查:")
    print("=" * 60)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"模型总参数量: {total_params}")
    print(f"attention_weight: {attention_weight}")
    has_nan = any(torch.isnan(p).any() for p in model.parameters())
    print(f"是否有NaN权重: {has_nan}")
    print("=" * 60 + "\n")

    y_pred_final = np.array([])
    model.eval()
    with torch.no_grad():
        for batch_solvent, batch_solute, batch_T in predict_loader:
            batch_solvent = batch_solvent.to(device)
            batch_solute = batch_solute.to(device)
            batch_T = batch_T.to(device)

            if torch.cuda.is_available():
                y_pred = model(batch_solvent.cuda(), batch_solute.cuda(),
                              batch_T.cuda())
                y_pred = y_pred.cpu().numpy().reshape(-1,)
            else:
                y_pred = model(batch_solvent, batch_solute, batch_T)
                y_pred = y_pred.cpu().numpy().reshape(-1,)

            y_pred_final = np.concatenate((y_pred_final, y_pred))
            
    df[model_name] = y_pred_final
    
    return df


epochs = [250]

Ts = [75,90,120]


hyperparameters_dict = {
    'hidden_dim': 38,
    "lr": 0.0008,
    "batch_size": 104,
    "n_epochs": 434,
    "early_stopping_patience": 34,
    "attention_weight": 0.8
}

# 多数据集对比评估

if __name__ == '__main__':
    from multiprocessing import freeze_support
    freeze_support()  # Windows multiprocessing 支持
    
    print("\n" + "="*80)
    print("【多数据集对比评估】")
    print("="*80)

    results_summary = []

    # 定义要测试的数据集
    datasets = [
        ("训练集",
         ("D:\\化工预测\\论文复现结果\\GH-GAT\\data\\processed\\"
          "new_dataset\\train_dataset\\v2\\molecule_1\\molecule_train.csv"),
         "train_pred.csv"),
        ("测试集",
         ("D:\\化工预测\\论文复现结果\\GH-GAT\\data\\processed\\"
          "new_dataset\\train_dataset\\v2\\molecule_1\\molecule_test.csv"),
         "test_pred.csv")
    ]

    for dataset_name, dataset_path, output_filename in datasets:
        print(f"\n{'='*80}")
        print(f"【{dataset_name}评估】")
        print(f"{'='*80}")
        
        # 加载数据集
        df = pd.read_csv(dataset_path)
        print(f"样本数: {len(df)}")
        print(f"log-gamma范围: [{df['log-gamma'].min():.4f}, {df['log-gamma'].max():.4f}]")
        
        # 预测
        print("开始预测...")
        df_pred = pred_GNNGH_T(df, model_name='GHGEAT', hyperparameters=hyperparameters_dict)
        
        # 计算MAE
        from sklearn.metrics import mean_absolute_error, r2_score
        y_true = df_pred['log-gamma'].values
        y_pred = df_pred['GHGEAT'].values
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)

        print(f"\n{dataset_name}结果:")
        print(f"  MAE: {mae:.6f}")
        print(f"  R²:  {r2:.6f}")

        # 保存结果
        output_path = (f'D:\\化工预测\\论文复现结果\\GH-GAT\\scr\\pred\\'
                       f'GHGEAT_1202\\{output_filename}')
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df_pred.to_csv(output_path, index=False)
        print(f"  已保存到: {output_filename}")
        
        # 保存到结果汇总
        results_summary.append({
            'dataset': dataset_name,
            'samples': len(df),
            'mae': mae,
            'r2': r2
        })

    # 显示汇总对比
    print("\n" + "="*80)
    print("【结果汇总对比】")
    print("=" * 80)
    print(f"\n当前模型配置: attention_weight="
          f"{hyperparameters_dict['attention_weight']}, "
          f"batch_size={hyperparameters_dict['batch_size']}, "
          f"lr={hyperparameters_dict['lr']}")
    print("参考最优配置: attention_weight=0.13, batch_size=34, "
          "lr=0.00052 (验证集MAE=0.09113)")
    print(f"\n{'数据集':<10} {'样本数':<10} {'MAE':<12} {'R²':<12} {'性能'}")
    print("-" * 80)

    for result in results_summary:
        dataset = result['dataset']
        samples = result['samples']
        mae = result['mae']
        r2 = result['r2']
        
        # 性能评价
        if dataset == "训练集":
            if mae < 0.10:
                status = "✅ 拟合良好"
            elif mae < 0.20:
                status = "⚠️ 欠拟合"
            else:
                status = "❌ 拟合差"
        else:
            if mae < 0.12:
                status = "✅ 优秀"
            elif mae < 0.20:
                status = "⚠️ 可接受"
            else:
                status = "❌ 较差"

        print(f"{dataset:<10} {samples:<10} {mae:<12.6f} "
              f"{r2:<12.6f} {status}")

    # 泛化能力分析
    train_mae = next(r['mae'] for r in results_summary if r['dataset'] == '训练集')
    valid_mae = next(r['mae'] for r in results_summary if r['dataset'] == '验证集')
    test_mae = next(r['mae'] for r in results_summary if r['dataset'] == '测试集')

    print("\n" + "=" * 80)
    print("【泛化能力分析】")
    print("=" * 80)
    print(f"训练集MAE: {train_mae:.6f}")
    print(f"验证集MAE: {valid_mae:.6f}")
    print(f"测试集MAE: {test_mae:.6f}")

    overfit_gap = valid_mae - train_mae
    generalization_gap = test_mae - valid_mae

    print(f"\n过拟合程度（验证集-训练集）: {overfit_gap:+.6f}")
    if overfit_gap > 0.02:
        print("  ⚠️ 可能存在过拟合")
    elif overfit_gap < -0.02:
        print("  ✅ 训练集MAE更高，欠拟合")
    else:
        print("  ✅ 拟合正常")

    print(f"\n泛化差距（测试集-验证集）: {generalization_gap:+.6f}")
    if abs(generalization_gap) < 0.02:
        print("  ✅ 泛化能力优秀")
    elif abs(generalization_gap) < 0.05:
        print("  ⚠️ 泛化能力可接受")
    else:
        print("  ❌ 存在分布偏移或过拟合")

    print("=" * 80)

    # 如需测试其他数据集，取消注释以下代码

# 测试训练集（应该MAE很低，接近或小于验证集）
# df = pd.read_csv("D:\\化工预测\\论文复现结果\\GH-GAT\\data\\processed\\new_dataset\\train_dataset\\v2\\molecule\\molecule_train.csv")
# df_pred = pred_GNNGH_T(df, model_name='GHGEAT', hyperparameters=hyperparameters_dict)
# mae_train = mean_absolute_error(df_pred['log-gamma'], df_pred['GHGEAT'])
# print(f"训练集MAE: {mae_train:.6f}")
# df_pred.to_csv('D:\\化工预测\\论文复现结果\\GH-GAT\\scr\\pred\\GHGEAT_1202\\train_pred.csv', index=False)

# 测试测试集
# df = pd.read_csv("D:\\化工预测\\论文复现结果\\GH-GAT\\data\\processed\\new_dataset\\train_dataset\\v2\\molecule\\molecule_test.csv")
# df_pred = pred_GNNGH_T(df, model_name='GHGEAT', hyperparameters=hyperparameters_dict)
# mae_test = mean_absolute_error(df_pred['log-gamma'], df_pred['GHGEAT'])
# print(f"测试集MAE: {mae_test:.6f}")
# df_pred.to_csv('D:\\化工预测\\论文复现结果\\GH-GAT\\scr\\pred\\GHGEAT_1202\\test_pred.csv', index=False)

# df = pd.read_csv('F:\\化工预测\\论文复现结果\\GH-GEAT\\data\\processed\\Brouwer_2021_pred.csv')
# df_pred = pred_GNNGH_T(df, model_name='GHGEAT',
#                          hyperparameters=hyperparameters_dict)
# df_pred.to_csv('F:\\化工预测\\论文复现结果\\GH-GEAT\\scr\\pred\\GHGEAT\\brouwer_edge_pred.csv')
# print('Done!')
# for T in Ts:
#     print('-' * 50)
#     print('Temperature: ', T)
#
#     # Models trained on the complete train/validation set
#
#     print('Predicting with GHGEAT')
#     # df = pd.read_csv(T_path+'\\'+str(T)+'_train.csv')
#     # df_pred = pred_GNNprevious(df, model_name='GNNprevious_'+str(T),
#     #                   hyperparameters=hyperparameters_dict[T])
#     # df_pred.to_csv('F:\\化工预测\\论文复现结果\\GH-GNN\\models\\isothermal\\predictions\\GNN_previous'
#     #                +str(T)+'_train_pred.csv')
#
    # off_path = "F:\\化工预测\\论文复现结果\\GH-GEAT\\scr\\isothermal\\T_dataset\\test"
    # csv_path = os.path.join(off_path, str(T) + '_test.csv')
    # df = pd.read_csv(csv_path)
    # df_pred = pred_GNNGH_T(df, model_name='GHGEAT_' + str(T),
    #                        hyperparameters=hyperparameters_dict, T=T)
    # df_pred.to_csv('F:\\化工预测\\论文复现结果\\GH-GEAT\\scr\\isothermal\\predictions\\GHGEAT'
    #                + '\\' + str(T) + '_test_pred.csv')
    # print('Done!')
    


