"""
GNN-Gibbs-Helmholtz温度训练
"""
import numpy as np
# Scientific computing
import pandas as pd

# RDKiT
from rdkit import Chem
from sklearn.metrics import r2_score

# Internal utilities
import sys
from pathlib import Path

try:  # pragma: no cover
    from .GHGNN_architecture import GHGNN, count_parameters  # type: ignore
    from ..utilities.mol2graph import (  # type: ignore
        get_dataloader_pairs_T,
        sys2graph,
        n_atom_features,
        n_bond_features,
    )
except ImportError:  # pragma: no cover
    current_dir = Path(__file__).resolve().parent
    models_dir = current_dir.parent
    sys.path.append(str(current_dir))
    sys.path.append(str(models_dir))
    from GHGNN_architecture import GHGNN, count_parameters  # type: ignore  # noqa: E402
    from utilities.mol2graph import (  # type: ignore  # noqa: E402
        get_dataloader_pairs_T,
        sys2graph,
        n_atom_features,
        n_bond_features,
    )

# External utilities
from tqdm import tqdm
#from sklearn.preprocessing import MinMaxScaler
#tqdm.pandas()
from collections import OrderedDict
import copy
import time
import os

# Pytorch
import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau as reduce_lr
from torch.amp import autocast, GradScaler

    
def _compute_mae_r2(y_true: np.ndarray, y_pred: np.ndarray):
    y_true = np.asarray(y_true, dtype=np.float64).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=np.float64).reshape(-1)
    if y_true.size == 0:
        return 0.0, 0.0
    mae = float(np.mean(np.abs(y_pred - y_true)))
    if float(np.var(y_true)) < 1e-12:
        r2 = 0.0
    else:
        r2 = float(r2_score(y_true, y_pred))
    return mae, r2


def _prepare_split(df: pd.DataFrame):
    df = df.copy().reset_index(drop=True)
    mol_column_solvent = "Molecule_Solvent"
    mol_column_solute = "Molecule_Solute"
    graphs_solv, graphs_solu = "g_solv", "g_solu"
    target = "log-gamma"

    df[mol_column_solvent] = df["Solvent_SMILES"].apply(Chem.MolFromSmiles)
    df[mol_column_solute] = df["Solute_SMILES"].apply(Chem.MolFromSmiles)
    df[graphs_solv], df[graphs_solu] = sys2graph(df, mol_column_solvent, mol_column_solute, target, y_scaler=None)
    return df, graphs_solv, graphs_solu, target


def _predict_on_loader(model, data_loader, device):
    model.eval()
    chunks_pred = []
    chunks_true = []
    with torch.no_grad():
        for batch_solvent, batch_solute, batch_T in data_loader:
            batch_solvent = batch_solvent.to(device)
            batch_solute = batch_solute.to(device)
            batch_T = batch_T.to(device)
            y_hat = model(batch_solvent, batch_solute, batch_T).detach().cpu().numpy().reshape(-1)
            y = batch_solvent.y.detach().cpu().numpy().reshape(-1)
            chunks_pred.append(y_hat)
            chunks_true.append(y)
    y_pred = np.concatenate(chunks_pred) if chunks_pred else np.array([], dtype=np.float64)
    y_true = np.concatenate(chunks_true) if chunks_true else np.array([], dtype=np.float64)
    return y_true, y_pred


def _save_training_traj(outputs_dir: str, df_training: pd.DataFrame):
    os.makedirs(outputs_dir, exist_ok=True)
    out_path = os.path.join(outputs_dir, "Training.csv")
    df_training.to_csv(out_path, index=False)
    return out_path


def train_GNNGH_T(train_df, val_df, test_df, model_name, hyperparameters, resume=False):
    
    outputs_dir = os.path.join(os.path.dirname(__file__), "outputs")
    checkpoint_dir = os.path.join(outputs_dir, "checkpoint")
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Open report file
    report_path = os.path.join(outputs_dir, "Report_training_" + model_name + ".txt")
    report = open(report_path, "w", encoding="utf-8")
    def print_report(string, file=report):
        print(string)
        file.write(string + "\n")
        file.flush()

    print_report(' Report for ' + model_name)
    print_report('-'*50)
    
    train_df, graphs_solv, graphs_solu, target = _prepare_split(train_df)
    val_df, v_graphs_solv, v_graphs_solu, _ = _prepare_split(val_df)
    test_df, t_graphs_solv, t_graphs_solu, _ = _prepare_split(test_df)
    
    # Hyperparameters
    hidden_dim  = hyperparameters['hidden_dim']
    lr          = hyperparameters['lr']
    n_epochs    = hyperparameters['n_epochs']
    batch_size  = hyperparameters['batch_size']
    early_stopping_patience = hyperparameters.get('early_stopping_patience', 20)
    checkpoint_interval = int(hyperparameters.get("checkpoint_interval", 30))
    
    start       = time.time()
    
    # Data loaders
    train_loader = get_dataloader_pairs_T(train_df, 
                                          train_df.index.tolist(), 
                                          graphs_solv,
                                          graphs_solu,
                                          batch_size, 
                                          shuffle=True, 
                                          drop_last=True)
    val_loader = get_dataloader_pairs_T(val_df,
                                        val_df.index.tolist(),
                                        v_graphs_solv,
                                        v_graphs_solu,
                                        batch_size,
                                        shuffle=False,
                                        drop_last=False)
    test_loader = get_dataloader_pairs_T(test_df,
                                         test_df.index.tolist(),
                                         t_graphs_solv,
                                         t_graphs_solu,
                                         batch_size,
                                         shuffle=False,
                                         drop_last=False)
    
    available_device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # Model
    v_in = n_atom_features()
    e_in = n_bond_features()
    u_in = 3 # ap, bp, topopsa
    model    = GHGNN(v_in, e_in, u_in, hidden_dim)
    # Optional: load a pretrained checkpoint if provided.
    pretrained_path = hyperparameters.get("pretrained_path")
    if pretrained_path:
        try:
            model.load_state_dict(torch.load(pretrained_path, map_location=torch.device(available_device)), strict=False)
            print_report(f"Loaded pretrained weights: {pretrained_path}")
        except Exception as e:
            print_report(f"Warning: failed to load pretrained weights ({pretrained_path}): {e}")
    device   = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model    = model.to(device)
    scaler = GradScaler(enabled=(device.type == "cuda"))
    if device.type == 'cuda':
        device_name = torch.cuda.get_device_name(device.index or 0)
    else:
        device_name = 'CPU'
    print_report(f'Using device: {device_name}')
    
    print('    Number of model parameters: ', count_parameters(model))
    
    # Optimizer                                                           
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)  
    #optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    task_type = 'regression'
    scheduler = reduce_lr(optimizer, mode='min', factor=0.8, patience=3, min_lr=1e-7)
    
    # Mixed precision training with autocast
    if torch.cuda.is_available():
        pbar = tqdm(range(n_epochs))
    else:
        pbar = tqdm(range(n_epochs))

    
    # To save trajectory
    mae_train = []
    r2_train = []  # 添加 R² 列表
    mae_valid = []
    r2_valid = []
    train_loss = []
    best_MAE = np.inf
    best_model = None
    epochs_no_improve = 0

    # Check if we are resuming training
    if resume:
        # Load checkpoint
        checkpoint_path = os.path.join(checkpoint_dir, model_name + "_checkpoint.pth")
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location=torch.device(available_device))
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            best_MAE = checkpoint['best_MAE']
            mae_train = checkpoint['mae_train']
            r2_train = checkpoint.get('r2_train', [])  # 加载 R²，如果不存在则使用空列表
            mae_valid = checkpoint.get('mae_valid', [])
            r2_valid = checkpoint.get('r2_valid', [])
            train_loss = checkpoint['train_loss']
            print_report(f"Resuming training from epoch {start_epoch}")
        else:
            print_report("No checkpoint found, starting training from scratch")
            start_epoch = 0
    else:
        start_epoch = 0

    # Training loop
    for epoch in range(start_epoch, n_epochs):
        epoch_start = time.time()
        model.train()
        loss_sum = 0.0
        total_graphs = 0
        y_true_batches = []
        y_pred_batches = []
        
        for batch_data in train_loader:
            if len(batch_data) == 3:
                batch_solvent, batch_solute, T = batch_data
                T = T.to(device)
                has_temperature = True
            elif len(batch_data) == 2:
                batch_solvent, batch_solute = batch_data
                has_temperature = False
            else:
                raise ValueError(f"Unexpected batch size {len(batch_data)}")
            
            batch_solvent = batch_solvent.to(device)
            batch_solute = batch_solute.to(device)
            optimizer.zero_grad(set_to_none=True)
            
            with autocast(device_type=device.type, enabled=(device.type == "cuda")):
                if has_temperature:
                    pred = model(batch_solvent, batch_solute, T)
                else:
                    pred = model(batch_solvent, batch_solute)
                prediction = pred.to(torch.float32)
                real = batch_solvent.y.to(torch.float32).reshape(prediction.shape)
                loss = F.mse_loss(prediction, real)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            num_graphs = batch_solvent.num_graphs
            loss_sum += loss.item() * num_graphs
            total_graphs += num_graphs
            y_true_batches.append(real.detach().cpu())
            y_pred_batches.append(prediction.detach().cpu())
        
        epoch_loss = loss_sum / total_graphs
        y_true = torch.cat(y_true_batches, dim=0).numpy()
        y_pred = torch.cat(y_pred_batches, dim=0).numpy()
        epoch_mae, epoch_r2 = _compute_mae_r2(y_true, y_pred)

        # Validation each epoch
        yv_true, yv_pred = _predict_on_loader(model, val_loader, device)
        val_mae, val_r2 = _compute_mae_r2(yv_true, yv_pred)
        
        stats = OrderedDict()
        stats['Train_loss'] = epoch_loss
        stats['MAE_Train'] = epoch_mae
        stats['R2_Train'] = epoch_r2
        stats['MAE_Valid'] = val_mae
        stats['R2_Valid'] = val_r2
        
        scheduler.step(stats['MAE_Valid'])
        train_loss.append(epoch_loss)
        mae_train.append(epoch_mae)
        r2_train.append(epoch_r2)  # 保存 R²
        mae_valid.append(val_mae)
        r2_valid.append(val_r2)
        
        # 每轮训练结束后输出 Train/Valid MAE 和 R²
        print_report(
            f'Epoch {epoch+1}/{n_epochs} - '
            f'Train MAE: {epoch_mae:.6f}, R^2: {epoch_r2:.6f} | '
            f'Valid MAE: {val_mae:.6f}, R^2: {val_r2:.6f}'
        )
        epoch_time = time.time() - epoch_start
        print_report(f'Epoch {epoch+1} duration: {epoch_time:.2f}s')
        
        if mae_valid[-1] < best_MAE:
            best_model = copy.deepcopy(model.state_dict())
            best_MAE = mae_valid[-1]
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= early_stopping_patience:
                print_report(f'Early stopping triggered after {epoch+1} epochs (patience={early_stopping_patience})')
                break

        # Save checkpoint + test every checkpoint_interval epochs
        if checkpoint_interval > 0 and (epoch + 1) % checkpoint_interval == 0:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_MAE': best_MAE,
                'mae_train': mae_train,
                'r2_train': r2_train,
                'mae_valid': mae_valid,
                'r2_valid': r2_valid,
                'train_loss': train_loss,
                'hyperparameters': hyperparameters,
            }
            ckpt_path = os.path.join(checkpoint_dir, model_name + "_checkpoint.pth")
            epoch_ckpt_path = os.path.join(checkpoint_dir, f"{model_name}_checkpoint_epoch{epoch+1:04d}.pth")
            torch.save(checkpoint, ckpt_path)
            torch.save(checkpoint, epoch_ckpt_path)
            print_report(f"✓ Checkpoint saved: {epoch_ckpt_path}")

            yt_true, yt_pred = _predict_on_loader(model, test_loader, device)
            test_mae, test_r2 = _compute_mae_r2(yt_true, yt_pred)
            print_report(f"【测试集评估】Epoch {epoch+1}: MAE={test_mae:.6f}, R^2={test_r2:.6f}")

    print_report('-' * 30)
    best_epoch = mae_valid.index(min(mae_valid)) + 1 if len(mae_valid) > 0 else (mae_train.index(min(mae_train)) + 1)
    print_report('Best Epoch     : ' + str(best_epoch))
    if len(mae_train) >= best_epoch:
        print_report('Training MAE   : ' + str(mae_train[best_epoch - 1]))
        print_report('Training R^2   : ' + str(r2_train[best_epoch - 1]))
        print_report('Training Loss  : ' + str(train_loss[best_epoch - 1]))
    if len(mae_valid) >= best_epoch:
        print_report('Valid MAE      : ' + str(mae_valid[best_epoch - 1]))
        print_report('Valid R^2      : ' + str(r2_valid[best_epoch - 1]))

    # Save training trajectory
    df_model_training = pd.DataFrame(train_loss, columns=['Train_loss'])
    df_model_training['MAE_Train'] = mae_train
    df_model_training['R2_Train'] = r2_train  # 添加 R² 列
    traj_path = _save_training_traj(outputs_dir, df_model_training)
    print_report(f"✓ Training trajectory saved: {traj_path}")

    # Save best model
    if best_model is None:
        best_model = model.state_dict()
    best_model_path = os.path.join(outputs_dir, model_name + "_best.pth")
    torch.save(best_model, best_model_path)
    print_report(f"✓ Best model saved: {best_model_path}")

    # Final test evaluation with best model weights
    try:
        model.load_state_dict(best_model, strict=False)
        yt_true, yt_pred = _predict_on_loader(model, test_loader, device)
        test_mae, test_r2 = _compute_mae_r2(yt_true, yt_pred)
        import json
        test_out = os.path.join(outputs_dir, "test_results_best.json")
        with open(test_out, "w", encoding="utf-8") as fp:
            json.dump(
                {
                    "model": model_name,
                    "mae": float(test_mae),
                    "r2": float(test_r2),
                    "n_samples": int(yt_true.shape[0]),
                },
                fp,
                indent=2,
                ensure_ascii=False,
            )
        print_report(f"【测试集评估】Best model: MAE={test_mae:.6f}, R^2={test_r2:.6f} | Saved: {test_out}")
    except Exception as e:
        print_report(f"Warning: final best-model test evaluation failed: {e}")

    end = time.time()

    print_report('\nTraining time (min): ' + str((end - start) / 60))
    report.close()

if __name__ == "__main__":
    hyperparameters_dict = {
        "hidden_dim": 113,
        "lr": 0.0002532501358651798,
        "n_epochs": 300,
        "batch_size": 64,
        "early_stopping_patience": 10,
        "checkpoint_interval": 30,
    }

    train_df = pd.read_csv(r"dataset/all/all_merged_train.csv")
    val_df = pd.read_csv(r"dataset/all/all_merged_valid.csv")
    test_df = pd.read_csv(r"dataset/all/all_merged_test.csv")
    train_GNNGH_T(train_df, val_df, test_df, "GHGNN", hyperparameters_dict, resume=True)