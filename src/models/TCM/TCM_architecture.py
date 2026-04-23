"""
张量补全方法（TCM）实现
基于Tucker分解和ANOVA-ALS优化
"""
import numpy as np
from typing import Tuple, Optional, Callable, Dict, Any
from scipy.optimize import minimize
from tqdm import tqdm
import time
import os
import pickle


class TuckerDecomposition:
    """Tucker分解类"""
    
    def __init__(self, ranks: Tuple[int, int, int] = (5, 5, 2)):
        """
        初始化Tucker分解
        
        参数:
            ranks: (r1, r2, r3) 三个维度的秩
        """
        self.ranks = ranks
        self.r1, self.r2, self.r3 = ranks
        
        # 特征矩阵和核心张量
        self.H1 = None  # (M1, r1) 溶质特征矩阵
        self.H2 = None  # (M2, r2) 溶剂特征矩阵
        self.H3 = None  # (M3, r3) 温度特征矩阵
        self.K = None   # (r1, r2, r3) 核心张量
        self.reg_lambda = 0.0  # 岭回归正则
        
    def decompose(self, tensor: np.ndarray, mask: np.ndarray, 
                   max_iter: int = 100, tol: float = 1e-6, 
                   verbose: bool = True, early_stopping_patience: int = 10,
                   convergence_window: int = 5, convergence_rel_tol: float = 1e-4, min_epochs: int = 10,
                   start_iter: int = 0,
                   on_epoch_end: Optional[Callable[[int, float, float, int], None]] = None) -> np.ndarray:
        """
        执行Tucker分解
        
        参数:
            tensor: 输入张量 (M1, M2, M3)
            mask: 数据掩码 (M1, M2, M3)
            max_iter: 最大迭代次数
            tol: 收敛容差
            verbose: 是否显示进度
            early_stopping_patience: 早停耐心轮数
            
        返回:
            重构的张量
        """
        M1, M2, M3 = tensor.shape
        
        # 初始化（使用ANOVA初始化）
        self._anova_initialization(tensor, mask)
        
        # ALS优化（仅使用“连续若干轮MAE未下降”早停）
        best_error = float('inf')
        no_improve_rounds = 0
        history_mae = []
        
        # 若未指定具体轮数或为非正数，则以极大轮数代替，直到早停/收敛触发
        if max_iter is None or max_iter <= 0:
            max_iter = 10**9
        start_time = time.time()
        for iteration in range(start_iter, max_iter):
            # 交替优化各个因子矩阵
            self._update_H1(tensor, mask)
            self._update_H2(tensor, mask)
            self._update_H3(tensor, mask)
            self._update_K(tensor, mask)
            
            # 计算重构误差
            reconstructed = self.reconstruct()
            # 使用MAE作为监控指标
            error = np.nanmean(np.abs(tensor[mask] - reconstructed[mask]))
            history_mae.append(float(error))
            
            # 每轮输出 MAE 与累计用时
            if verbose:
                elapsed = time.time() - start_time
                print(f"Epoch {iteration + 1}: MAE={error:.6f}, elapsed={elapsed:.2f} s")
            
            # 仅基于“连续 early_stopping_patience 轮无下降”停止
            # 早停逻辑（以更优的MAE为准）
            if error + 1e-12 < best_error:
                best_error = error
                no_improve_rounds = 0
            else:
                no_improve_rounds += 1
                if no_improve_rounds >= early_stopping_patience:
                    if verbose:
                        print(f"早停: 连续 {early_stopping_patience} 轮MAE未下降，最佳MAE={best_error:.6f}")
                    break
            
            # 回调：用于保存断点
            if on_epoch_end is not None:
                try:
                    on_epoch_end(iteration + 1, float(error), float(best_error), int(no_improve_rounds))
                except Exception:
                    pass
        
        return self.reconstruct()
    
    def _anova_initialization(self, tensor: np.ndarray, mask: np.ndarray):
        """
        ANOVA初始化（简化版本）
        使用均值分解初始化
        """
        M1, M2, M3 = tensor.shape
        
        # 初始化特征矩阵（使用随机初始化，实际应该使用ANOVA分解）
        np.random.seed(42)
        self.H1 = np.random.randn(M1, self.r1) * 0.1
        self.H2 = np.random.randn(M2, self.r2) * 0.1
        self.H3 = np.random.randn(M3, self.r3) * 0.1
        
        # 初始化核心张量
        self.K = np.random.randn(self.r1, self.r2, self.r3) * 0.1
        
        # 使用均值填充初始化
        mean_value = np.nanmean(tensor[mask])
        if not np.isnan(mean_value):
            # 调整初始化使其接近均值
            scale = mean_value / (self.r1 * self.r2 * self.r3)
            self.K *= scale
    
    def _update_H1(self, tensor: np.ndarray, mask: np.ndarray):
        """使用ALS更新H1"""
        M1, M2, M3 = tensor.shape
        
        for i in range(M1):
            # 收集所有涉及溶质i的数据点
            indices = np.where(mask[i, :, :])
            if len(indices[0]) == 0:
                continue
            
            j_indices = indices[0]
            k_indices = indices[1]
            
            # 构建线性系统
            A = np.zeros((len(j_indices), self.r1))
            b = np.zeros(len(j_indices))
            
            for idx, (j, k) in enumerate(zip(j_indices, k_indices)):
                # 向量化：对K先与H2[j,:]收缩，再与H3[k,:]收缩，得到 (r1,) 行
                # tmp1 shape: (r1, r3)
                tmp1 = np.tensordot(self.K, self.H2[j, :], axes=(1, 0))
                # row shape: (r1,)
                row = np.tensordot(tmp1, self.H3[k, :], axes=(1, 0))
                A[idx, :] = row
                
                b[idx] = tensor[i, j, k]
            
            # 求解线性最小二乘问题
            try:
                if A.shape[0] >= self.r1:
                    AtA = A.T @ A
                    regI = self.reg_lambda * np.eye(self.r1)
                    Atb = A.T @ b
                    self.H1[i, :] = np.linalg.solve(AtA + regI, Atb)
                else:
                    self.H1[i, :] = np.linalg.lstsq(A, b, rcond=None)[0]
            except:
                pass  # 如果求解失败，保持原值
    
    def _update_H2(self, tensor: np.ndarray, mask: np.ndarray):
        """使用ALS更新H2"""
        M1, M2, M3 = tensor.shape
        
        for j in range(M2):
            indices = np.where(mask[:, j, :])
            if len(indices[0]) == 0:
                continue
            
            i_indices = indices[0]
            k_indices = indices[1]
            
            A = np.zeros((len(i_indices), self.r2))
            b = np.zeros(len(i_indices))
            
            for idx, (i, k) in enumerate(zip(i_indices, k_indices)):
                # 向量化：对K先与H1[i,:]收缩，再与H3[k,:]收缩，得到 (r2,) 行
                # tmp1 shape: (r2, r3)
                tmp1 = np.tensordot(self.K, self.H1[i, :], axes=(0, 0))  # contract a
                # after this, tmp1 shape is (r2, r3) because K[a,b,g] with a contracted
                row = np.tensordot(tmp1, self.H3[k, :], axes=(1, 0))     # contract g
                A[idx, :] = row
                
                b[idx] = tensor[i, j, k]
            
            try:
                if A.shape[0] >= self.r2:
                    AtA = A.T @ A
                    regI = self.reg_lambda * np.eye(self.r2)
                    Atb = A.T @ b
                    self.H2[j, :] = np.linalg.solve(AtA + regI, Atb)
                else:
                    self.H2[j, :] = np.linalg.lstsq(A, b, rcond=None)[0]
            except:
                pass
    
    def _update_H3(self, tensor: np.ndarray, mask: np.ndarray):
        """使用ALS更新H3"""
        M1, M2, M3 = tensor.shape
        
        for k in range(M3):
            indices = np.where(mask[:, :, k])
            if len(indices[0]) == 0:
                continue
            
            i_indices = indices[0]
            j_indices = indices[1]
            
            A = np.zeros((len(i_indices), self.r3))
            b = np.zeros(len(i_indices))
            
            for idx, (i, j) in enumerate(zip(i_indices, j_indices)):
                # 向量化：对K先与H1[i,:]收缩，再与H2[j,:]收缩，得到 (r3,) 行
                # tmp1 shape: (r2, r3)
                tmp1 = np.tensordot(self.K, self.H1[i, :], axes=(0, 0))  # contract a -> (r2, r3)
                # tmp2 shape: (r3,)
                row = np.tensordot(tmp1, self.H2[j, :], axes=(0, 0))     # contract b -> (r3,)
                A[idx, :] = row
                
                b[idx] = tensor[i, j, k]
            
            try:
                if A.shape[0] >= self.r3:
                    AtA = A.T @ A
                    regI = self.reg_lambda * np.eye(self.r3)
                    Atb = A.T @ b
                    self.H3[k, :] = np.linalg.solve(AtA + regI, Atb)
                else:
                    self.H3[k, :] = np.linalg.lstsq(A, b, rcond=None)[0]
            except:
                pass
    
    def _update_K(self, tensor: np.ndarray, mask: np.ndarray):
        """使用ALS更新核心张量K"""
        M1, M2, M3 = tensor.shape
        
        # 将问题转换为向量形式
        indices = np.where(mask)
        n_points = len(indices[0])
        
        if n_points == 0:
            return
        
        A = np.zeros((n_points, self.r1 * self.r2 * self.r3))
        b = np.zeros(n_points)
        
        for idx, (i, j, k) in enumerate(zip(indices[0], indices[1], indices[2])):
            # 行向量化：外积后展平，元素顺序按 (a,b,g)
            row = (self.H1[i, :][:, None, None] *
                   self.H2[j, :][None, :, None] *
                   self.H3[k, :][None, None, :]).reshape(-1)
            A[idx, :] = row
            b[idx] = tensor[i, j, k]
        
        try:
            # 岭回归解
            AtA = A.T @ A
            regI = self.reg_lambda * np.eye(self.r1 * self.r2 * self.r3)
            Atb = A.T @ b
            k_vec = np.linalg.solve(AtA + regI, Atb)
            self.K = k_vec.reshape(self.r1, self.r2, self.r3)
        except:
            pass
    
    def reconstruct(self) -> np.ndarray:
        """
        重构张量
        
        返回:
            重构的张量 (M1, M2, M3)
        """
        if self.H1 is None or self.H2 is None or self.H3 is None or self.K is None:
            raise ValueError("请先执行分解")
        # 向量化重构：T = sum_{a,b,g} H1[:,a] H2[:,b] H3[:,g] K[a,b,g]
        # 采用einsum高效张量收缩
        return np.einsum('ia,jb,kg,abg->ijk', self.H1, self.H2, self.H3, self.K, optimize=True)
    
    def predict(self, i: int, j: int, k: int) -> float:
        """
        预测特定位置的值
        
        参数:
            i: 溶质索引
            j: 溶剂索引
            k: 温度索引
            
        返回:
            预测值
        """
        if self.H1 is None or self.H2 is None or self.H3 is None or self.K is None:
            raise ValueError("请先执行分解")
        
        sum_val = 0.0
        for a in range(self.r1):
            for b_idx in range(self.r2):
                for g in range(self.r3):
                    sum_val += self.H1[i, a] * self.H2[j, b_idx] * self.H3[k, g] * self.K[a, b_idx, g]
        
        return sum_val


class TensorCompletionModel:
    """张量补全模型（TCM）"""
    
    def __init__(self, ranks: Tuple[int, int, int] = (5, 5, 2)):
        """
        初始化TCM模型
        
        参数:
            ranks: (r1, r2, r3) Tucker分解的秩
        """
        self.ranks = ranks
        self.tucker = TuckerDecomposition(ranks)
        self.is_fitted = False
        self._last_epoch = 0
    
    def fit(self, tensor: np.ndarray, mask: np.ndarray, 
            max_iter: int = 100, tol: float = 1e-6, verbose: bool = True,
            early_stopping_patience: int = 10,
            convergence_window: int = 5, convergence_rel_tol: float = 1e-4, min_epochs: int = 10,
            resume_state: Optional[Dict[str, Any]] = None,
            checkpoint_dir: Optional[str] = None,
            checkpoint_every: int = 0,
            reg_lambda: float = 1e-3):
        """
        训练模型
        
        参数:
            tensor: 数据张量 (M1, M2, M3)
            mask: 数据掩码 (M1, M2, M3)
            max_iter: 最大迭代次数
            tol: 收敛容差
            verbose: 是否显示进度
        """
        # 恢复断点
        start_iter = 0
        if resume_state is not None:
            self.set_state(resume_state)
            start_iter = int(resume_state.get('epoch', 0))
            if verbose:
                print(f"从断点恢复: epoch={start_iter}")

        # 设置正则
        try:
            self.tucker.reg_lambda = float(reg_lambda) if reg_lambda is not None else 0.0
        except Exception:
            self.tucker.reg_lambda = 0.0

        def on_epoch_end(epoch: int, mae: float, best_mae: float, no_improve: int):
            self._last_epoch = epoch
            if checkpoint_dir and checkpoint_every and checkpoint_every > 0:
                if epoch % checkpoint_every == 0:
                    os.makedirs(checkpoint_dir, exist_ok=True)
                    ckpt_path = os.path.join(checkpoint_dir, f"tcm_epoch_{epoch}.ckpt")
                    self.save_checkpoint(ckpt_path, epoch=epoch, last_mae=mae, best_mae=best_mae, no_improve=no_improve)
                    if verbose:
                        print(f"[Checkpoint] 已保存到 {ckpt_path}")

        self.tucker.decompose(
            tensor, mask,
            max_iter=max_iter,
            tol=tol,
            verbose=verbose,
            early_stopping_patience=early_stopping_patience,
            convergence_window=convergence_window,
            convergence_rel_tol=convergence_rel_tol,
            min_epochs=min_epochs,
            start_iter=start_iter,
            on_epoch_end=on_epoch_end
        )
        self.is_fitted = True
    
    def predict(self, tensor: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """
        预测缺失值
        
        参数:
            tensor: 数据张量
            mask: 数据掩码（True表示有数据，False表示需要预测）
            
        返回:
            预测的张量
        """
        if not self.is_fitted:
            raise ValueError("请先训练模型")
        
        # 返回完整的重构张量，由调用方根据掩码选择评估位置，避免泄漏
        return self.tucker.reconstruct()
    
    def get_temperature_features(self) -> np.ndarray:
        """
        获取温度特征矩阵H3
        
        返回:
            H3矩阵 (M3, r3)
        """
        if not self.is_fitted:
            raise ValueError("请先训练模型")
        
        return self.tucker.H3.copy()

    # -------- 断点与状态管理 --------
    def get_state(self) -> Dict[str, Any]:
        return {
            'ranks': self.tucker.ranks,
            'H1': None if self.tucker.H1 is None else self.tucker.H1.copy(),
            'H2': None if self.tucker.H2 is None else self.tucker.H2.copy(),
            'H3': None if self.tucker.H3 is None else self.tucker.H3.copy(),
            'K': None if self.tucker.K is None else self.tucker.K.copy(),
            'epoch': int(self._last_epoch)
        }

    def set_state(self, state: Dict[str, Any]) -> None:
        ranks = tuple(state.get('ranks', self.tucker.ranks))
        self.tucker.ranks = ranks
        self.tucker.r1, self.tucker.r2, self.tucker.r3 = ranks
        self.tucker.H1 = state.get('H1', None)
        self.tucker.H2 = state.get('H2', None)
        self.tucker.H3 = state.get('H3', None)
        self.tucker.K = state.get('K', None)
        self._last_epoch = int(state.get('epoch', 0))

    def save_checkpoint(self, file_path: str, **extra) -> None:
        data = {
            'model_state': self.get_state(),
            'extra': extra or {}
        }
        with open(file_path, 'wb') as f:
            pickle.dump(data, f)

    @staticmethod
    def load_checkpoint(file_path: str) -> Dict[str, Any]:
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        return data


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray, 
                     mask: Optional[np.ndarray] = None) -> dict:
    """
    计算评估指标
    
    参数:
        y_true: 真实值
        y_pred: 预测值
        mask: 可选的数据掩码
        
    返回:
        包含各种指标的字典
    """
    if mask is not None:
        y_true = y_true[mask]
        y_pred = y_pred[mask]
    
    # 移除NaN值
    valid_mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    y_true = y_true[valid_mask]
    y_pred = y_pred[valid_mask]
    
    if len(y_true) == 0:
        return {}
    
    # 基本指标
    mse = np.mean((y_true - y_pred) ** 2)
    mae = np.mean(np.abs(y_true - y_pred))
    rmse = np.sqrt(mse)
    
    # Winsorized指标（限制异常值影响）
    def winsorize(data, percentile=5):
        lower = np.percentile(data, percentile)
        upper = np.percentile(data, 100 - percentile)
        return np.clip(data, lower, upper)
    
    y_true_w = winsorize(y_true)
    y_pred_w = winsorize(y_pred)
    
    wmse = np.mean((y_true_w - y_pred_w) ** 2)
    wmae = np.mean(np.abs(y_true_w - y_pred_w))
    
    # R²
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
    
    return {
        'MSE': mse,
        'MAE': mae,
        'RMSE': rmse,
        'wMSE': wmse,
        'wMAE': wmae,
        'R2': r2
    }

