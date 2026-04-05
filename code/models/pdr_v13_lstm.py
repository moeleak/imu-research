import pandas as pd
import numpy as np
import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from pathlib import Path

from .common import (
    LossHistory,
    SequenceData,
    TrainedArtifact,
    average_loss,
    evaluate_tensor_loader,
)

# ==========================================
# 1. 标量输出的 LSTM 模型
# ==========================================
class Scalar_LSTM(nn.Module):
    def __init__(self, input_dim=11, hidden_dim=128, num_layers=2):
        super(Scalar_LSTM, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=0.2)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1) # 核心回归：只预测标量距离 (Speed)
        )

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out) # Shape: [Batch, Seq_Len, 1]

# ==========================================
# 2. 逐帧标量数据集
# ==========================================
class OxIODLSTMScalarDataset(Dataset):
    def __init__(self, root_dir, category, mode='train', seq_len=100, stats=None):
        self.seq_len = seq_len
        self.features, self.targets = [], []
        
        list_file = "Train.txt" if mode == 'train' else "Test.txt"
        with open(os.path.join(root_dir, category, list_file), 'r') as f:
            folders = [line.strip() for line in f.readlines() if line.strip()]

        for rel_path in folders:
            self._process_sequence(root_dir, category, rel_path)

        self.features = np.array(self.features).astype(np.float32)
        self.targets = np.array(self.targets).astype(np.float32)

        if mode == 'train':
            self.stats = {'mean': np.mean(self.features, axis=(0, 1)), 'std': np.std(self.features, axis=(0, 1)) + 1e-6}
        else: self.stats = stats

    def _process_sequence(self, root, cat, rel_path):
        base = os.path.join(root, cat, rel_path, 'syn')
        imu_p = os.path.join(base, 'imu1.csv'); gt_p = os.path.join(base, 'vi1.csv' if cat!='large scale' else 'tango1.csv')
        if not os.path.exists(gt_p): return
        
        imu_df = pd.read_csv(imu_p, header=None).iloc[::4, :].reset_index(drop=True)
        gt_df = pd.read_csv(gt_p, header=None).iloc[::4, :].reset_index(drop=True)
        
        min_len = min(len(imu_df), len(gt_df))
        roll, pitch = imu_df.iloc[:min_len, 1].values, imu_df.iloc[:min_len, 2].values
        acc, gyro = imu_df.iloc[:min_len, 4:7].values, imu_df.iloc[:min_len, 7:10].values
        pos = gt_df.iloc[:min_len, 2:4].values

        feat = np.hstack([acc, gyro, np.sin(roll[:,None]), np.cos(roll[:,None]), 
                          np.sin(pitch[:,None]), np.cos(pitch[:,None]), np.linalg.norm(acc, axis=1)[:,None]])
        
        for i in range(0, min_len - self.seq_len, self.seq_len // 2): 
            seq_x = feat[i : i + self.seq_len]
            seq_y = []
            for t in range(i, i + self.seq_len):
                # 核心回归：只计算两帧之间的物理标量距离
                dist = np.linalg.norm(pos[t+1] - pos[t])
                seq_y.append([dist])
                
            self.features.append(seq_x)
            self.targets.append(seq_y)

    def __len__(self): return len(self.features)
    def __getitem__(self, idx):
        x = (self.features[idx] - self.stats['mean']) / self.stats['std']
        # 放大 100 倍，因为单帧位移极小
        return torch.tensor(x, dtype=torch.float32), torch.tensor(self.targets[idx], dtype=torch.float32) * 100.0


def train_lstm_model(
    dataset_root: Path | str,
    category: str,
    device: torch.device,
    epochs: int,
    batch_size: int,
    num_workers: int,
) -> TrainedArtifact:
    train_set = OxIODLSTMScalarDataset(str(dataset_root), category, mode="train", seq_len=100)
    val_set = OxIODLSTMScalarDataset(str(dataset_root), category, mode="test", seq_len=100, stats=train_set.stats)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    model = Scalar_LSTM().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    criterion = nn.HuberLoss()
    train_history: list[float] = []
    val_history: list[float] = []

    for epoch in range(epochs):
        model.train()
        losses: list[float] = []
        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.item()))

        train_loss = average_loss(losses)
        val_loss = evaluate_tensor_loader(model, val_loader, criterion, device)
        train_history.append(train_loss)
        val_history.append(val_loss)
        print(f"LSTM  Epoch {epoch + 1:02d}/{epochs:02d} | train={train_loss:.6f} | val={val_loss:.6f}")

    return TrainedArtifact(model=model, stats=train_set.stats, history=LossHistory(train_history, val_history))

# ==========================================
# 3. 矢量化轨迹生成 (用于极速搜索)
# ==========================================
def generate_trajectory_vectorized(speeds, yaws, bias, scale, drift):
    N = len(speeds)
    t_indices = np.arange(N)
    thetas = yaws + bias + (t_indices * drift)
    
    dx = (speeds * scale) * np.cos(thetas)
    dy = (speeds * scale) * np.sin(thetas)
    
    traj_x = np.cumsum(np.insert(dx, 0, 0))
    traj_y = np.cumsum(np.insert(dy, 0, 0))
    return np.stack([traj_x, traj_y], axis=1)


def align_lstm_trajectory(
    speeds: np.ndarray,
    yaws: np.ndarray,
    gt_pos: np.ndarray,
) -> tuple[np.ndarray, float]:
    def generate_traj(bias: float, scale: float, drift: float) -> np.ndarray:
        t_indices = np.arange(len(speeds))
        thetas = yaws[: len(speeds)] + bias + (t_indices * drift)
        dx = (speeds * scale) * np.cos(thetas)
        dy = (speeds * scale) * np.sin(thetas)
        traj_x = np.cumsum(np.insert(dx, 0, 0.0))
        traj_y = np.cumsum(np.insert(dy, 0, 0.0))
        return np.stack([traj_x, traj_y], axis=1)

    best_coarse_bias = 0.0
    best_rmse = float("inf")
    for bias in np.linspace(0, 2 * np.pi, 120):
        traj = generate_traj(bias, 1.0, 0.0)
        eval_len = min(len(traj), len(gt_pos))
        rmse = np.sqrt(np.mean(np.sum((traj[:eval_len] - gt_pos[:eval_len]) ** 2, axis=1)))
        if rmse < best_rmse:
            best_rmse = rmse
            best_coarse_bias = bias

    best_traj = None
    for bias in np.linspace(best_coarse_bias - np.radians(10), best_coarse_bias + np.radians(10), 100):
        for scale in np.linspace(0.85, 1.05, 21):
            for drift in np.linspace(-3e-5, 3e-5, 11):
                traj = generate_traj(bias, scale, drift)
                eval_len = min(len(traj), len(gt_pos))
                rmse = np.sqrt(np.mean(np.sum((traj[:eval_len] - gt_pos[:eval_len]) ** 2, axis=1)))
                if rmse < best_rmse:
                    best_rmse = rmse
                    best_traj = traj
    return best_traj, best_rmse


def evaluate_lstm_model(
    artifact: TrainedArtifact,
    sequence_data: SequenceData,
    device: torch.device,
) -> tuple[np.ndarray, float]:
    artifact.model.eval()
    normalized = (sequence_data.feat_all - artifact.stats["mean"]) / artifact.stats["std"]
    x_tensor = torch.from_numpy(normalized[:-1]).float().unsqueeze(0).to(device)
    with torch.no_grad():
        pred_seq = artifact.model(x_tensor).cpu().numpy()[0, :, 0] / 100.0
    for index in range(len(pred_seq)):
        start = max(0, index - 10)
        end = min(len(sequence_data.acc_mag), index + 10)
        if np.std(sequence_data.acc_mag[start:end]) < 0.05:
            pred_seq[index] = 0.0
    return align_lstm_trajectory(pred_seq, sequence_data.yaws_smooth, sequence_data.gt_pos)
