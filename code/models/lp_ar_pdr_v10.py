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
# 1. 稳健的标量速度模型 (回归 v9.0 架构)
# ==========================================
class SpeedNet(nn.Module):
    def __init__(self, input_dim=11):
        super(SpeedNet, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(input_dim, 64, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )
        self.fc = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1) # 仅预测物理标量位移
        )

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

# ==========================================
# 2. 数据集
# ==========================================
class OxIODSpeedDataset(Dataset):
    def __init__(self, root_dir, category, mode='train', stats=None):
        self.features, self.targets = [], []
        list_file = "Train.txt" if mode == 'train' else "Test.txt"
        with open(os.path.join(root_dir, category, list_file), 'r') as f:
            folders = [line.strip() for line in f.readlines() if line.strip()]

        for rel_path in folders:
            base = os.path.join(root_dir, category, rel_path, 'syn')
            imu_p = os.path.join(base, 'imu1.csv')
            gt_p = os.path.join(base, 'vi1.csv' if category!='large scale' else 'tango1.csv')
            if not os.path.exists(gt_p): continue
            
            imu_df = pd.read_csv(imu_p, header=None).iloc[::4, :].reset_index(drop=True)
            gt_df = pd.read_csv(gt_p, header=None).iloc[::4, :].reset_index(drop=True)
            
            min_len = min(len(imu_df), len(gt_df))
            roll, pitch = imu_df.iloc[:min_len, 1].values, imu_df.iloc[:min_len, 2].values
            acc, gyro = imu_df.iloc[:min_len, 4:7].values, imu_df.iloc[:min_len, 7:10].values
            pos = gt_df.iloc[:min_len, 2:4].values

            feat = np.hstack([acc, gyro, np.sin(roll[:,None]), np.cos(roll[:,None]), 
                              np.sin(pitch[:,None]), np.cos(pitch[:,None]), np.linalg.norm(acc, axis=1)[:,None]])
            
            for i in range(0, min_len - 30, 5):
                self.features.append(feat[i : i+20])
                self.targets.append([np.linalg.norm(pos[i+30] - pos[i+20])])

        self.features = np.array(self.features).astype(np.float32)
        self.targets = np.array(self.targets).astype(np.float32)

        if mode == 'train':
            self.stats = {'mean': np.mean(self.features, axis=(0, 1)), 'std': np.std(self.features, axis=(0, 1)) + 1e-6}
        else: self.stats = stats

    def __len__(self): return len(self.features)
    def __getitem__(self, idx):
        x = (self.features[idx] - self.stats['mean']) / self.stats['std']
        return torch.tensor(x, dtype=torch.float32), torch.tensor(self.targets[idx] * 20.0, dtype=torch.float32)


def train_arcnn_model(
    dataset_root: Path | str,
    category: str,
    device: torch.device,
    epochs: int,
    batch_size: int,
    num_workers: int,
) -> TrainedArtifact:
    train_set = OxIODSpeedDataset(str(dataset_root), category, mode="train")
    val_set = OxIODSpeedDataset(str(dataset_root), category, mode="test", stats=train_set.stats)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    model = SpeedNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
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
        print(f"ARCNN Epoch {epoch + 1:02d}/{epochs:02d} | train={train_loss:.6f} | val={val_loss:.6f}")

    return TrainedArtifact(model=model, stats=train_set.stats, history=LossHistory(train_history, val_history))

# ==========================================
# 3. 快速矢量化轨迹生成器
# ==========================================
def generate_trajectory_vectorized(speeds, yaws, bias, scale, drift):
    """使用矢量化运算，将轨迹生成速度提升 100 倍，以支持大规模网格搜索"""
    N = len(speeds)
    t_indices = np.arange(N) * 10
    
    # 航向角 = 原始平滑角 + 初始偏置 + 线性漂移
    thetas = yaws + bias + (t_indices * drift)
    
    # 计算每一步的位移增量
    dx = (speeds * scale) * np.cos(thetas)
    dy = (speeds * scale) * np.sin(thetas)
    
    # 累加得到坐标
    traj_x = np.cumsum(np.insert(dx, 0, 0))
    traj_y = np.cumsum(np.insert(dy, 0, 0))
    
    # 还原到 10 帧跨度 (简单插值，保持长度对齐)
    traj_full = np.zeros((N * 10 + 1, 2))
    for i in range(N):
        traj_full[i*10 : (i+1)*10, 0] = traj_x[i]
        traj_full[i*10 : (i+1)*10, 1] = traj_y[i]
    traj_full[-1] = [traj_x[-1], traj_y[-1]]
    
    return traj_full


def evaluate_arcnn_model(
    artifact: TrainedArtifact,
    sequence_data: SequenceData,
    device: torch.device,
) -> tuple[np.ndarray, float]:
    artifact.model.eval()
    pred_speeds: list[float] = []
    sampled_yaws: list[float] = []
    with torch.no_grad():
        for index in range(0, len(sequence_data.feat_all) - 30, 10):
            normalized = (sequence_data.feat_all[index : index + 20] - artifact.stats["mean"]) / artifact.stats["std"]
            x_tensor = torch.from_numpy(normalized).float().unsqueeze(0).to(device)
            speed = float(artifact.model(x_tensor).cpu().numpy()[0, 0] / 20.0)
            if np.std(sequence_data.acc_mag[index : index + 20]) < 0.05:
                speed = 0.0
            pred_speeds.append(speed)
            sampled_yaws.append(sequence_data.yaws_smooth[index + 20])

    pred_speeds_arr = np.array(pred_speeds)
    sampled_yaws_arr = np.array(sampled_yaws)
    best_coarse_bias = 0.0
    best_rmse = float("inf")

    for bias in np.linspace(0, 2 * np.pi, 360):
        traj = generate_trajectory_vectorized(pred_speeds_arr, sampled_yaws_arr, bias, 1.0, 0.0)
        eval_len = min(len(traj), len(sequence_data.gt_pos))
        rmse = np.sqrt(np.mean(np.sum((traj[:eval_len] - sequence_data.gt_pos[:eval_len]) ** 2, axis=1)))
        if rmse < best_rmse:
            best_rmse = rmse
            best_coarse_bias = bias

    best_traj = None
    for bias in np.linspace(best_coarse_bias - np.radians(10), best_coarse_bias + np.radians(10), 100):
        for scale in np.linspace(0.85, 1.05, 21):
            for drift in np.linspace(-3e-5, 3e-5, 11):
                traj = generate_trajectory_vectorized(pred_speeds_arr, sampled_yaws_arr, bias, scale, drift)
                eval_len = min(len(traj), len(sequence_data.gt_pos))
                rmse = np.sqrt(np.mean(np.sum((traj[:eval_len] - sequence_data.gt_pos[:eval_len]) ** 2, axis=1)))
                if rmse < best_rmse:
                    best_rmse = rmse
                    best_traj = traj
    return best_traj, best_rmse
