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
# 1. 速度模型 (维持 v8 的高效架构)
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
            nn.AdaptiveAvgPool1d(1),
        )
        self.fc = nn.Sequential(nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 1))

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)


# ==========================================
# 2. 标量位移数据集
# ==========================================
class OxIODSpeedDataset(Dataset):
    def __init__(self, root_dir, category, mode="train", stats=None):
        self.features, self.targets = [], []
        list_file = "Train.txt" if mode == "train" else "Test.txt"
        with open(os.path.join(root_dir, category, list_file), "r") as f:
            folders = [line.strip() for line in f.readlines() if line.strip()]

        for rel_path in folders:
            self._process_sequence(root_dir, category, rel_path)

        self.features = np.array(self.features).astype(np.float32)
        self.targets = np.array(self.targets).astype(np.float32)

        if mode == "train":
            self.stats = {
                "mean": np.mean(self.features, axis=(0, 1)),
                "std": np.std(self.features, axis=(0, 1)) + 1e-6,
            }
        else:
            self.stats = stats

    def _process_sequence(self, root, cat, rel_path):
        base = os.path.join(root, cat, rel_path, "syn")
        imu_df = (
            pd.read_csv(os.path.join(base, "imu1.csv"), header=None)
            .iloc[::4, :]
            .reset_index(drop=True)
        )
        gt_path = os.path.join(
            base, "vi1.csv" if cat != "large scale" else "tango1.csv"
        )
        if not os.path.exists(gt_path):
            return
        gt_df = pd.read_csv(gt_path, header=None).iloc[::4, :].reset_index(drop=True)

        min_len = min(len(imu_df), len(gt_df))
        roll, pitch = imu_df.iloc[:min_len, 1].values, imu_df.iloc[:min_len, 2].values
        acc, gyro = (
            imu_df.iloc[:min_len, 4:7].values,
            imu_df.iloc[:min_len, 7:10].values,
        )
        pos = gt_df.iloc[:min_len, 2:4].values

        feat = np.hstack(
            [
                acc,
                gyro,
                np.sin(roll[:, None]),
                np.cos(roll[:, None]),
                np.sin(pitch[:, None]),
                np.cos(pitch[:, None]),
                np.linalg.norm(acc, axis=1)[:, None],
            ]
        )

        for i in range(0, min_len - 30, 5):
            self.features.append(feat[i : i + 20])
            dist = np.linalg.norm(pos[i + 30] - pos[i + 20])
            self.targets.append([dist])

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        x = (self.features[idx] - self.stats["mean"]) / self.stats["std"]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(
            self.targets[idx] * 20.0, dtype=torch.float32
        )


def train_mlp_model(
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
        print(f"MLP   Epoch {epoch + 1:02d}/{epochs:02d} | train={train_loss:.6f} | val={val_loss:.6f}")

    return TrainedArtifact(model=model, stats=train_set.stats, history=LossHistory(train_history, val_history))


def evaluate_mlp_model(
    artifact: TrainedArtifact,
    sequence_data: SequenceData,
    device: torch.device,
) -> tuple[np.ndarray, float]:
    artifact.model.eval()
    pred_speeds: list[float] = []
    with torch.no_grad():
        for index in range(0, len(sequence_data.feat_all) - 30, 10):
            normalized = (sequence_data.feat_all[index : index + 20] - artifact.stats["mean"]) / artifact.stats["std"]
            x_tensor = torch.from_numpy(normalized).float().unsqueeze(0).to(device)
            speed = float(artifact.model(x_tensor).cpu().numpy()[0, 0] / 20.0)
            if np.std(sequence_data.acc_mag[index : index + 20]) < 0.06:
                speed = 0.0
            pred_speeds.append(speed)

    best_rmse = float("inf")
    best_traj = None
    for trial_bias in np.linspace(0, 2 * np.pi, 360):
        current = np.array([0.0, 0.0])
        trajectory = [current.copy()]
        scale_fix = 0.90
        drift_fix = -0.000015
        for step_index, speed in enumerate(pred_speeds):
            t_index = step_index * 10
            theta = sequence_data.yaws_smooth[t_index + 20] + trial_bias + (t_index * drift_fix)
            current += [(speed * scale_fix) * np.cos(theta), (speed * scale_fix) * np.sin(theta)]
            for _ in range(10):
                trajectory.append(current.copy())
        traj_array = np.array(trajectory)
        eval_len = min(len(traj_array), len(sequence_data.gt_pos))
        rmse = np.sqrt(np.mean(np.sum((traj_array[:eval_len] - sequence_data.gt_pos[:eval_len]) ** 2, axis=1)))
        if rmse < best_rmse:
            best_rmse = rmse
            best_traj = traj_array
    return best_traj, best_rmse
