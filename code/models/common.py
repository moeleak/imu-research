from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.signal import savgol_filter
from torch.utils.data import DataLoader


@dataclass(frozen=True)
class LossHistory:
    train: list[float]
    val: list[float]


@dataclass
class TrainedArtifact:
    model: torch.nn.Module
    stats: dict[str, np.ndarray]
    history: LossHistory | None = None


@dataclass
class SequenceData:
    yaws: np.ndarray
    yaws_smooth: np.ndarray
    acc_mag: np.ndarray
    gt_pos: np.ndarray
    feat_all: np.ndarray


def average_loss(losses: list[float]) -> float:
    return float(np.mean(losses)) if losses else float("nan")


def evaluate_tensor_loader(
    model: torch.nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    model.eval()
    losses: list[float] = []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            losses.append(float(criterion(model(x), y).item()))
    return average_loss(losses)


def evaluate_graph_loader(
    model: torch.nn.Module,
    loader: Any,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    model.eval()
    losses: list[float] = []
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            losses.append(float(criterion(model(data), data.y).item()))
    return average_loss(losses)


def load_sequence_data(dataset_root: Path, category: str, sequence: str) -> SequenceData:
    base = dataset_root / category / sequence / "syn"
    imu_df = pd.read_csv(base / "imu1.csv", header=None).iloc[::4, :].reset_index(drop=True)
    gt_df = pd.read_csv(base / "vi1.csv", header=None).iloc[::4, :].reset_index(drop=True)
    min_len = min(len(imu_df), len(gt_df))
    imu_df = imu_df.iloc[:min_len].reset_index(drop=True)
    gt_df = gt_df.iloc[:min_len].reset_index(drop=True)

    yaws = imu_df.iloc[:, 3].values
    yaws_smooth = np.arctan2(
        savgol_filter(np.sin(yaws), 51, 3),
        savgol_filter(np.cos(yaws), 51, 3),
    )
    acc_mag = np.linalg.norm(imu_df.iloc[:, 4:7].values, axis=1)
    gt_pos = gt_df.iloc[:, 2:4].values - gt_df.iloc[0, 2:4].values
    roll = imu_df.iloc[:, 1].values
    pitch = imu_df.iloc[:, 2].values
    feat_all = np.hstack(
        [
            imu_df.iloc[:, 4:10].values,
            np.sin(roll[:, None]),
            np.cos(roll[:, None]),
            np.sin(pitch[:, None]),
            np.cos(pitch[:, None]),
            acc_mag[:, None],
        ]
    ).astype(np.float32)

    return SequenceData(
        yaws=yaws,
        yaws_smooth=yaws_smooth,
        acc_mag=acc_mag,
        gt_pos=gt_pos,
        feat_all=feat_all,
    )
