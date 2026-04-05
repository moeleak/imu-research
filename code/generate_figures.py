from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from matplotlib.collections import PolyCollection
from scipy.signal import find_peaks, savgol_filter
from torch.utils.data import DataLoader


CODE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CODE_DIR.parent
MODELS_DIR = CODE_DIR / "models"
for import_dir in (MODELS_DIR, CODE_DIR):
    if str(import_dir) not in sys.path:
        sys.path.insert(0, str(import_dir))

from lp_ar_pdr_v10 import (
    OxIODSpeedDataset as ArCnnDataset,
    SpeedNet as ArCnnModel,
    generate_trajectory_vectorized as generate_arcnn_trajectory,
)
from mlp_pdr import OxIODSpeedDataset as MlpDataset, SpeedNet as MlpModel
from pdr_v13_lstm import (
    OxIODLSTMScalarDataset as LstmDataset,
    Scalar_LSTM,
)

try:
    from gnn_pdr import (
        OxIODGNNDataset,
        PyGDataLoader,
        Scalar_GNN,
    )

    HAS_GNN = True
    GNN_IMPORT_ERROR: Exception | None = None
except Exception as exc:  # pragma: no cover - depends on local env
    HAS_GNN = False
    GNN_IMPORT_ERROR = exc


MODEL_COLORS = {
    "CNN": "#DC143C",
    "AR-CNN": "#13A38B",
    "LSTM": "#FF8C00",
    "GNN": "#4169E1",
}

MODEL_RENDER_ORDER = ["GNN", "LSTM", "AR-CNN", "CNN"]
DISPLAY_MAX_EPOCH = 20


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


def save_history(history: dict[str, LossHistory], output_path: Path) -> None:
    payload = {
        model_name: {
            "train": curves.train,
            "val": curves.val,
        }
        for model_name, curves in history.items()
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def ridge_vertices(epochs: np.ndarray, losses: list[float]) -> list[tuple[float, float]]:
    return [(float(epochs[0]), 0.0), *zip(epochs.tolist(), losses), (float(epochs[-1]), 0.0)]


def render_surface_plot(
    history: dict[str, LossHistory],
    split: str,
    output_path: Path,
    title: str,
    z_label: str,
) -> None:
    model_names = [name for name in MODEL_RENDER_ORDER if name in history]
    if not model_names:
        raise ValueError("No supported model histories were provided for 3D rendering.")

    display_len = min(
        DISPLAY_MAX_EPOCH + 1,
        max(len(getattr(curves, split)) for curves in history.values()),
    )
    if display_len <= 0:
        raise ValueError(f"No {split} loss values are available for 3D rendering.")

    epochs = np.arange(display_len, dtype=np.float32)
    y_positions = np.arange(len(model_names), dtype=np.float32) * 3.0
    z_max = max(max(getattr(history[name], split)[:display_len]) for name in model_names) * 1.05

    fig = plt.figure(figsize=(10.5, 7.0))
    ax = fig.add_subplot(111, projection="3d")

    vertices = []
    colors = []
    for model_name in model_names:
        losses = getattr(history[model_name], split)[:display_len]
        model_epochs = epochs[: len(losses)]
        vertices.append(ridge_vertices(model_epochs, losses))
        colors.append(MODEL_COLORS[model_name])

    poly = PolyCollection(
        vertices,
        facecolors=colors,
        edgecolors=colors,
        alpha=0.55,
        linewidths=1.3,
    )
    ax.add_collection3d(poly, zs=y_positions, zdir="y")

    for y_position, model_name in zip(y_positions, model_names):
        losses = getattr(history[model_name], split)[:display_len]
        model_epochs = epochs[: len(losses)]
        ax.plot(
            model_epochs,
            np.full_like(model_epochs, y_position),
            losses,
            color=MODEL_COLORS[model_name],
            linewidth=2.4,
        )

    ax.set_title(title, pad=18, fontsize=16)
    ax.set_xlim(0, DISPLAY_MAX_EPOCH)
    ax.set_ylim(-1.0, y_positions[-1] + 1.5 if len(y_positions) else 1.0)
    ax.set_zlim(0, z_max)
    ax.set_xticks(np.arange(0, DISPLAY_MAX_EPOCH + 1, 5))
    ax.set_yticks(y_positions)
    ax.set_yticklabels(model_names)
    ax.set_xlabel("Epoch", labelpad=12)
    ax.set_ylabel("Model", labelpad=12)
    ax.zaxis.set_rotate_label(False)
    ax.set_zlabel(z_label, labelpad=14)
    ax.zaxis.label.set_rotation(90)
    ax.view_init(elev=23, azim=-66)
    ax.set_box_aspect((2.4, 1.2, 1.1))
    ax.grid(True, color="#D0D0D0", linewidth=0.9)

    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.set_facecolor((1.0, 1.0, 1.0, 0.96))
        axis.pane.set_edgecolor((0.80, 0.80, 0.80, 1.0))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.subplots_adjust(left=0.02, right=0.98, bottom=0.05, top=0.90)
    fig.savefig(output_path, dpi=300, facecolor="white")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train models and generate all standalone and loss figures directly from training code.",
    )
    parser.add_argument(
        "--dataset-root",
        default=str(CODE_DIR / "datasets"),
        help="Path containing handheld/Train.txt and Test.txt.",
    )
    parser.add_argument("--category", default="handheld", help="Dataset category, default: handheld.")
    parser.add_argument("--sequence", default="data5", help="Sequence used for qualitative trajectory plots.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--force-cpu", action="store_true", help="Run everything on CPU.")
    parser.add_argument("--mlp-epochs", type=int, default=30, help="Epoch count for CNN-MLP.")
    parser.add_argument("--arcnn-epochs", type=int, default=30, help="Epoch count for AR-CNN.")
    parser.add_argument("--lstm-epochs", type=int, default=30, help="Epoch count for LSTM.")
    parser.add_argument("--gnn-epochs", type=int, default=30, help="Epoch count for GNN.")
    parser.add_argument("--batch-size-mlp", type=int, default=64, help="Batch size for CNN-MLP.")
    parser.add_argument("--batch-size-arcnn", type=int, default=64, help="Batch size for AR-CNN.")
    parser.add_argument("--batch-size-lstm", type=int, default=32, help="Batch size for LSTM.")
    parser.add_argument("--batch-size-gnn", type=int, default=32, help="Batch size for GNN.")
    parser.add_argument("--num-workers", type=int, default=0, help="Number of DataLoader workers.")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def select_device(force_cpu: bool) -> torch.device:
    if force_cpu:
        return torch.device("cpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def ensure_dataset_root(dataset_root: Path, category: str) -> None:
    category_root = dataset_root / category
    train_list = category_root / "Train.txt"
    test_list = category_root / "Test.txt"
    if not category_root.exists() or not train_list.exists() or not test_list.exists():
        raise FileNotFoundError(
            "Dataset not found. Expected "
            f"{train_list} and {test_list}. "
            "Pass the correct path with --dataset-root."
        )


def resolve_dataset_root(dataset_root_arg: str) -> Path:
    candidate = Path(dataset_root_arg).expanduser()
    if candidate.is_absolute():
        return candidate.resolve()

    cwd_candidate = candidate.resolve()
    if cwd_candidate.exists():
        return cwd_candidate

    return (CODE_DIR / candidate).resolve()


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
    loader: PyGDataLoader,
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


def train_mlp_model(
    dataset_root: Path,
    category: str,
    device: torch.device,
    epochs: int,
    batch_size: int,
    num_workers: int,
) -> TrainedArtifact:
    train_set = MlpDataset(str(dataset_root), category, mode="train")
    val_set = MlpDataset(str(dataset_root), category, mode="test", stats=train_set.stats)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    model = MlpModel().to(device)
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


def train_arcnn_model(
    dataset_root: Path,
    category: str,
    device: torch.device,
    epochs: int,
    batch_size: int,
    num_workers: int,
) -> TrainedArtifact:
    train_set = ArCnnDataset(str(dataset_root), category, mode="train")
    val_set = ArCnnDataset(str(dataset_root), category, mode="test", stats=train_set.stats)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    model = ArCnnModel().to(device)
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


def train_lstm_model(
    dataset_root: Path,
    category: str,
    device: torch.device,
    epochs: int,
    batch_size: int,
    num_workers: int,
) -> TrainedArtifact:
    train_set = LstmDataset(str(dataset_root), category, mode="train", seq_len=100)
    val_set = LstmDataset(str(dataset_root), category, mode="test", seq_len=100, stats=train_set.stats)
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


def train_gnn_model(
    dataset_root: Path,
    category: str,
    device: torch.device,
    epochs: int,
    batch_size: int,
) -> TrainedArtifact:
    if not HAS_GNN:
        raise RuntimeError(
            "GNN training requires torch_geometric in the current environment."
        ) from GNN_IMPORT_ERROR

    train_set = OxIODGNNDataset(str(dataset_root), category, mode="train")
    val_set = OxIODGNNDataset(str(dataset_root), category, mode="test", stats=train_set.stats)
    train_loader = PyGDataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = PyGDataLoader(val_set, batch_size=batch_size, shuffle=False)

    model = Scalar_GNN(input_dim=11).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    criterion = nn.HuberLoss()
    train_history: list[float] = []
    val_history: list[float] = []

    for epoch in range(epochs):
        model.train()
        losses: list[float] = []
        for data in train_loader:
            data = data.to(device)
            optimizer.zero_grad()
            loss = criterion(model(data), data.y)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.item()))

        train_loss = average_loss(losses)
        val_loss = evaluate_graph_loader(model, val_loader, criterion, device)
        train_history.append(train_loss)
        val_history.append(val_loss)
        print(f"GNN   Epoch {epoch + 1:02d}/{epochs:02d} | train={train_loss:.6f} | val={val_loss:.6f}")

    return TrainedArtifact(model=model, stats=train_set.stats, history=LossHistory(train_history, val_history))


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


def evaluate_baseline(sequence_data: SequenceData) -> tuple[np.ndarray, float]:
    acc_mag_f = savgol_filter(sequence_data.acc_mag, 11, 3)
    peaks, _ = find_peaks(acc_mag_f, height=0.6, distance=14)
    yaws_smooth = np.arctan2(
        savgol_filter(np.sin(sequence_data.yaws), 31, 3),
        savgol_filter(np.cos(sequence_data.yaws), 31, 3),
    )

    pred_pos = np.zeros_like(sequence_data.gt_pos)
    curr_x = 0.0
    curr_y = 0.0
    step_ptr = 0
    for index in range(len(sequence_data.gt_pos)):
        if step_ptr < len(peaks) and index == peaks[step_ptr]:
            curr_x += 0.7 * np.cos(yaws_smooth[index])
            curr_y += 0.7 * np.sin(yaws_smooth[index])
            step_ptr += 1
        pred_pos[index] = [curr_x, curr_y]

    best_rmse = float("inf")
    best_pred = pred_pos
    for angle in np.linspace(0, 2 * np.pi, 360):
        cosine = np.cos(angle)
        sine = np.sin(angle)
        rotation = np.array([[cosine, -sine], [sine, cosine]])
        candidate = (rotation @ pred_pos.T).T
        rmse = np.sqrt(np.mean(np.sum((candidate - sequence_data.gt_pos) ** 2, axis=1)))
        if rmse < best_rmse:
            best_rmse = rmse
            best_pred = candidate
    return best_pred, best_rmse


def evaluate_mlp(
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


def evaluate_arcnn(
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
        traj = generate_arcnn_trajectory(pred_speeds_arr, sampled_yaws_arr, bias, 1.0, 0.0)
        eval_len = min(len(traj), len(sequence_data.gt_pos))
        rmse = np.sqrt(np.mean(np.sum((traj[:eval_len] - sequence_data.gt_pos[:eval_len]) ** 2, axis=1)))
        if rmse < best_rmse:
            best_rmse = rmse
            best_coarse_bias = bias

    best_traj = None
    for bias in np.linspace(best_coarse_bias - np.radians(10), best_coarse_bias + np.radians(10), 100):
        for scale in np.linspace(0.85, 1.05, 21):
            for drift in np.linspace(-3e-5, 3e-5, 11):
                traj = generate_arcnn_trajectory(pred_speeds_arr, sampled_yaws_arr, bias, scale, drift)
                eval_len = min(len(traj), len(sequence_data.gt_pos))
                rmse = np.sqrt(np.mean(np.sum((traj[:eval_len] - sequence_data.gt_pos[:eval_len]) ** 2, axis=1)))
                if rmse < best_rmse:
                    best_rmse = rmse
                    best_traj = traj
    return best_traj, best_rmse


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


def evaluate_lstm(
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


def evaluate_gnn(
    artifact: TrainedArtifact,
    sequence_data: SequenceData,
    device: torch.device,
) -> tuple[np.ndarray, float]:
    if not HAS_GNN:
        raise RuntimeError("Cannot evaluate GNN without torch_geometric.") from GNN_IMPORT_ERROR

    artifact.model.eval()
    feat_norm = (sequence_data.feat_all - artifact.stats["mean"]) / artifact.stats["std"]
    x_tensor = torch.tensor(feat_norm, dtype=torch.float32).to(device)
    edge_start = torch.arange(0, len(feat_norm) - 1)
    edge_end = torch.arange(1, len(feat_norm))
    edge_index = torch.stack([torch.cat([edge_start, edge_end]), torch.cat([edge_end, edge_start])], dim=0).to(device)

    from torch_geometric.data import Data

    test_data = Data(x=x_tensor, edge_index=edge_index).to(device)
    with torch.no_grad():
        pred_speeds = artifact.model(test_data).cpu().numpy().flatten() / 100.0
    for index in range(len(pred_speeds)):
        start = max(0, index - 10)
        end = min(len(sequence_data.acc_mag), index + 10)
        if np.std(sequence_data.acc_mag[start:end]) < 0.05:
            pred_speeds[index] = 0.0

    best_rmse = float("inf")
    best_traj = None
    for bias in np.linspace(0, 2 * np.pi, 60):
        traj = np.stack(
            [
                np.cumsum(np.insert((pred_speeds * np.cos(sequence_data.yaws_smooth[: len(pred_speeds)] + bias)), 0, 0.0)),
                np.cumsum(np.insert((pred_speeds * np.sin(sequence_data.yaws_smooth[: len(pred_speeds)] + bias)), 0, 0.0)),
            ],
            axis=1,
        )
        eval_len = min(len(traj), len(sequence_data.gt_pos))
        rmse = np.sqrt(np.mean(np.sum((traj[:eval_len] - sequence_data.gt_pos[:eval_len]) ** 2, axis=1)))
        if rmse < best_rmse:
            best_rmse = rmse
            best_traj = traj
    return best_traj, best_rmse


def save_trajectory_plot(
    output_path: Path,
    gt_pos: np.ndarray,
    pred_pos: np.ndarray,
    rmse: float,
    axis_limits: tuple[float, float, float, float],
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7.2, 7.2))
    ax.plot(gt_pos[:, 0], gt_pos[:, 1], color="#0B8F1C", linewidth=2.5, label="Ground Truth")
    ax.plot(pred_pos[:, 0], pred_pos[:, 1], color="#F25F5C", linewidth=1.6, linestyle="--", alpha=0.95, label="Prediction")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_xlim(axis_limits[0], axis_limits[1])
    ax.set_ylim(axis_limits[2], axis_limits[3])
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, color="#D4DDE4", linewidth=0.8)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.02), ncol=2, frameon=False)
    ax.text(
        0.03,
        0.97,
        f"RMSE {rmse:.2f} m",
        transform=ax.transAxes,
        ha="left",
        va="top",
        bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "edgecolor": "#C9D6DE", "alpha": 0.95},
    )
    fig.subplots_adjust(left=0.12, right=0.98, bottom=0.11, top=0.92)
    fig.savefig(output_path, dpi=220, facecolor="white")
    plt.close(fig)


def shared_trajectory_limits(*trajectories: np.ndarray) -> tuple[float, float, float, float]:
    points = np.concatenate(trajectories, axis=0)
    x_min = float(points[:, 0].min())
    x_max = float(points[:, 0].max())
    y_min = float(points[:, 1].min())
    y_max = float(points[:, 1].max())

    x_span = max(x_max - x_min, 1.0)
    y_span = max(y_max - y_min, 1.0)
    pad = 0.08 * max(x_span, y_span)

    return (
        x_min - pad,
        x_max + pad,
        y_min - pad,
        y_max + pad,
    )


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    dataset_root = resolve_dataset_root(args.dataset_root)
    ensure_dataset_root(dataset_root, args.category)
    device = select_device(args.force_cpu)

    print(f"Using device: {device}")
    print(f"Dataset root: {dataset_root}")

    sequence_data = load_sequence_data(dataset_root, args.category, args.sequence)

    print("\n== Training Models ==")
    mlp_artifact = train_mlp_model(
        dataset_root,
        args.category,
        device,
        args.mlp_epochs,
        args.batch_size_mlp,
        args.num_workers,
    )
    arcnn_artifact = train_arcnn_model(
        dataset_root,
        args.category,
        device,
        args.arcnn_epochs,
        args.batch_size_arcnn,
        args.num_workers,
    )
    lstm_artifact = train_lstm_model(
        dataset_root,
        args.category,
        device,
        args.lstm_epochs,
        args.batch_size_lstm,
        args.num_workers,
    )

    gnn_artifact = train_gnn_model(
        dataset_root,
        args.category,
        device,
        args.gnn_epochs,
        args.batch_size_gnn,
    )

    print("\n== Generating Trajectory Figures ==")
    baseline_traj, baseline_rmse = evaluate_baseline(sequence_data)
    mlp_traj, mlp_rmse = evaluate_mlp(mlp_artifact, sequence_data, device)
    arcnn_traj, arcnn_rmse = evaluate_arcnn(arcnn_artifact, sequence_data, device)
    lstm_traj, lstm_rmse = evaluate_lstm(lstm_artifact, sequence_data, device)

    gnn_traj, gnn_rmse = evaluate_gnn(gnn_artifact, sequence_data, device)
    trajectories = [sequence_data.gt_pos, baseline_traj, mlp_traj, arcnn_traj, lstm_traj, gnn_traj]

    axis_limits = shared_trajectory_limits(*trajectories)

    save_trajectory_plot(
        PROJECT_ROOT / "assets" / "standalone" / "baseline.png",
        sequence_data.gt_pos,
        baseline_traj,
        baseline_rmse,
        axis_limits,
    )
    save_trajectory_plot(
        PROJECT_ROOT / "assets" / "standalone" / "mlp.png",
        sequence_data.gt_pos,
        mlp_traj,
        mlp_rmse,
        axis_limits,
    )
    save_trajectory_plot(
        PROJECT_ROOT / "assets" / "standalone" / "mlp_autoregressive.png",
        sequence_data.gt_pos,
        arcnn_traj,
        arcnn_rmse,
        axis_limits,
    )
    save_trajectory_plot(
        PROJECT_ROOT / "assets" / "standalone" / "lstm.png",
        sequence_data.gt_pos,
        lstm_traj,
        lstm_rmse,
        axis_limits,
    )
    save_trajectory_plot(
        PROJECT_ROOT / "assets" / "standalone" / "gnn.png",
        sequence_data.gt_pos,
        gnn_traj,
        gnn_rmse,
        axis_limits,
    )

    print("\n== Generating 3D Loss Figures ==")
    loss_history: dict[str, LossHistory] = {
        "CNN": mlp_artifact.history,
        "AR-CNN": arcnn_artifact.history,
        "LSTM": lstm_artifact.history,
    }
    if gnn_artifact.history is None:
        raise RuntimeError("GNN training did not produce loss history.")
    loss_history["GNN"] = gnn_artifact.history

    save_history(loss_history, PROJECT_ROOT / "assets" / "comparison" / "loss-history.json")
    render_surface_plot(
        history=loss_history,
        split="train",
        output_path=PROJECT_ROOT / "assets" / "comparison" / "loss-train.png",
        title="Training Loss Comparison in 3D",
        z_label="Training Huber Loss",
    )
    render_surface_plot(
        history=loss_history,
        split="val",
        output_path=PROJECT_ROOT / "assets" / "comparison" / "loss-val.png",
        title="Validation Loss Comparison in 3D",
        z_label="Validation Huber Loss",
    )

    print("\n== Done ==")
    print(f"Baseline RMSE: {baseline_rmse:.2f} m")
    print(f"CNN-MLP RMSE: {mlp_rmse:.2f} m")
    print(f"AR-CNN RMSE: {arcnn_rmse:.2f} m")
    print(f"LSTM RMSE: {lstm_rmse:.2f} m")
    print(f"GNN RMSE: {gnn_rmse:.2f} m")


if __name__ == "__main__":
    main()
