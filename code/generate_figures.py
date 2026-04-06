from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.collections import PolyCollection


CODE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CODE_DIR.parent
from models.baseline import evaluate_baseline
from models.common import LossHistory, load_sequence_data
from models.lp_ar_pdr_v10 import evaluate_arcnn_model, train_arcnn_model
from models.mlp_pdr import evaluate_mlp_model, train_mlp_model
from models.pdr_v13_lstm import evaluate_lstm_model, train_lstm_model

try:
    from models.gnn_pdr import evaluate_gnn_model, train_gnn_model

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


def load_history(history_path: Path) -> dict[str, LossHistory]:
    payload = json.loads(history_path.read_text(encoding="utf-8"))
    return {
        model_name: LossHistory(train=curves["train"], val=curves["val"])
        for model_name, curves in payload.items()
    }


def ridge_vertices(epochs: np.ndarray, losses: list[float]) -> list[tuple[float, float]]:
    return [(float(epochs[0]), 0.0), *zip(epochs.tolist(), losses), (float(epochs[-1]), 0.0)]


def render_surface_plot(
    history: dict[str, LossHistory],
    split: str,
    output_path: Path,
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
    fig.subplots_adjust(left=0.02, right=0.98, bottom=0.05, top=0.98)
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
    parser.add_argument(
        "--render-loss-only",
        action="store_true",
        help="Redraw the loss comparison figures from an existing loss-history JSON file.",
    )
    parser.add_argument(
        "--loss-history-path",
        default=str(PROJECT_ROOT / "assets" / "comparison" / "loss-history.json"),
        help="Path to the saved loss-history JSON used with --render-loss-only.",
    )
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


def save_trajectory_plot(
    output_path: Path,
    gt_pos: np.ndarray,
    pred_pos: np.ndarray,
    rmse: float,
    axis_limits: tuple[float, float, float, float],
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # Use a portrait canvas so side-by-side slides show more of the actual path
    # instead of spending width on empty margins.
    fig, ax = plt.subplots(figsize=(5.6, 7.6))
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
    fig.subplots_adjust(left=0.10, right=0.99, bottom=0.08, top=0.93)
    fig.savefig(output_path, dpi=220, facecolor="white", bbox_inches="tight", pad_inches=0.02)
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

    if args.render_loss_only:
        history_path = Path(args.loss_history_path).expanduser()
        if not history_path.is_absolute():
            history_path = (PROJECT_ROOT / history_path).resolve()
        loss_history = load_history(history_path)
        print("\n== Regenerating 3D Loss Figures ==")
        render_surface_plot(
            history=loss_history,
            split="train",
            output_path=PROJECT_ROOT / "assets" / "comparison" / "loss-train.png",
            z_label="Training Huber Loss",
        )
        render_surface_plot(
            history=loss_history,
            split="val",
            output_path=PROJECT_ROOT / "assets" / "comparison" / "loss-val.png",
            z_label="Validation Huber Loss",
        )
        print("\n== Done ==")
        return

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

    if not HAS_GNN:
        raise RuntimeError("GNN training requires torch_geometric in the current environment.") from GNN_IMPORT_ERROR
    gnn_artifact = train_gnn_model(
        dataset_root,
        args.category,
        device,
        args.gnn_epochs,
        args.batch_size_gnn,
    )

    print("\n== Generating Trajectory Figures ==")
    baseline_traj, baseline_rmse = evaluate_baseline(sequence_data)
    mlp_traj, mlp_rmse = evaluate_mlp_model(mlp_artifact, sequence_data, device)
    arcnn_traj, arcnn_rmse = evaluate_arcnn_model(arcnn_artifact, sequence_data, device)
    lstm_traj, lstm_rmse = evaluate_lstm_model(lstm_artifact, sequence_data, device)
    gnn_traj, gnn_rmse = evaluate_gnn_model(gnn_artifact, sequence_data, device)
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
        z_label="Training Huber Loss",
    )
    render_surface_plot(
        history=loss_history,
        split="val",
        output_path=PROJECT_ROOT / "assets" / "comparison" / "loss-val.png",
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
