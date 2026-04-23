import argparse
from datetime import datetime
import multiprocessing as mp
import sys
from pathlib import Path
from typing import Any

import matplotlib
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import yaml


matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


THIS_DIR = Path(__file__).resolve().parent
DEEP_LEARNING_DIR = THIS_DIR.parent
PROJECT_ROOT = DEEP_LEARNING_DIR.parent
if str(DEEP_LEARNING_DIR) not in sys.path:
    sys.path.append(str(DEEP_LEARNING_DIR))

from models.model_pytorch import build_model  # noqa: E402
from train.data_pipeline import OxfordPetBBoxDataset  # noqa: E402


def load_config(config_path: Path) -> dict[str, Any]:
    with config_path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    if not isinstance(config, dict):
        raise ValueError(f"Config must be a mapping: {config_path}")
    return config


def resolve_project_path(path_value: str | Path, project_root: Path) -> Path:
    path = Path(path_value).expanduser()
    if path.is_absolute():
        return path
    return (project_root / path).resolve()


def resolve_num_workers(requested_workers: int) -> int:
    if requested_workers <= 0:
        return 0
    try:
        queue = mp.get_context("spawn").Queue()
        queue.close()
        queue.join_thread()
        return requested_workers
    except (OSError, PermissionError):
        print("Warning: multiprocessing workers are not permitted here. Falling back to num_workers=0.")
        return 0


def resolve_split_file(data_dir: Path, split: str) -> Path:
    named_splits = {
        "trainval": data_dir / "annotations" / "trainval.txt",
        "test": data_dir / "annotations" / "test.txt",
    }
    split_lower = split.lower()
    if split_lower in named_splits:
        return named_splits[split_lower]
    return resolve_project_path(split, PROJECT_ROOT)


def build_loss(name: str, params: dict[str, Any]) -> torch.nn.Module:
    loss_name = name.lower()
    if loss_name == "smooth_l1":
        return torch.nn.SmoothL1Loss(**params)
    if loss_name == "huber":
        return torch.nn.HuberLoss(**params)
    if loss_name == "mse":
        return torch.nn.MSELoss(**params)
    if loss_name == "l1":
        return torch.nn.L1Loss(**params)
    raise ValueError(f"Unsupported PyTorch loss: {name}")


def xywh_to_xyxy(boxes: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    x_center = boxes[:, 0]
    y_center = boxes[:, 1]
    width = boxes[:, 2]
    height = boxes[:, 3]
    x1 = (x_center - width / 2.0).clamp(0.0, 1.0)
    y1 = (y_center - height / 2.0).clamp(0.0, 1.0)
    x2 = (x_center + width / 2.0).clamp(0.0, 1.0)
    y2 = (y_center + height / 2.0).clamp(0.0, 1.0)
    return x1, y1, x2, y2


def bbox_iou(pred_xywh: torch.Tensor, target_xywh: torch.Tensor) -> torch.Tensor:
    pred_x1, pred_y1, pred_x2, pred_y2 = xywh_to_xyxy(pred_xywh)
    tgt_x1, tgt_y1, tgt_x2, tgt_y2 = xywh_to_xyxy(target_xywh)

    inter_x1 = torch.maximum(pred_x1, tgt_x1)
    inter_y1 = torch.maximum(pred_y1, tgt_y1)
    inter_x2 = torch.minimum(pred_x2, tgt_x2)
    inter_y2 = torch.minimum(pred_y2, tgt_y2)
    inter_w = (inter_x2 - inter_x1).clamp(min=0.0)
    inter_h = (inter_y2 - inter_y1).clamp(min=0.0)
    inter_area = inter_w * inter_h

    pred_area = (pred_x2 - pred_x1).clamp(min=0.0) * (pred_y2 - pred_y1).clamp(min=0.0)
    tgt_area = (tgt_x2 - tgt_x1).clamp(min=0.0) * (tgt_y2 - tgt_y1).clamp(min=0.0)
    union_area = (pred_area + tgt_area - inter_area).clamp(min=1e-8)
    return inter_area / union_area


def build_run_dir(output_root: Path, run_name: str) -> Path:
    base_name = run_name.strip() if run_name and run_name.strip() else datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = output_root / base_name
    if not run_dir.exists():
        run_dir.mkdir(parents=True, exist_ok=False)
        return run_dir

    # Avoid collisions for repeated launches in the same second.
    suffix = 1
    while True:
        candidate = output_root / f"{base_name}_{suffix:02d}"
        if not candidate.exists():
            candidate.mkdir(parents=True, exist_ok=False)
            return candidate
        suffix += 1


def save_iou_cdf_plot(ious: torch.Tensor, path: Path) -> None:
    sorted_ious, _ = torch.sort(ious)
    n = sorted_ious.numel()
    y = torch.arange(1, n + 1, dtype=torch.float32) / n
    plt.figure(figsize=(8, 5))
    plt.plot(sorted_ious.tolist(), y.tolist(), linewidth=2)
    plt.xlim(0.0, 1.0)
    plt.ylim(0.0, 1.0)
    plt.xlabel("IoU")
    plt.ylabel("Cumulative Probability")
    plt.title("IoU CDF")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def save_success_curve_plot(ious: torch.Tensor, path: Path, num_points: int = 101) -> None:
    thresholds = torch.linspace(0.0, 1.0, steps=num_points)
    success_rates = [(ious >= t).to(torch.float32).mean().item() for t in thresholds]
    plt.figure(figsize=(8, 5))
    plt.plot(thresholds.tolist(), success_rates, linewidth=2)
    plt.xlim(0.0, 1.0)
    plt.ylim(0.0, 1.0)
    plt.xlabel("IoU Threshold")
    plt.ylabel("Success Rate")
    plt.title("Success Rate vs IoU Threshold")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def evaluate_localization(
    model: torch.nn.Module,
    loader: DataLoader,
    criterion: torch.nn.Module,
    device: torch.device,
    success_iou_thresholds: list[float],
) -> dict[str, Any]:
    model.eval()
    total_samples = 0
    total_loss = 0.0
    total_mae = 0.0
    total_mse = 0.0
    iou_batches: list[torch.Tensor] = []
    center_error_l2_batches: list[torch.Tensor] = []
    size_abs_error_batches: list[torch.Tensor] = []

    progress = tqdm(loader, desc="Evaluating", leave=False)
    with torch.no_grad():
        for images, targets in progress:
            images = images.to(device)
            targets = targets.to(device)
            preds = model(images)

            batch_size = images.size(0)
            loss = criterion(preds, targets)
            mae_sum = F.l1_loss(preds, targets, reduction="sum")
            mse_sum = F.mse_loss(preds, targets, reduction="sum")
            ious = bbox_iou(preds, targets)
            center_errors_l2 = torch.sqrt(torch.sum((preds[:, :2] - targets[:, :2]) ** 2, dim=1))
            size_abs_error = torch.abs(preds[:, 2:] - targets[:, 2:])

            total_samples += batch_size
            total_loss += loss.item() * batch_size
            total_mae += mae_sum.item()
            total_mse += mse_sum.item()
            iou_batches.append(ious.detach().cpu())
            center_error_l2_batches.append(center_errors_l2.detach().cpu())
            size_abs_error_batches.append(size_abs_error.detach().cpu())

            progress.set_postfix(loss=f"{loss.item():.4f}")

    if total_samples == 0:
        raise ValueError("Evaluation loader produced zero samples.")

    ious = torch.cat(iou_batches, dim=0)
    center_error_l2 = torch.cat(center_error_l2_batches, dim=0)
    size_abs_error = torch.cat(size_abs_error_batches, dim=0)
    denom = total_samples * 4

    metrics: dict[str, Any] = {
        "num_samples": float(total_samples),
        "loss": total_loss / total_samples,
        "mae": total_mae / denom,
        "mse": total_mse / denom,
        "mean_iou": ious.mean().item(),
        "median_iou": ious.median().item(),
        "mean_center_error_l2": center_error_l2.mean().item(),
        "median_center_error_l2": center_error_l2.median().item(),
        "mean_size_abs_error_w": size_abs_error[:, 0].mean().item(),
        "mean_size_abs_error_h": size_abs_error[:, 1].mean().item(),
    }
    for thr in success_iou_thresholds:
        key = f"success_rate_iou_{thr:.2f}".replace(".", "p")
        metrics[key] = (ious >= thr).to(torch.float32).mean().item()

    return {"metrics": metrics, "ious": ious}


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate PyTorch localization model.")
    parser.add_argument(
        "--config",
        type=Path,
        default=THIS_DIR / "config.yaml",
        help="Path to YAML config file.",
    )
    args = parser.parse_args()
    cfg = load_config(args.config)

    data_dir = resolve_project_path(cfg.get("data_dir", "data"), PROJECT_ROOT)
    checkpoint = resolve_project_path(
        cfg.get("checkpoint", str(Path("deep_learning") / "checkpoints" / "model_best.pt")),
        PROJECT_ROOT,
    )
    if checkpoint.is_dir():
        checkpoint = checkpoint / "model.pt"
    split = str(cfg.get("split", "trainval"))
    image_size = int(cfg.get("image_size", 224))
    batch_size = int(cfg.get("batch_size", 32))
    num_workers = resolve_num_workers(int(cfg.get("num_workers", 2)))
    device_cfg = str(cfg.get("device", "auto")).lower()

    output_root = resolve_project_path(
        cfg.get("output_root", str(Path("deep_learning") / "eval" / "runs")),
        PROJECT_ROOT,
    )
    run_name_value = cfg.get("run_name", "")
    run_name = run_name_value if isinstance(run_name_value, str) else ""

    loc_cfg = cfg.get("localization_metrics", {})
    if not isinstance(loc_cfg, dict):
        raise ValueError("localization_metrics must be a mapping")
    success_iou_thresholds = loc_cfg.get("success_iou_thresholds", [0.5, 0.75, 0.9])
    if not isinstance(success_iou_thresholds, list) or not success_iou_thresholds:
        raise ValueError("localization_metrics.success_iou_thresholds must be a non-empty list")
    success_iou_thresholds = [float(x) for x in success_iou_thresholds]
    plots_subdir = str(loc_cfg.get("plots_subdir", "plots"))

    if device_cfg == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("device is set to cuda, but CUDA is not available.")
        device = torch.device("cuda")
    elif device_cfg == "cpu":
        device = torch.device("cpu")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    loss_cfg = cfg.get("loss", {})
    if not isinstance(loss_cfg, dict):
        raise ValueError("loss must be a mapping")
    loss_name = str(loss_cfg.get("name", "smooth_l1"))
    loss_params = loss_cfg.get("params", {})
    if not isinstance(loss_params, dict):
        raise ValueError("loss.params must be a mapping")
    criterion = build_loss(loss_name, loss_params)

    split_file = resolve_split_file(data_dir, split)
    dataset = OxfordPetBBoxDataset(data_dir, split_file, image_size=image_size)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
    )

    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint.resolve()}")
    model = build_model(pretrained=False, freeze_backbone=False, apply_sigmoid=True).to(device)
    checkpoint_data = torch.load(checkpoint, map_location=device)
    if isinstance(checkpoint_data, dict) and "model_state_dict" in checkpoint_data:
        state_dict = checkpoint_data["model_state_dict"]
    else:
        state_dict = checkpoint_data
    model.load_state_dict(state_dict)

    run_dir = build_run_dir(output_root, run_name)
    plots_dir = run_dir / plots_subdir
    plots_dir.mkdir(parents=True, exist_ok=True)

    result = evaluate_localization(
        model=model,
        loader=loader,
        criterion=criterion,
        device=device,
        success_iou_thresholds=success_iou_thresholds,
    )
    metrics: dict[str, Any] = result["metrics"]
    ious: torch.Tensor = result["ious"]

    save_iou_cdf_plot(ious, plots_dir / "iou_cdf.png")
    save_success_curve_plot(ious, plots_dir / "success_vs_iou_threshold.png")

    config_used: dict[str, Any] = {
        **cfg,
        "data_dir": str(data_dir),
        "checkpoint": str(checkpoint),
        "split_file": str(split_file),
        "output_root": str(output_root),
        "run_name": run_dir.name,
        "run_dir": str(run_dir),
    }

    print(f"Run dir: {run_dir}")
    print(f"Device: {device}")
    print(f"Checkpoint: {checkpoint}")
    print(f"Split file: {split_file}")
    print(f"Samples: {int(metrics['num_samples'])}")
    print(f"Loss ({loss_name}): {metrics['loss']:.6f}")
    print(f"MAE: {metrics['mae']:.6f}")
    print(f"MSE: {metrics['mse']:.6f}")
    print(f"Mean IoU: {metrics['mean_iou']:.6f}")
    print(f"Median IoU: {metrics['median_iou']:.6f}")
    print(f"Mean center error (L2): {metrics['mean_center_error_l2']:.6f}")
    print(f"Median center error (L2): {metrics['median_center_error_l2']:.6f}")
    print(f"Mean |width error|: {metrics['mean_size_abs_error_w']:.6f}")
    print(f"Mean |height error|: {metrics['mean_size_abs_error_h']:.6f}")
    for thr in success_iou_thresholds:
        key = f"success_rate_iou_{thr:.2f}".replace(".", "p")
        print(f"Success rate IoU>={thr:.2f}: {metrics[key]:.6f}")

    metrics_path = run_dir / "metrics.yaml"
    config_path = run_dir / "config_used.yaml"
    summary_path = run_dir / "summary.txt"
    with metrics_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(metrics, f, sort_keys=True)
    with config_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(config_used, f, sort_keys=False)
    with summary_path.open("w", encoding="utf-8") as f:
        for key, value in metrics.items():
            f.write(f"{key}: {value}\n")
    print(f"Saved metrics to: {metrics_path}")
    print(f"Saved config used to: {config_path}")
    print(f"Saved summary to: {summary_path}")
    print(f"Saved plots to: {plots_dir}")


if __name__ == "__main__":
    main()
