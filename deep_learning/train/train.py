import argparse
from datetime import datetime
import multiprocessing as mp
import time
import sys
from pathlib import Path
from typing import Any

import torch
from torch import nn
from tqdm import tqdm
import yaml


THIS_DIR = Path(__file__).resolve().parent
DEEP_LEARNING_DIR = THIS_DIR.parent
PROJECT_ROOT = DEEP_LEARNING_DIR.parent
if str(DEEP_LEARNING_DIR) not in sys.path:
    sys.path.append(str(DEEP_LEARNING_DIR))

from models.model_scratch import build_model as build_model_scratch  # noqa: E402
from models.model_pretrained import build_model as build_model_pretrained  # noqa: E402
try:
    # Works when invoked as a package path from repo root.
    from train.data_pipeline import create_dataloaders  # type: ignore  # noqa: E402
except ModuleNotFoundError:
    # Works when invoked directly from deep_learning/train.
    from data_pipeline import create_dataloaders  # type: ignore  # noqa: E402


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
        # Some restricted Windows environments block worker IPC creation.
        queue = mp.get_context("spawn").Queue()
        queue.close()
        queue.join_thread()
        return requested_workers
    except (OSError, PermissionError):
        print("Warning: multiprocessing workers are not permitted here. Falling back to num_workers=0.")
        return 0


def save_yaml(data: dict[str, Any], path: Path) -> None:
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False)


def make_logger(log_path: Path):
    log_path.parent.mkdir(parents=True, exist_ok=True)

    def _log(message: str) -> None:
        print(message)
        timestamp = datetime.now().isoformat(timespec="seconds")
        with log_path.open("a", encoding="utf-8") as f:
            f.write(f"[{timestamp}] {message}\n")

    return _log


def build_run_dir(
    checkpoint_root: Path,
    checkpoint_prefix: str,
    run_name: str,
) -> Path:
    folder_name = run_name.strip() if run_name.strip() else f"{checkpoint_prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir = checkpoint_root / folder_name
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def format_duration(seconds: float) -> str:
    total = int(round(seconds))
    hours, rem = divmod(total, 3600)
    minutes, secs = divmod(rem, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def save_model_checkpoint(
    checkpoint: dict[str, Any],
    model_path: Path,
) -> Path:
    try:
        torch.save(checkpoint, model_path)
        return model_path
    except (OSError, PermissionError, RuntimeError) as exc:
        fallback_dir = THIS_DIR / "checkpoints" / model_path.parent.name
        fallback_dir.mkdir(parents=True, exist_ok=True)
        fallback_model_path = fallback_dir / model_path.name
        print(f"Warning: failed to write checkpoint to {model_path}: {exc}")
        print(f"Saving checkpoint to fallback path: {fallback_model_path}")
        torch.save(checkpoint, fallback_model_path)
        return fallback_model_path


def build_optimizer(
    name: str,
    trainable_params,
    lr: float,
    weight_decay: float,
    params: dict[str, Any],
) -> torch.optim.Optimizer:
    optimizer_name = name.lower()
    if optimizer_name == "adamw":
        return torch.optim.AdamW(trainable_params, lr=lr, weight_decay=weight_decay, **params)
    if optimizer_name == "adam":
        return torch.optim.Adam(trainable_params, lr=lr, weight_decay=weight_decay, **params)
    if optimizer_name == "sgd":
        return torch.optim.SGD(trainable_params, lr=lr, weight_decay=weight_decay, **params)
    raise ValueError(f"Unsupported PyTorch optimizer: {name}")


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


class SmoothL1IoULoss(nn.Module):
    def __init__(
        self,
        smooth_l1_weight: float = 1.0,
        iou_weight: float = 1.0,
        **smooth_l1_kwargs: Any,
    ):
        super().__init__()
        self.smooth_l1_weight = smooth_l1_weight
        self.iou_weight = iou_weight
        self.smooth_l1 = nn.SmoothL1Loss(**smooth_l1_kwargs)

    def forward(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        reg_loss = self.smooth_l1(preds, targets)
        iou_loss = 1.0 - bbox_iou(preds, targets).mean()
        return self.smooth_l1_weight * reg_loss + self.iou_weight * iou_loss


def build_loss(name: str, params: dict[str, Any]) -> nn.Module:
    loss_name = name.lower()
    if loss_name == "smooth_l1":
        return nn.SmoothL1Loss(**params)
    if loss_name == "smooth_l1_iou":
        params_copy = dict(params)
        smooth_l1_weight = float(params_copy.pop("smooth_l1_weight", params_copy.pop("l1_weight", 1.0)))
        iou_weight = float(params_copy.pop("iou_weight", 1.0))
        return SmoothL1IoULoss(smooth_l1_weight=smooth_l1_weight, iou_weight=iou_weight, **params_copy)
    if loss_name == "huber":
        return nn.HuberLoss(**params)
    if loss_name == "mse":
        return nn.MSELoss(**params)
    if loss_name == "l1":
        return nn.L1Loss(**params)
    raise ValueError(f"Unsupported PyTorch loss: {name}")


def run_epoch(
    model: nn.Module,
    loader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    epoch: int,
    epochs: int,
) -> float:
    is_train = optimizer is not None
    phase = "train" if is_train else "val"
    model.train() if is_train else model.eval()
    running_loss = 0.0
    num_samples = 0

    progress = tqdm(loader, desc=f"Epoch {epoch:03d}/{epochs} [{phase}]", leave=False)
    for images, targets in progress:
        images = images.to(device)
        targets = targets.to(device)

        with torch.set_grad_enabled(is_train):
            preds = model(images)
            loss = criterion(preds, targets)

        if is_train:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

        batch_size = images.size(0)
        running_loss += loss.item() * batch_size
        num_samples += batch_size
        progress.set_postfix(loss=f"{loss.item():.4f}")

    return running_loss / max(1, num_samples)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train PyTorch ResNet-18 for bounding box regression.")
    parser.add_argument(
        "--config",
        type=Path,
        default=THIS_DIR / "config.yaml",
        help="Path to YAML config file.",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    data_dir = resolve_project_path(cfg.get("data_dir", "data"), PROJECT_ROOT)
    epochs = int(cfg.get("epochs", 20))
    batch_size = int(cfg.get("batch_size", 32))
    lr = float(cfg.get("lr", 1e-3))
    weight_decay = float(cfg.get("weight_decay", 1e-4))
    image_size = int(cfg.get("image_size", 224))
    freeze_backbone = bool(cfg.get("freeze_backbone", False))
    use_pretrained_backbone = bool(cfg.get("use_pretrained_backbone", False))
    num_workers = resolve_num_workers(int(cfg.get("num_workers", 2)))
    early_stopping_patience = int(cfg.get("early_stopping_patience", 5))
    train_split = resolve_project_path(
        cfg.get("train_split", str(Path("data") / "annotations" / "custom_split" / "train.txt")),
        PROJECT_ROOT,
    )
    val_split = resolve_project_path(
        cfg.get("val_split", str(Path("data") / "annotations" / "custom_split" / "val.txt")),
        PROJECT_ROOT,
    )
    checkpoint_root = resolve_project_path(
        cfg.get("checkpoint_root", str(Path("deep_learning") / "checkpoints")),
        PROJECT_ROOT,
    )
    checkpoint_prefix = str(cfg.get("checkpoint_prefix", "model_best"))
    checkpoint_run_name = str(cfg.get("checkpoint_run_name", ""))

    optimizer_cfg = cfg.get("optimizer", {})
    if not isinstance(optimizer_cfg, dict):
        raise ValueError("optimizer must be a mapping")
    optimizer_name = str(optimizer_cfg.get("name", "adamw"))
    optimizer_params = optimizer_cfg.get("params", {})
    if not isinstance(optimizer_params, dict):
        raise ValueError("optimizer.params must be a mapping")

    loss_cfg = cfg.get("loss", {})
    if not isinstance(loss_cfg, dict):
        raise ValueError("loss must be a mapping")
    loss_name = str(loss_cfg.get("name", "smooth_l1"))
    loss_params = loss_cfg.get("params", {})
    if not isinstance(loss_params, dict):
        raise ValueError("loss.params must be a mapping")

    train_loader, val_loader = create_dataloaders(
        data_dir=data_dir,
        image_size=image_size,
        batch_size=batch_size,
        num_workers=num_workers,
        train_split_file=train_split,
        val_split_file=val_split,
    )

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    if use_pretrained_backbone:
        model = build_model_pretrained(pretrained=True, freeze_backbone=freeze_backbone, apply_sigmoid=True).to(device)
    else:
        model = build_model_scratch(pretrained=False, freeze_backbone=freeze_backbone, apply_sigmoid=True).to(device)
    criterion = build_loss(loss_name, loss_params)
    optimizer = build_optimizer(
        optimizer_name,
        [p for p in model.parameters() if p.requires_grad],
        lr,
        weight_decay,
        optimizer_params,
    )

    best_val_loss = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0
    history: list[dict[str, float]] = []
    last_checkpoint: dict[str, Any] | None = None
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    run_dir = build_run_dir(checkpoint_root, checkpoint_prefix, checkpoint_run_name)
    best_model_path = run_dir / "best_model.pt"
    last_model_path = run_dir / "last_model.pt"
    log = make_logger(run_dir / "training_log.txt")

    config_used = {
        **cfg,
        "config_path": str(args.config.resolve()),
        "data_dir": str(data_dir),
        "train_split": str(train_split),
        "val_split": str(val_split),
        "checkpoint_root": str(checkpoint_root),
        "checkpoint_prefix": checkpoint_prefix,
        "checkpoint_run_name": checkpoint_run_name,
        "run_dir": str(run_dir),
        "device": str(device),
        "use_pretrained_backbone": use_pretrained_backbone,
        "resolved_num_workers": num_workers,
    }
    save_yaml(config_used, run_dir / "config_used.yaml")
    log(f"Checkpoint run dir: {run_dir}")
    train_start_time = time.perf_counter()

    for epoch in range(1, epochs + 1):
        train_loss = run_epoch(model, train_loader, criterion, optimizer, device, epoch, epochs)
        val_loss = run_epoch(model, val_loader, criterion, optimizer=None, device=device, epoch=epoch, epochs=epochs)
        log(f"Epoch {epoch:03d}/{epochs} | train_loss={train_loss:.6f} | val_loss={val_loss:.6f}")
        history.append({"epoch": float(epoch), "train_loss": train_loss, "val_loss": val_loss})

        current_checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "train_loss": train_loss,
            "val_loss": val_loss,
            "config_used": config_used,
        }
        last_checkpoint = current_checkpoint

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            epochs_without_improvement = 0
            saved_best_path = save_model_checkpoint(current_checkpoint, best_model_path)
            log(f"Updated best checkpoint: {saved_best_path}")
        else:
            epochs_without_improvement += 1
            log(
                f"No validation improvement for {epochs_without_improvement} epoch(s) "
                f"(patience={early_stopping_patience})."
            )
            if epochs_without_improvement >= early_stopping_patience:
                log(
                    f"Early stopping triggered at epoch {epoch}: "
                    f"val_loss did not improve for {early_stopping_patience} epochs."
                )
                break

    if last_checkpoint is None:
        raise RuntimeError("No training epochs were executed; cannot save last checkpoint.")

    saved_last_path = save_model_checkpoint(last_checkpoint, last_model_path)
    total_train_seconds = time.perf_counter() - train_start_time
    run_summary = {
        "completed_at": datetime.now().isoformat(timespec="seconds"),
        "total_train_seconds": total_train_seconds,
        "total_train_duration_hms": format_duration(total_train_seconds),
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "final_epoch": int(last_checkpoint["epoch"]),
        "final_train_loss": float(last_checkpoint["train_loss"]),
        "final_val_loss": float(last_checkpoint["val_loss"]),
        "best_model_path": str(best_model_path),
        "last_model_path": str(saved_last_path),
        "history": history,
    }
    save_yaml(run_summary, run_dir / "run_summary.yaml")
    log(f"Saved last checkpoint: {saved_last_path}")
    log(f"Saved run summary: {run_dir / 'run_summary.yaml'}")
    log(
        "Total training time: "
        f"{run_summary['total_train_duration_hms']} "
        f"({run_summary['total_train_seconds']:.2f} seconds)"
    )


if __name__ == "__main__":
    main()
