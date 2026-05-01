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
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader, Subset


THIS_DIR = Path(__file__).resolve().parent
DEEP_LEARNING_DIR = THIS_DIR.parent
PROJECT_ROOT = DEEP_LEARNING_DIR.parent
if str(DEEP_LEARNING_DIR) not in sys.path:
    sys.path.append(str(DEEP_LEARNING_DIR))

from models.model_pretrained import build_model as build_model_pretrained  # noqa: E402
try:
    # Works when invoked as a package path from repo root.
    from train.data_pipeline import OxfordPetBBoxDataset  # type: ignore  # noqa: E402
except ModuleNotFoundError:
    # Works when invoked directly from deep_learning/train.
    from data_pipeline import OxfordPetBBoxDataset  # type: ignore  # noqa: E402


# Fixed training settings (non-listed fields remain hard-coded)
IMAGES_NPY = Path("preprocessed_data") / "images.npy"
BBOXES_NPY = Path("preprocessed_data") / "bboxes.npy"
WEIGHT_DECAY = 1e-4
FREEZE_BACKBONE = False
NUM_WORKERS = 2
EARLY_STOPPING_PATIENCE = 10
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1
SPLIT_SEED = 42
CHECKPOINT_ROOT = Path("deep_learning") / "checkpoints"
CHECKPOINT_RUN_NAME = "run"
OPTIMIZER_NAME = "adamw"
OPTIMIZER_PARAMS: dict[str, Any] = {}
LOSS_NAME = "smooth_l1_iou"
LOSS_PARAMS: dict[str, Any] = {"smooth_l1_weight": 1.0, "iou_weight": 1.0}
NUM_FOLDS = 5
TRAIN_INDICES_PATH = Path("preprocessed_data") / "train_indices.npy"


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
    run_name: str,
) -> Path:
    folder_name = run_name.strip() if run_name.strip() else f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir = checkpoint_root / folder_name
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def format_duration(seconds: float) -> str:
    total = int(round(seconds))
    hours, rem = divmod(total, 3600)
    minutes, secs = divmod(rem, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def save_loss_curve(
    history: list[dict[str, float]],
    output_path: Path,
    best_epoch: int,
) -> None:
    if not history:
        raise ValueError("Cannot generate loss curve: history is empty.")

    epochs = [int(item["epoch"]) for item in history]
    train_losses = [float(item["train_loss"]) for item in history]
    val_losses = [float(item["val_loss"]) for item in history]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, train_losses, marker="o", linewidth=2, label="Train Loss")
    ax.plot(epochs, val_losses, marker="o", linewidth=2, label="Validation Loss")
    if best_epoch > 0:
        ax.axvline(best_epoch, linestyle="--", linewidth=1.5, alpha=0.7, label=f"Best Epoch ({best_epoch})")
    ax.set_title("Training vs Validation Loss by Epoch")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def save_cv_summary_curve(
    fold_best_losses: list[float],
    output_path: Path,
) -> None:
    folds = list(range(1, len(fold_best_losses) + 1))
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(folds, fold_best_losses, marker="o", linewidth=2)
    ax.set_title("Cross-Validation: Best Validation Loss per Fold")
    ax.set_xlabel("Fold")
    ax.set_ylabel("Best Validation Loss")
    ax.set_xticks(folds)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def save_fold_performance_comparison(
    fold_best_losses: list[float],
    fold_final_val_losses: list[float],
    output_path: Path,
) -> None:
    folds = list(range(1, len(fold_best_losses) + 1))
    x = np.arange(len(folds))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(x - width / 2, fold_best_losses, width=width, label="Best Val Loss")
    ax.bar(x + width / 2, fold_final_val_losses, width=width, label="Final Val Loss")
    ax.set_title("Fold Performance Comparison")
    ax.set_xlabel("Fold")
    ax.set_ylabel("Validation Loss")
    ax.set_xticks(x)
    ax.set_xticklabels([str(f) for f in folds])
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def build_fold_dataloaders(
    dataset: OxfordPetBBoxDataset,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    batch_size: int,
    num_workers: int,
    seed: int,
) -> tuple[DataLoader, DataLoader]:
    train_ds = Subset(dataset, train_idx.tolist())
    val_ds = Subset(dataset, val_idx.tolist())
    loader_generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        generator=loader_generator,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    return train_loader, val_loader


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
    images_npy_cfg = cfg.get("images_npy", str(IMAGES_NPY))
    bboxes_npy_cfg = cfg.get("bboxes_npy", str(BBOXES_NPY))
    train_indices_cfg = cfg.get("train_indices_path", str(TRAIN_INDICES_PATH))
    checkpoint_root_cfg = cfg.get("checkpoint_root", str(CHECKPOINT_ROOT))
    checkpoint_run_name_cfg = cfg.get("checkpoint_run_name", CHECKPOINT_RUN_NAME)
    num_folds_cfg = cfg.get("num_folds", cfg.get("num_foflds", NUM_FOLDS))

    images_npy = resolve_project_path(images_npy_cfg, PROJECT_ROOT)
    bboxes_npy = resolve_project_path(bboxes_npy_cfg, PROJECT_ROOT)
    epochs = int(cfg.get("epochs", 20))
    batch_size = int(cfg.get("batch_size", 32))
    lr = float(cfg.get("lr", 1e-3))
    weight_decay = WEIGHT_DECAY
    freeze_backbone = FREEZE_BACKBONE
    num_workers = resolve_num_workers(NUM_WORKERS)
    early_stopping_patience = EARLY_STOPPING_PATIENCE
    train_ratio = TRAIN_RATIO
    val_ratio = VAL_RATIO
    test_ratio = TEST_RATIO
    split_seed = SPLIT_SEED
    checkpoint_root = resolve_project_path(checkpoint_root_cfg, PROJECT_ROOT)
    checkpoint_run_name = str(checkpoint_run_name_cfg)
    num_folds = int(num_folds_cfg)
    if num_folds < 2:
        raise ValueError("num_folds must be at least 2.")
    optimizer_name = OPTIMIZER_NAME
    optimizer_params = dict(OPTIMIZER_PARAMS)
    loss_name = LOSS_NAME
    loss_params = dict(LOSS_PARAMS)
    train_indices_path = resolve_project_path(train_indices_cfg, PROJECT_ROOT)
    train_indices = np.load(train_indices_path).astype(np.int64, copy=False)

    dataset = OxfordPetBBoxDataset(str(images_npy), str(bboxes_npy))
    dataset_size = len(dataset)
    if train_indices.ndim != 1 or len(train_indices) == 0:
        raise ValueError("train_indices.npy must be a non-empty 1D array.")
    if int(train_indices.min()) < 0 or int(train_indices.max()) >= dataset_size:
        raise ValueError(
            f"train_indices.npy contains out-of-range values for dataset size {dataset_size}: "
            f"min={int(train_indices.min())}, max={int(train_indices.max())}"
        )
    if len(np.unique(train_indices)) != len(train_indices):
        raise ValueError("train_indices.npy contains duplicate indices.")
    fold_indices = np.array_split(train_indices, num_folds)

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    run_dir = build_run_dir(checkpoint_root, checkpoint_run_name)
    log = make_logger(run_dir / "training_log.txt")

    config_used = {
        **cfg,
        "config_path": str(args.config.resolve()),
        "images_npy": str(images_npy),
        "bboxes_npy": str(bboxes_npy),
        "split_ratios": {
            "train": train_ratio,
            "val": val_ratio,
            "test": test_ratio,
        },
        "split_seed": split_seed,
        "checkpoint_root": str(checkpoint_root),
        "checkpoint_run_name": checkpoint_run_name,
        "run_dir": str(run_dir),
        "device": str(device),
        "use_pretrained_backbone": True,
        "resolved_num_workers": num_workers,
        "train_indices_path": str(train_indices_path),
        "num_folds": num_folds,
    }
    save_yaml(config_used, run_dir / "config_used.yaml")
    log(f"Checkpoint run dir: {run_dir}")
    log(f"Cross-validation folds: {num_folds}")
    train_start_time = time.perf_counter()
    fold_best_losses: list[float] = []
    fold_final_val_losses: list[float] = []
    overall_best_val_loss = float("inf")
    overall_best_checkpoint: dict[str, Any] | None = None
    overall_best_fold = 0
    overall_best_epoch = 0

    for fold_idx in range(num_folds):
        val_idx = fold_indices[fold_idx]
        train_parts = [fold_indices[i] for i in range(num_folds) if i != fold_idx]
        train_idx = np.concatenate(train_parts, axis=0)
        fold_number = fold_idx + 1
        fold_seed = split_seed + fold_idx
        train_loader, val_loader = build_fold_dataloaders(
            dataset=dataset,
            train_idx=train_idx,
            val_idx=val_idx,
            batch_size=batch_size,
            num_workers=num_workers,
            seed=fold_seed,
        )

        fold_model_path = run_dir / f"fold_{fold_number}.pt"

        model = build_model_pretrained(pretrained=True, freeze_backbone=freeze_backbone, apply_sigmoid=True).to(device)
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
        log(
            f"[Fold {fold_number}/{num_folds}] "
            f"train_samples={len(train_idx)} val_samples={len(val_idx)}"
        )

        for epoch in range(1, epochs + 1):
            train_loss = run_epoch(model, train_loader, criterion, optimizer, device, epoch, epochs)
            val_loss = run_epoch(model, val_loader, criterion, optimizer=None, device=device, epoch=epoch, epochs=epochs)
            log(
                f"[Fold {fold_number}/{num_folds}] "
                f"Epoch {epoch:03d}/{epochs} | train_loss={train_loss:.6f} | val_loss={val_loss:.6f}"
            )
            history.append({"epoch": float(epoch), "train_loss": train_loss, "val_loss": val_loss})

            current_checkpoint = {
                "fold": fold_number,
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
                saved_best_path = save_model_checkpoint(current_checkpoint, fold_model_path)
                log(f"[Fold {fold_number}/{num_folds}] Updated best checkpoint: {saved_best_path}")
                if val_loss < overall_best_val_loss:
                    overall_best_val_loss = val_loss
                    overall_best_checkpoint = current_checkpoint
                    overall_best_fold = fold_number
                    overall_best_epoch = epoch
            else:
                epochs_without_improvement += 1
                log(
                    f"[Fold {fold_number}/{num_folds}] "
                    f"No validation improvement for {epochs_without_improvement} epoch(s) "
                    f"(patience={early_stopping_patience})."
                )
                if epochs_without_improvement >= early_stopping_patience:
                    log(
                        f"[Fold {fold_number}/{num_folds}] Early stopping at epoch {epoch}: "
                        f"val_loss did not improve for {early_stopping_patience} epochs."
                    )
                    break

        if last_checkpoint is None:
            raise RuntimeError(f"No training epochs were executed for fold {fold_number}.")

        fold_curve_path = run_dir / f"fold_{fold_number}_train_vs_val_loss.png"
        save_loss_curve(history, fold_curve_path, best_epoch)
        fold_best_losses.append(best_val_loss)
        fold_final_val_losses.append(float(last_checkpoint["val_loss"]))
        log(f"[Fold {fold_number}/{num_folds}] Saved best checkpoint: {fold_model_path}")
        log(f"[Fold {fold_number}/{num_folds}] Saved loss curve: {fold_curve_path}")
        log(
            f"[Fold {fold_number}/{num_folds}] "
            f"best_epoch={best_epoch} best_val_loss={best_val_loss:.6f}"
        )

    if overall_best_checkpoint is None:
        raise RuntimeError("No best checkpoint produced across folds.")
    best_overall_path = run_dir / "best.pt"
    saved_best_overall_path = save_model_checkpoint(overall_best_checkpoint, best_overall_path)

    total_train_seconds = time.perf_counter() - train_start_time
    cv_curve_path = run_dir / "cv_best_val_loss.png"
    save_cv_summary_curve(fold_best_losses, cv_curve_path)
    fold_compare_path = run_dir / "fold_performance_comparison.png"
    save_fold_performance_comparison(fold_best_losses, fold_final_val_losses, fold_compare_path)
    mean_best_val_loss = float(np.mean(np.array(fold_best_losses, dtype=np.float64)))
    log(f"Saved CV summary curve: {cv_curve_path}")
    log(f"Saved fold performance comparison: {fold_compare_path}")
    log(
        f"Saved overall best checkpoint: {saved_best_overall_path} "
        f"(fold={overall_best_fold}, epoch={overall_best_epoch}, val_loss={overall_best_val_loss:.6f})"
    )
    log(f"Cross-validation mean best val loss: {mean_best_val_loss:.6f}")
    log(
        "Total training time: "
        f"{format_duration(total_train_seconds)} "
        f"({total_train_seconds:.2f} seconds)"
    )


if __name__ == "__main__":
    main()
