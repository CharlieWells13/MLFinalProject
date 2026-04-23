import argparse
import sys
from pathlib import Path
from typing import Any

import torch
from torch import nn
import yaml


THIS_DIR = Path(__file__).resolve().parent
DEEP_LEARNING_DIR = THIS_DIR.parent.parent
if str(DEEP_LEARNING_DIR) not in sys.path:
    sys.path.append(str(DEEP_LEARNING_DIR))

from models.model_pytorch import build_model  # noqa: E402
from train.pytorch.data_pipeline import create_dataloaders  # noqa: E402


def load_config(config_path: Path) -> dict[str, Any]:
    with config_path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    if not isinstance(config, dict):
        raise ValueError(f"Config must be a mapping: {config_path}")
    return config


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


def build_loss(name: str, params: dict[str, Any]) -> nn.Module:
    loss_name = name.lower()
    if loss_name == "smooth_l1":
        return nn.SmoothL1Loss(**params)
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
) -> float:
    is_train = optimizer is not None
    model.train() if is_train else model.eval()
    running_loss = 0.0
    num_samples = 0

    for images, targets in loader:
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
    data_dir = Path(cfg.get("data_dir", "data"))
    epochs = int(cfg.get("epochs", 20))
    batch_size = int(cfg.get("batch_size", 32))
    lr = float(cfg.get("lr", 1e-3))
    weight_decay = float(cfg.get("weight_decay", 1e-4))
    image_size = int(cfg.get("image_size", 224))
    freeze_backbone = bool(cfg.get("freeze_backbone", False))
    num_workers = int(cfg.get("num_workers", 2))
    output = Path(cfg.get("output", str(Path("deep_learning") / "checkpoints" / "model_pytorch_best.pt")))

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
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(pretrained=False, freeze_backbone=freeze_backbone, apply_sigmoid=True).to(device)
    criterion = build_loss(loss_name, loss_params)
    optimizer = build_optimizer(
        optimizer_name,
        [p for p in model.parameters() if p.requires_grad],
        lr,
        weight_decay,
        optimizer_params,
    )

    best_val_loss = float("inf")
    output.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, epochs + 1):
        train_loss = run_epoch(model, train_loader, criterion, optimizer, device)
        val_loss = run_epoch(model, val_loader, criterion, optimizer=None, device=device)
        print(f"Epoch {epoch:03d}/{epochs} | train_loss={train_loss:.6f} | val_loss={val_loss:.6f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": val_loss,
                },
                output,
            )
            print(f"Saved best checkpoint to {output}")


if __name__ == "__main__":
    main()
