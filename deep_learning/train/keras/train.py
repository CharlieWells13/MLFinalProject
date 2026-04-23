import argparse
import sys
from pathlib import Path
from typing import Any

import tensorflow as tf
import yaml


THIS_DIR = Path(__file__).resolve().parent
DEEP_LEARNING_DIR = THIS_DIR.parent.parent
if str(DEEP_LEARNING_DIR) not in sys.path:
    sys.path.append(str(DEEP_LEARNING_DIR))

from models.model_keras import build_model  # noqa: E402
from train.keras.data_pipeline import create_datasets  # noqa: E402


def load_config(config_path: Path) -> dict[str, Any]:
    with config_path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    if not isinstance(config, dict):
        raise ValueError(f"Config must be a mapping: {config_path}")
    return config


def build_optimizer(name: str, lr: float, weight_decay: float, params: dict[str, Any]) -> tf.keras.optimizers.Optimizer:
    optimizer_name = name.lower()
    if optimizer_name == "adamw":
        return tf.keras.optimizers.AdamW(learning_rate=lr, weight_decay=weight_decay, **params)
    if optimizer_name == "adam":
        return tf.keras.optimizers.Adam(learning_rate=lr, **params)
    if optimizer_name == "sgd":
        return tf.keras.optimizers.SGD(learning_rate=lr, **params)
    raise ValueError(f"Unsupported Keras optimizer: {name}")


def build_loss(name: str, params: dict[str, Any]):
    loss_name = name.lower()
    if loss_name == "huber":
        return tf.keras.losses.Huber(**params)
    if loss_name == "mse":
        return tf.keras.losses.MeanSquaredError(**params)
    if loss_name == "mae":
        return tf.keras.losses.MeanAbsoluteError(**params)
    raise ValueError(f"Unsupported Keras loss: {name}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Keras ResNet-18 for bounding box regression.")
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
    output = Path(cfg.get("output", str(Path("deep_learning") / "checkpoints" / "model_keras_best.keras")))

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
    loss_name = str(loss_cfg.get("name", "huber"))
    loss_params = loss_cfg.get("params", {})
    if not isinstance(loss_params, dict):
        raise ValueError("loss.params must be a mapping")

    train_ds, val_ds = create_datasets(
        data_dir=data_dir,
        image_size=image_size,
        batch_size=batch_size,
    )

    model = build_model(
        input_shape=(image_size, image_size, 3),
        apply_sigmoid=True,
        pretrained=False,
        freeze_backbone=freeze_backbone,
    )
    optimizer = build_optimizer(optimizer_name, lr, weight_decay, optimizer_params)
    model.compile(optimizer=optimizer, loss=build_loss(loss_name, loss_params))

    output.parent.mkdir(parents=True, exist_ok=True)
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(output),
            monitor="val_loss",
            mode="min",
            save_best_only=True,
        )
    ]

    model.fit(train_ds, validation_data=val_ds, epochs=epochs, callbacks=callbacks)
    print(f"Best model path: {output}")


if __name__ == "__main__":
    main()
