import argparse
import platform
import sys
import xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml


THIS_DIR = Path(__file__).resolve().parent
DEEP_LEARNING_DIR = THIS_DIR.parent
PROJECT_ROOT = DEEP_LEARNING_DIR.parent
DEFAULT_CONFIG_PATH = THIS_DIR / "config.yaml"
DEFAULT_IMAGES_NPY = PROJECT_ROOT / "preprocessed_data" / "images.npy"
DEFAULT_SPLIT_NPY = PROJECT_ROOT / "preprocessed_data" / "test_indices.npy"

if str(DEEP_LEARNING_DIR) not in sys.path:
    sys.path.append(str(DEEP_LEARNING_DIR))

from models.model_scratch import build_model as build_model_scratch  # noqa: E402
from models.model_pretrained import build_model as build_model_pretrained  # noqa: E402


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Config must be a mapping: {path}")
    return data


def resolve_project_path(path_value: str | Path, project_root: Path) -> Path:
    path = Path(path_value).expanduser()
    if path.is_absolute():
        return path
    return (project_root / path).resolve()


def parse_split_indices_file(split_file: Path, dataset_size: int) -> list[int]:
    if not split_file.is_file():
        raise FileNotFoundError(f"Split indices file not found: {split_file}")
    indices = np.load(split_file)
    if indices.ndim != 1:
        raise ValueError(f"Split indices must be a 1D array, got shape={indices.shape}")
    if len(indices) == 0:
        return []
    if not np.issubdtype(indices.dtype, np.integer):
        raise ValueError(f"Split indices must contain integers, got dtype={indices.dtype}")

    min_idx = int(indices.min())
    max_idx = int(indices.max())
    if min_idx < 0 or max_idx >= dataset_size:
        raise ValueError(
            f"Split indices out of range for dataset size {dataset_size}: min={min_idx}, max={max_idx}"
        )

    unique_count = len(np.unique(indices))
    if unique_count != len(indices):
        raise ValueError(f"Split indices contain duplicates: {len(indices) - unique_count} repeated entries.")

    return indices.astype(np.int64, copy=False).tolist()


def resolve_checkpoint(path_value: str | Path) -> Path:
    checkpoint = Path(path_value).expanduser()
    if checkpoint.is_dir():
        candidates = [
            checkpoint / "best_model.pt",
            checkpoint / "model.pt",
            checkpoint / "last_model.pt",
        ]
        checkpoint = next((p for p in candidates if p.is_file()), checkpoint / "best_model.pt")
    return checkpoint


def to_xyxy_pixels(xywh_norm: torch.Tensor, width: int, height: int) -> tuple[int, int, int, int]:
    x_center = float(xywh_norm[0].clamp(0.0, 1.0))
    y_center = float(xywh_norm[1].clamp(0.0, 1.0))
    box_w = float(xywh_norm[2].clamp(0.0, 1.0))
    box_h = float(xywh_norm[3].clamp(0.0, 1.0))

    x1 = (x_center - box_w / 2.0) * width
    y1 = (y_center - box_h / 2.0) * height
    x2 = (x_center + box_w / 2.0) * width
    y2 = (y_center + box_h / 2.0) * height

    xmin = max(1, min(width, int(round(x1))))
    ymin = max(1, min(height, int(round(y1))))
    xmax = max(1, min(width, int(round(x2))))
    ymax = max(1, min(height, int(round(y2))))

    if xmax < xmin:
        xmax = xmin
    if ymax < ymin:
        ymax = ymin
    return xmin, ymin, xmax, ymax


def xyxy_to_xywh(xmin: int, ymin: int, xmax: int, ymax: int) -> tuple[float, float, float, float]:
    box_w = max(0.0, float(xmax - xmin))
    box_h = max(0.0, float(ymax - ymin))
    return float(xmin), float(ymin), box_w, box_h


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Load a trained bbox model, run inference from preprocessed .npy data, and write one XML."
    )
    parser.add_argument("--checkpoint", type=str, default="", help="Checkpoint file or run directory.")
    parser.add_argument("--images-npy", type=str, default=None, help="Path to preprocessed images .npy file.")
    parser.add_argument("--split", type=str, default=None, help="Path to split indices .npy file.")
    parser.add_argument("--output-dir", type=str, default=None, help="Directory for the output XML file.")
    parser.add_argument("--image-size", type=int, default=None, help="Expected model input size (square).")
    parser.add_argument(
        "--device", type=str, default="auto", choices=("auto", "cpu", "cuda"), help="Inference device."
    )
    parser.add_argument("--use-pretrained-backbone", action="store_true")
    parser.add_argument("--use-scratch-backbone", action="store_true")
    parser.add_argument("--max-images", type=int, default=None, help="Optional limit (0 means no limit).")
    args = parser.parse_args()

    cfg: dict[str, Any] = load_yaml(DEFAULT_CONFIG_PATH)

    checkpoint_cfg = args.checkpoint or cfg.get("checkpoint")
    if not checkpoint_cfg:
        raise ValueError("checkpoint is required (via --checkpoint or config).")
    checkpoint = resolve_project_path(resolve_checkpoint(checkpoint_cfg), PROJECT_ROOT)

    images_npy = resolve_project_path(args.images_npy or cfg.get("images_npy", str(DEFAULT_IMAGES_NPY)), PROJECT_ROOT)
    split_path = resolve_project_path(args.split or cfg.get("split", str(DEFAULT_SPLIT_NPY)), PROJECT_ROOT)
    output_root = resolve_project_path(
        args.output_dir or cfg.get("output_root") or cfg.get("output_dir", "deep_learning/predict/runs"),
        PROJECT_ROOT,
    )

    run_name_cfg = str(cfg.get("run_name", "")).strip()
    run_name = run_name_cfg if run_name_cfg else datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_root / f"{run_name}.xml"
    image_size = int(args.image_size if args.image_size is not None else cfg.get("image_size", 224))
    max_images = int(args.max_images if args.max_images is not None else cfg.get("max_images", 0))
    max_images = None if max_images <= 0 else max_images

    if args.device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("device is set to cuda, but CUDA is not available.")
        device = torch.device("cuda")
    elif args.device == "cpu":
        device = torch.device("cpu")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint}")
    if not images_npy.is_file():
        raise FileNotFoundError(f"images.npy not found: {images_npy}")

    images = np.load(images_npy, mmap_mode="r")
    if images.ndim != 4:
        raise ValueError(f"images.npy must be rank-4 [N,H,W,C], got shape={images.shape}")

    split_indices = parse_split_indices_file(split_path, dataset_size=len(images))
    if max_images is not None:
        split_indices = split_indices[:max_images]
    if not split_indices:
        raise ValueError("No indices selected from split file.")

    checkpoint_data = torch.load(checkpoint, map_location=device)
    checkpoint_meta = checkpoint_data.get("config_used", {}) if isinstance(checkpoint_data, dict) else {}

    if args.use_pretrained_backbone and args.use_scratch_backbone:
        raise ValueError("Choose only one of --use-pretrained-backbone or --use-scratch-backbone.")
    if args.use_pretrained_backbone:
        use_pretrained_backbone = True
    elif args.use_scratch_backbone:
        use_pretrained_backbone = False
    else:
        use_pretrained_backbone = bool(checkpoint_meta.get("use_pretrained_backbone", False))

    state_dict = checkpoint_data["model_state_dict"] if isinstance(checkpoint_data, dict) and "model_state_dict" in checkpoint_data else checkpoint_data

    if use_pretrained_backbone:
        model = build_model_pretrained(pretrained=True, freeze_backbone=False, apply_sigmoid=True).to(device)
    else:
        model = build_model_scratch(pretrained=False, freeze_backbone=False, apply_sigmoid=True).to(device)
    model.load_state_dict(state_dict)
    model.eval()

    output_root.mkdir(parents=True, exist_ok=True)

    print(f"Checkpoint: {checkpoint}")
    print(f"Images npy: {images_npy}")
    print(f"Split: {split_path}")
    print(f"Device: {device}")
    print(f"Images selected: {len(split_indices)}")
    print(f"Output file: {output_file}")
    print(f"Backbone: {'pretrained' if use_pretrained_backbone else 'scratch'}")

    root = ET.Element("predictions")
    root.set("model", str(checkpoint_meta.get("model_name", "deep_learning_model")))
    root.set("n_test", str(len(split_indices)))

    meta_el = ET.SubElement(root, "metadata")
    ET.SubElement(meta_el, "timestamp").text = datetime.now().isoformat(timespec="seconds")

    hw_el = ET.SubElement(meta_el, "hardware")
    ET.SubElement(hw_el, "platform").text = platform.system()
    ET.SubElement(hw_el, "os_version").text = platform.release()
    ET.SubElement(hw_el, "machine").text = platform.machine()
    ET.SubElement(hw_el, "processor").text = platform.processor() or platform.machine()

    with torch.no_grad():
        for i, dataset_index in enumerate(split_indices, start=1):
            image_arr = np.array(images[dataset_index])
            if image_arr.dtype != np.uint8:
                image_arr = (image_arr * 255.0).clip(0, 255).astype(np.uint8)

            height, width = int(image_arr.shape[0]), int(image_arr.shape[1])
            img_tensor = torch.from_numpy(image_arr).permute(2, 0, 1).float().div(255.0).unsqueeze(0).to(device)

            if img_tensor.shape[-1] != image_size or img_tensor.shape[-2] != image_size:
                img_tensor = torch.nn.functional.interpolate(
                    img_tensor, size=(image_size, image_size), mode="bilinear", align_corners=False
                )

            pred_xywh = model(img_tensor)[0].detach().cpu()
            xmin, ymin, xmax, ymax = to_xyxy_pixels(pred_xywh, width=width, height=height)
            x, y, box_w, box_h = xyxy_to_xywh(xmin, ymin, xmax, ymax)

            img_el = ET.SubElement(root, "image")
            img_el.set("dataset_index", str(int(dataset_index)))
            bbox_el = ET.SubElement(img_el, "predicted_bbox")
            ET.SubElement(bbox_el, "x").text = f"{x:.4f}"
            ET.SubElement(bbox_el, "y").text = f"{y:.4f}"
            ET.SubElement(bbox_el, "width").text = f"{box_w:.4f}"
            ET.SubElement(bbox_el, "height").text = f"{box_h:.4f}"

            if i % 100 == 0 or i == len(split_indices):
                print(f"Processed {i}/{len(split_indices)} images...")

    tree = ET.ElementTree(root)
    ET.indent(tree, space="  ")
    tree.write(str(output_file), encoding="unicode", xml_declaration=True)
    print(f"Saved predictions XML: {output_file}")
    print("Done.")


if __name__ == "__main__":
    main()
