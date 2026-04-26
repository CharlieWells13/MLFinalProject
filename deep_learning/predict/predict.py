import argparse
import sys
import xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torchvision import transforms
import yaml


THIS_DIR = Path(__file__).resolve().parent
DEEP_LEARNING_DIR = THIS_DIR.parent
PROJECT_ROOT = DEEP_LEARNING_DIR.parent
DEFAULT_CONFIG_PATH = THIS_DIR / "config.yaml"
if str(DEEP_LEARNING_DIR) not in sys.path:
    sys.path.append(str(DEEP_LEARNING_DIR))

from models.model_scratch import build_model as build_model_scratch  # noqa: E402
from models.model_pretrained import build_model as build_model_pretrained  # noqa: E402


IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


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


def parse_split_file(split_file: Path) -> list[str]:
    if not split_file.is_file():
        raise FileNotFoundError(f"Split file not found: {split_file}")
    image_ids: list[str] = []
    with split_file.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            image_ids.append(line.split()[0])
    return image_ids


def parse_split_indices_file(split_file: Path, dataset_size: int) -> list[int]:
    if not split_file.is_file():
        raise FileNotFoundError(f"Split file not found: {split_file}")
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


def select_image_path(image_dir: Path, image_id: str) -> Path | None:
    for ext in IMAGE_EXTENSIONS:
        candidate = image_dir / f"{image_id}{ext}"
        if candidate.is_file():
            return candidate
    return None


def collect_images(image_dir: Path, split_file: Path | None, max_images: int | None) -> list[Path]:
    all_paths = sorted(
        p
        for p in image_dir.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )

    if split_file is not None:
        if split_file.suffix.lower() == ".npy":
            split_indices = parse_split_indices_file(split_file, dataset_size=len(all_paths))
            paths = [all_paths[i] for i in split_indices]
            if max_images is not None:
                return paths[:max_images]
            return paths

        image_ids = parse_split_file(split_file)
        paths: list[Path] = []
        missing = 0
        for image_id in image_ids:
            image_path = select_image_path(image_dir, image_id)
            if image_path is None:
                missing += 1
                continue
            paths.append(image_path)
            if max_images is not None and len(paths) >= max_images:
                break
        if missing:
            print(f"Warning: skipped {missing} split entries with missing image files.")
        return paths

    if max_images is not None:
        return all_paths[:max_images]
    return all_paths


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


def build_prediction_xml(
    image_path: Path,
    width: int,
    height: int,
    depth: int,
    label: str,
    bbox_xyxy: tuple[int, int, int, int],
) -> ET.Element:
    xmin, ymin, xmax, ymax = bbox_xyxy

    root = ET.Element("annotation")
    ET.SubElement(root, "folder").text = image_path.parent.name
    ET.SubElement(root, "filename").text = image_path.name

    source = ET.SubElement(root, "source")
    ET.SubElement(source, "database").text = "MODEL_PREDICTION"
    ET.SubElement(source, "annotation").text = "PREDICT_TO_XML"
    ET.SubElement(source, "image").text = "unknown"

    size = ET.SubElement(root, "size")
    ET.SubElement(size, "width").text = str(width)
    ET.SubElement(size, "height").text = str(height)
    ET.SubElement(size, "depth").text = str(depth)

    ET.SubElement(root, "segmented").text = "0"

    obj = ET.SubElement(root, "object")
    ET.SubElement(obj, "name").text = label
    ET.SubElement(obj, "pose").text = "Unspecified"
    ET.SubElement(obj, "truncated").text = "0"
    ET.SubElement(obj, "occluded").text = "0"
    ET.SubElement(obj, "difficult").text = "0"

    bndbox = ET.SubElement(obj, "bndbox")
    ET.SubElement(bndbox, "xmin").text = str(xmin)
    ET.SubElement(bndbox, "ymin").text = str(ymin)
    ET.SubElement(bndbox, "xmax").text = str(xmax)
    ET.SubElement(bndbox, "ymax").text = str(ymax)

    return root


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Load a trained bbox model, run inference on images, and write prediction XML files."
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="",
        help="Checkpoint file or checkpoint run directory.",
    )
    parser.add_argument(
        "--image-dir",
        type=str,
        default=None,
        help="Directory containing input images.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default=None,
        help="Optional split file with image IDs (.txt) or preprocessed indices (.npy).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Root directory where a per-run output folder is created.",
    )
    parser.add_argument("--image-size", type=int, default=None, help="Model input resize (square).")
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=("auto", "cpu", "cuda"),
        help="Inference device.",
    )
    parser.add_argument(
        "--use-pretrained-backbone",
        action="store_true",
        help="Force loading the pretrained-backbone model architecture.",
    )
    parser.add_argument(
        "--use-scratch-backbone",
        action="store_true",
        help="Force loading the scratch-backbone model architecture.",
    )
    parser.add_argument(
        "--label",
        type=str,
        default="object",
        help="Constant placeholder label written into XML (<name>...</name>) for localization-only output.",
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=None,
        help="Optional limit for quick runs (0 means no limit).",
    )
    args = parser.parse_args()

    cfg: dict[str, Any] = load_yaml(DEFAULT_CONFIG_PATH)

    checkpoint_cfg = args.checkpoint or cfg.get("checkpoint")
    if not checkpoint_cfg:
        raise ValueError("checkpoint is required (via --checkpoint or config).")
    checkpoint = resolve_project_path(resolve_checkpoint(checkpoint_cfg), PROJECT_ROOT)

    image_dir = resolve_project_path(args.image_dir or cfg.get("image_dir", "data/images"), PROJECT_ROOT)
    split_value = args.split if args.split is not None else str(cfg.get("split", ""))
    split_file = resolve_project_path(split_value, PROJECT_ROOT) if split_value else None
    output_root = resolve_project_path(
        args.output_dir or cfg.get("output_root") or cfg.get("output_dir", "deep_learning/predict/runs"),
        PROJECT_ROOT,
    )
    run_name_cfg = str(cfg.get("run_name", "")).strip()
    run_name = run_name_cfg if run_name_cfg else datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = output_root / run_name
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

    if not image_dir.is_dir():
        raise FileNotFoundError(f"Image directory not found: {image_dir}")
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint}")

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

    if isinstance(checkpoint_data, dict) and "model_state_dict" in checkpoint_data:
        state_dict = checkpoint_data["model_state_dict"]
    else:
        state_dict = checkpoint_data

    if use_pretrained_backbone:
        model = build_model_pretrained(pretrained=True, freeze_backbone=False, apply_sigmoid=True).to(device)
    else:
        model = build_model_scratch(pretrained=False, freeze_backbone=False, apply_sigmoid=True).to(device)
    model.load_state_dict(state_dict)
    model.eval()

    image_paths = collect_images(image_dir=image_dir, split_file=split_file, max_images=max_images)
    if not image_paths:
        raise ValueError("No input images found.")

    output_dir.mkdir(parents=True, exist_ok=True)
    transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ]
    )

    print(f"Checkpoint: {checkpoint}")
    print(f"Device: {device}")
    print(f"Images: {len(image_paths)}")
    print(f"Output dir: {output_dir}")
    print(f"Backbone: {'pretrained' if use_pretrained_backbone else 'scratch'}")

    with torch.no_grad():
        for i, image_path in enumerate(image_paths, start=1):
            image = Image.open(image_path).convert("RGB")
            width, height = image.size
            depth = len(image.getbands())

            image_tensor = transform(image).unsqueeze(0).to(device)
            pred_xywh = model(image_tensor)[0].detach().cpu()
            bbox_xyxy = to_xyxy_pixels(pred_xywh, width=width, height=height)

            xml_root = build_prediction_xml(
                image_path=image_path,
                width=width,
                height=height,
                depth=depth,
                label=args.label,
                bbox_xyxy=bbox_xyxy,
            )
            ET.indent(xml_root)  # type: ignore[attr-defined]
            out_file = output_dir / f"{image_path.stem}.xml"
            ET.ElementTree(xml_root).write(out_file, encoding="utf-8", xml_declaration=False)

            if i % 100 == 0 or i == len(image_paths):
                print(f"Processed {i}/{len(image_paths)} images...")

    print("Done.")


if __name__ == "__main__":
    main()
