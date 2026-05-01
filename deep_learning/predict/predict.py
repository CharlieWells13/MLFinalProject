import argparse
import platform
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
PREPROCESSED_IMAGES_NPY = PROJECT_ROOT / "preprocessed_data" / "images.npy"
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


def collect_images_with_indices(
    image_dir: Path, split_file: Path | None, max_images: int | None
) -> list[tuple[Path, int]]:
    """Return (image_path, dataset_index) pairs.

    For .npy splits the stored integer values are the dataset indices.
    For .txt splits the image ID is parsed as an integer index.
    Without a split file the position in the sorted directory listing is used.
    """
    all_paths = sorted(
        p
        for p in image_dir.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )

    if split_file is not None:
        if split_file.suffix.lower() == ".npy":
            split_indices = parse_split_indices_file(split_file, dataset_size=len(all_paths))
            pairs = [(all_paths[i], i) for i in split_indices]
            if max_images is not None:
                return pairs[:max_images]
            return pairs

        image_ids = parse_split_file(split_file)
        pairs: list[tuple[Path, int]] = []
        missing = 0
        for image_id in image_ids:
            image_path = select_image_path(image_dir, image_id)
            if image_path is None:
                missing += 1
                continue
            try:
                idx = int(image_id)
            except ValueError:
                raise ValueError(
                    f"Cannot determine dataset_index from image ID '{image_id}'. "
                    "Use a .npy split file or name images by their integer dataset index."
                )
            pairs.append((image_path, idx))
            if max_images is not None and len(pairs) >= max_images:
                break
        if missing:
            print(f"Warning: skipped {missing} split entries with missing image files.")
        return pairs

    pairs = [(p, i) for i, p in enumerate(all_paths)]
    if max_images is not None:
        return pairs[:max_images]
    return pairs


def to_xywh_pixels(xywh_norm: torch.Tensor, width: int, height: int) -> tuple[float, float, float, float]:
    """Convert normalized center-xywh to top-left pixel xywh matching ground-truth format."""
    x_center = float(xywh_norm[0])
    y_center = float(xywh_norm[1])
    box_w = float(xywh_norm[2])
    box_h = float(xywh_norm[3])

    x = (x_center - box_w / 2.0) * width
    y = (y_center - box_h / 2.0) * height
    w = box_w * width
    h = box_h * height

    return x, y, w, h


def build_metadata_element(
    device: torch.device,
    checkpoint: Path,
    image_size: int,
    use_pretrained_backbone: bool,
    checkpoint_meta: dict[str, Any],
) -> ET.Element:
    meta = ET.Element("metadata")
    ET.SubElement(meta, "timestamp").text = datetime.now().isoformat(timespec="seconds")

    hw = ET.SubElement(meta, "hardware")
    ET.SubElement(hw, "device").text = str(device)
    ET.SubElement(hw, "platform").text = platform.system()
    ET.SubElement(hw, "os_version").text = platform.release()
    ET.SubElement(hw, "machine").text = platform.machine()
    ET.SubElement(hw, "processor").text = platform.processor() or platform.machine()

    hp = ET.SubElement(meta, "hyperparams")
    ET.SubElement(hp, "backbone").text = "pretrained" if use_pretrained_backbone else "scratch"
    ET.SubElement(hp, "image_size").text = str(image_size)
    ET.SubElement(hp, "checkpoint").text = str(checkpoint)
    for key in ("epochs", "lr", "weight_decay", "batch_size"):
        val = checkpoint_meta.get(key)
        if val is not None:
            ET.SubElement(hp, key).text = str(val)
    loss_cfg = checkpoint_meta.get("loss", {})
    if isinstance(loss_cfg, dict) and loss_cfg.get("name"):
        ET.SubElement(hp, "loss").text = str(loss_cfg["name"])
    opt_cfg = checkpoint_meta.get("optimizer", {})
    if isinstance(opt_cfg, dict) and opt_cfg.get("name"):
        ET.SubElement(hp, "optimizer").text = str(opt_cfg["name"])

    return meta


def build_predictions_xml(
    model_name: str,
    predictions: list[tuple[int, float, float, float, float]],
    metadata: ET.Element,
) -> ET.Element:
    """Build a single aggregated predictions XML element.

    predictions: list of (dataset_index, x, y, width, height) in pixel coords.
    """
    root = ET.Element("predictions")
    root.set("model", model_name)
    root.set("n_test", str(len(predictions)))
    root.append(metadata)
    for dataset_index, x, y, w, h in predictions:
        img_el = ET.SubElement(root, "image")
        img_el.set("dataset_index", str(dataset_index))
        bbox_el = ET.SubElement(img_el, "predicted_bbox")
        ET.SubElement(bbox_el, "x").text = f"{x:.4f}"
        ET.SubElement(bbox_el, "y").text = f"{y:.4f}"
        ET.SubElement(bbox_el, "width").text = f"{w:.4f}"
        ET.SubElement(bbox_el, "height").text = f"{h:.4f}"
    return root


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Load a trained bbox model, run inference on images, and write a predictions XML file."
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
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")

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

    model_name = "resnet18_pretrained" if use_pretrained_backbone else "resnet18_scratch"

    # When the split is a .npy index file, load images from the preprocessed images.npy
    # so dataset_index values correctly match bboxes.npy (both use os.listdir() ordering).
    # Fallback to loading raw files for .txt splits or when images.npy is absent.
    use_npy = (
        split_file is not None
        and split_file.suffix.lower() == ".npy"
        and PREPROCESSED_IMAGES_NPY.is_file()
    )

    if use_npy:
        _arr = np.load(PREPROCESSED_IMAGES_NPY, mmap_mode="r")
        split_indices = parse_split_indices_file(split_file, dataset_size=len(_arr))
        if max_images is not None:
            split_indices = split_indices[:max_images]
        n_total = len(split_indices)
    else:
        image_pairs = collect_images_with_indices(image_dir=image_dir, split_file=split_file, max_images=max_images)
        if not image_pairs:
            raise ValueError("No input images found.")
        n_total = len(image_pairs)

    output_dir.mkdir(parents=True, exist_ok=True)
    transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ]
    )

    print(f"Checkpoint: {checkpoint}")
    print(f"Device: {device}")
    print(f"Images: {n_total}")
    print(f"Output dir: {output_dir}")
    print(f"Backbone: {'pretrained' if use_pretrained_backbone else 'scratch'}")
    if use_npy:
        print(f"Image source: {PREPROCESSED_IMAGES_NPY} (preprocessed, correct index ordering)")

    predictions: list[tuple[int, float, float, float, float]] = []
    with torch.no_grad():
        if use_npy:
            all_imgs = np.load(PREPROCESSED_IMAGES_NPY, mmap_mode="r")
            for i, idx in enumerate(split_indices, start=1):
                # images.npy stores float32 [0,1] in (H,W,C) — convert to (C,H,W) tensor
                img_arr = np.array(all_imgs[idx])
                img_tensor = torch.from_numpy(img_arr).permute(2, 0, 1).unsqueeze(0).to(device)
                pred_xywh_norm = model(img_tensor)[0].detach().cpu()
                x, y, w, h = to_xywh_pixels(pred_xywh_norm, width=image_size, height=image_size)
                predictions.append((idx, x, y, w, h))
                if i % 100 == 0 or i == n_total:
                    print(f"Processed {i}/{n_total} images...")
        else:
            for i, (image_path, dataset_index) in enumerate(image_pairs, start=1):
                image = Image.open(image_path).convert("RGB")
                image_tensor = transform(image).unsqueeze(0).to(device)
                pred_xywh_norm = model(image_tensor)[0].detach().cpu()
                x, y, w, h = to_xywh_pixels(pred_xywh_norm, width=image_size, height=image_size)
                predictions.append((dataset_index, x, y, w, h))
                if i % 100 == 0 or i == n_total:
                    print(f"Processed {i}/{n_total} images...")

    metadata = build_metadata_element(
        device=device,
        checkpoint=checkpoint,
        image_size=image_size,
        use_pretrained_backbone=use_pretrained_backbone,
        checkpoint_meta=checkpoint_meta,
    )
    xml_root = build_predictions_xml(model_name, predictions, metadata)
    ET.indent(xml_root)  # type: ignore[attr-defined]
    out_file = output_dir / "predictions.xml"
    ET.ElementTree(xml_root).write(out_file, encoding="utf-8", xml_declaration=True)
    print(f"Saved predictions for {len(predictions)} images to {out_file}")


if __name__ == "__main__":
    main()
