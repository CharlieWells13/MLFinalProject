from __future__ import annotations

import argparse
import platform
import xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np


ROOT = Path(__file__).resolve().parent.parent
ML_DIR = ROOT / "machine_learning"
PREPROCESSED_DIR = ROOT / "preprocessed_data"
PREDICTIONS_DIR = ML_DIR / "predictions"
DEFAULT_MODEL_PATH = ML_DIR / "models" / "random_forest_pipeline.joblib"


def export_predictions_xml(
    y_pred: np.ndarray,
    indices: np.ndarray,
    split_name: str,
    model_name: str,
    best_params: dict[str, object],
    out_path: Path,
) -> None:
    root = ET.Element("predictions")
    root.set("model", model_name)
    root.set("n_test", str(len(indices)))

    meta_el = ET.SubElement(root, "metadata")
    ET.SubElement(meta_el, "timestamp").text = datetime.now().isoformat(timespec="seconds")

    hw_el = ET.SubElement(meta_el, "hardware")
    ET.SubElement(hw_el, "platform").text = platform.system()
    ET.SubElement(hw_el, "os_version").text = platform.release()
    ET.SubElement(hw_el, "machine").text = platform.machine()
    ET.SubElement(hw_el, "processor").text = platform.processor() or platform.machine()

    hp_el = ET.SubElement(meta_el, "hyperparams")
    ET.SubElement(hp_el, "n_estimators").text = str(best_params["rf__n_estimators"])
    ET.SubElement(hp_el, "max_depth").text = str(best_params["rf__max_depth"])
    ET.SubElement(hp_el, "min_samples_leaf").text = str(best_params["rf__min_samples_leaf"])
    ET.SubElement(hp_el, "pca_n_components").text = str(best_params["pca__n_components"])

    for i, idx in enumerate(indices):
        img_el = ET.SubElement(root, "image")
        img_el.set("dataset_index", str(int(idx)))
        px, py, pw, ph = y_pred[i]
        bbox_el = ET.SubElement(img_el, "predicted_bbox")
        ET.SubElement(bbox_el, "x").text = f"{px:.4f}"
        ET.SubElement(bbox_el, "y").text = f"{py:.4f}"
        ET.SubElement(bbox_el, "width").text = f"{pw:.4f}"
        ET.SubElement(bbox_el, "height").text = f"{ph:.4f}"

    tree = ET.ElementTree(root)
    ET.indent(tree, space="  ")
    tree.write(str(out_path), encoding="unicode", xml_declaration=True)
    print(f"Saved {split_name} predictions ({len(indices)} images) -> {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Load RF model and export train/test XML predictions.")
    parser.add_argument(
        "--model-path",
        default=str(DEFAULT_MODEL_PATH),
        help="Path to saved random_forest_pipeline.joblib",
    )
    parser.add_argument(
        "--images-path",
        default=str(PREPROCESSED_DIR / "images.npy"),
        help="Path to preprocessed images .npy",
    )
    parser.add_argument(
        "--train-indices",
        default=str(PREPROCESSED_DIR / "train_indices.npy"),
        help="Path to train indices .npy",
    )
    parser.add_argument(
        "--test-indices",
        default=str(PREPROCESSED_DIR / "test_indices.npy"),
        help="Path to test indices .npy",
    )
    parser.add_argument(
        "--output-dir",
        default=str(PREDICTIONS_DIR),
        help="Directory for XML predictions",
    )
    args = parser.parse_args()

    model_path = Path(args.model_path)
    if not model_path.is_file():
        raise FileNotFoundError(f"Model not found: {model_path}")

    images = np.load(Path(args.images_path))
    train_idx = np.load(Path(args.train_indices))
    test_idx = np.load(Path(args.test_indices))

    x_flat = images.reshape(len(images), -1)
    x_train = x_flat[train_idx]
    x_test = x_flat[test_idx]

    print(f"Images: {images.shape}")
    print(f"Train: {x_train.shape[0]} samples | Test: {x_test.shape[0]} samples")

    model = joblib.load(model_path)
    print(f"Loaded model -> {model_path}")

    y_pred_train = model.predict(x_train)
    y_pred_test = model.predict(x_test)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Best params are included for metadata compatibility with evaluation script.
    best_params = {
        "pca__n_components": 100,
        "rf__n_estimators": 200,
        "rf__max_depth": None,
        "rf__min_samples_leaf": 2,
    }

    export_predictions_xml(
        y_pred_test,
        test_idx,
        "test",
        "random_forest",
        best_params,
        output_dir / "rf_predictions_test.xml",
    )
    export_predictions_xml(
        y_pred_train,
        train_idx,
        "train",
        "random_forest",
        best_params,
        output_dir / "rf_predictions_train.xml",
    )


if __name__ == "__main__":
    main()
