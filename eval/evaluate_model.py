import argparse
import json
import xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parent
DEFAULT_GT_PATH = PROJECT_ROOT / "preprocessed_data" / "bboxes.npy"
DEFAULT_IMAGES_NPY = PROJECT_ROOT / "preprocessed_data" / "images.npy"
DEFAULT_OUTPUT_DIR = THIS_DIR / "results"

IOU_THRESHOLDS = [0.25, 0.50, 0.75, 0.90]


# ---------------------------------------------------------------------------
# Parsing
# example: python eval\evaluate_model.py --test-predictions eval\predictions\resnet_18\file.xml
# ---------------------------------------------------------------------------

def parse_predictions_xml(path: Path) -> tuple[dict[int, tuple[float, float, float, float]], dict]:
    root = ET.parse(path).getroot()

    metadata: dict = {"model": root.get("model", "unknown"), "source_file": str(path)}

    meta_el = root.find("metadata")
    if meta_el is not None:
        ts = meta_el.findtext("timestamp")
        if ts:
            metadata["prediction_timestamp"] = ts
        hw = meta_el.find("hardware")
        if hw is not None:
            metadata["hardware"] = {el.tag: el.text for el in hw}
        hp = meta_el.find("hyperparams")
        if hp is not None:
            metadata["hyperparams"] = {el.tag: el.text for el in hp}

    predictions: dict[int, tuple[float, float, float, float]] = {}
    for img_el in root.findall("image"):
        idx = int(img_el.get("dataset_index"))
        bbox = img_el.find("predicted_bbox")
        predictions[idx] = (
            float(bbox.findtext("x")),
            float(bbox.findtext("y")),
            float(bbox.findtext("width")),
            float(bbox.findtext("height")),
        )

    return predictions, metadata


def load_ground_truth(path: Path) -> dict[int, tuple[float, float, float, float]]:
    bboxes = np.load(path)
    return {i: tuple(float(v) for v in bboxes[i]) for i in range(len(bboxes))}


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def iou_xywh(pred: tuple, gt: tuple) -> float:
    px, py, pw, ph = pred
    gx, gy, gw, gh = gt

    ix1 = max(px, gx)
    iy1 = max(py, gy)
    ix2 = min(px + pw, gx + gw)
    iy2 = min(py + ph, gy + gh)

    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    intersection = iw * ih

    if intersection == 0.0:
        return 0.0
    union = pw * ph + gw * gh - intersection
    return intersection / union if union > 0.0 else 0.0


def compute_metrics(
    predictions: dict[int, tuple],
    ground_truth: dict[int, tuple],
) -> dict:
    ious = []
    per_image = []

    for idx, pred in predictions.items():
        gt = ground_truth.get(idx)
        if gt is None:
            continue
        score = iou_xywh(pred, gt)
        ious.append(score)
        per_image.append({"dataset_index": idx, "iou": round(score, 6)})

    arr = np.array(ious)
    summary = {
        "n_evaluated": len(ious),
        "mean_iou": float(np.mean(arr)),
        "median_iou": float(np.median(arr)),
        "std_iou": float(np.std(arr)),
        "min_iou": float(np.min(arr)),
        "max_iou": float(np.max(arr)),
    }
    for t in IOU_THRESHOLDS:
        summary[f"iou_at_{int(t * 100)}"] = float(np.mean(arr >= t))

    return {**summary, "per_image": per_image}


# ---------------------------------------------------------------------------
# Visualizations
# ---------------------------------------------------------------------------

def _load_image_from_npy(images_npy: Path, dataset_index: int) -> "Image.Image | None":
    if not images_npy.is_file():
        return None
    arr = np.load(images_npy, mmap_mode="r")
    img_arr = np.array(arr[dataset_index])
    if img_arr.dtype != np.uint8:
        img_arr = (img_arr * 255).clip(0, 255).astype(np.uint8)
    return Image.fromarray(img_arr)


def _select_median_sample(per_image: list[dict]) -> dict:
    """Pick the entry whose IoU is closest to the median — a representative example."""
    median = float(np.median([e["iou"] for e in per_image]))
    return min(per_image, key=lambda e: abs(e["iou"] - median))


def generate_sample_bbox_plot(
    sample: dict,
    predictions: dict[int, tuple],
    ground_truth: dict[int, tuple],
    images_npy: Path,
    output_path: Path,
    split_label: str = "Test",
) -> bool:
    idx = sample["dataset_index"]
    img = _load_image_from_npy(images_npy, idx)
    if img is None:
        return False

    pred = predictions[idx]
    gt = ground_truth[idx]

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(img)

    gx, gy, gw, gh = gt
    ax.add_patch(patches.Rectangle(
        (gx, gy), gw, gh, linewidth=2, edgecolor="red", facecolor="none", label="Ground Truth"
    ))

    px, py, pw, ph = pred
    ax.add_patch(patches.Rectangle(
        (px, py), pw, ph, linewidth=2, edgecolor="green", facecolor="none", label="Predicted"
    ))

    ax.set_title(f"{split_label} sample — index {idx}\nIoU: {sample['iou']:.3f}", fontsize=10)
    ax.legend(loc="upper right", fontsize=8)
    ax.axis("off")

    fig.tight_layout()
    fig.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return True


def generate_random_bbox_grid_plot(
    per_image: list[dict],
    predictions: dict[int, tuple],
    ground_truth: dict[int, tuple],
    images_npy: Path,
    output_path: Path,
    n_samples: int = 10,
    seed: int = 42,
) -> bool:
    if not per_image:
        return False

    rng = np.random.default_rng(seed)
    sample_count = min(n_samples, len(per_image))
    selected = rng.choice(per_image, size=sample_count, replace=False).tolist()

    ncols = 5
    nrows = int(np.ceil(sample_count / ncols))
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(4 * ncols, 4 * nrows))
    axes_arr = np.array(axes, dtype=object).reshape(-1)

    rendered = 0
    for i, sample in enumerate(selected):
        ax = axes_arr[i]
        idx = sample["dataset_index"]
        img = _load_image_from_npy(images_npy, idx)
        if img is None:
            ax.axis("off")
            continue

        pred = predictions[idx]
        gt = ground_truth[idx]

        ax.imshow(img)
        gx, gy, gw, gh = gt
        ax.add_patch(patches.Rectangle((gx, gy), gw, gh, linewidth=2, edgecolor="red", facecolor="none"))
        px, py, pw, ph = pred
        ax.add_patch(patches.Rectangle((px, py), pw, ph, linewidth=2, edgecolor="green", facecolor="none"))
        ax.set_title(f"idx {idx} | IoU {sample['iou']:.3f}", fontsize=9)
        ax.axis("off")
        rendered += 1

    for j in range(sample_count, len(axes_arr)):
        axes_arr[j].axis("off")

    if rendered == 0:
        plt.close(fig)
        return False

    fig.suptitle(f"Random prediction samples (n={rendered})", fontsize=12)
    fig.tight_layout()
    fig.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return True


def generate_iou_distribution_plot(
    test_per_image: list[dict],
    output_path: Path,
    train_per_image: list[dict] | None = None,
) -> None:
    bins = np.linspace(0, 1, 26)
    test_ious = [e["iou"] for e in test_per_image]

    fig, ax = plt.subplots(figsize=(7, 4))

    if train_per_image:
        train_ious = [e["iou"] for e in train_per_image]
        ax.hist(train_ious, bins=bins, alpha=0.6, color="steelblue", label=f"Train (n={len(train_ious)})")
        ax.hist(test_ious, bins=bins, alpha=0.6, color="coral", label=f"Test (n={len(test_ious)})")
    else:
        ax.hist(test_ious, bins=bins, alpha=0.8, color="steelblue", label=f"Test (n={len(test_ious)})")

    ax.set_xlabel("IoU Score")
    ax.set_ylabel("Count")
    ax.set_title("IoU Distribution")
    ax.set_xlim(0, 1)
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def generate_visuals(
    test_metrics: dict,
    test_predictions: dict,
    ground_truth: dict,
    images_npy: Path,
    visuals_dir: Path,
    train_metrics: dict | None = None,
    train_predictions: dict | None = None,
) -> dict:
    """Generate all plots, return dict of relative paths (relative to visuals_dir)."""
    visuals_dir.mkdir(parents=True, exist_ok=True)
    saved = {}

    # IoU distribution
    dist_path = visuals_dir / "iou_distribution.png"
    generate_iou_distribution_plot(
        test_metrics["per_image"],
        dist_path,
        train_per_image=train_metrics["per_image"] if train_metrics else None,
    )
    saved["iou_distribution"] = str(dist_path.relative_to(visuals_dir))

    # Random bbox grid — test
    random_grid_path = visuals_dir / "prediction_visualization.png"
    ok = generate_random_bbox_grid_plot(
        per_image=test_metrics["per_image"],
        predictions=test_predictions,
        ground_truth=ground_truth,
        images_npy=images_npy,
        output_path=random_grid_path,
        n_samples=10,
        seed=42,
    )
    if ok:
        saved["prediction_visualization"] = str(random_grid_path.relative_to(visuals_dir))

    # Sample bbox — train
    if train_metrics and train_predictions:
        train_sample = _select_median_sample(train_metrics["per_image"])
        train_bbox_path = visuals_dir / "sample_bbox_train.png"
        ok = generate_sample_bbox_plot(train_sample, train_predictions, ground_truth, images_npy, train_bbox_path, "Train")
        if ok:
            saved["sample_bbox_train"] = str(train_bbox_path.relative_to(visuals_dir))

    return saved


def build_report_dir(output_dir: Path, test_predictions_path: Path) -> Path:
    """Create a deterministic output folder: <prediction_parent>_<prediction_stem>."""
    folder_name = f"{test_predictions_path.parent.name}_{test_predictions_path.stem}"
    run_dir = output_dir / folder_name
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def resolve_output_dir(user_output_dir: str | None) -> Path:
    """
    Always save reports under eval/results.
    If a legacy model_eval_database path is passed, map it to eval/results.
    """
    if not user_output_dir:
        return DEFAULT_OUTPUT_DIR

    requested = Path(user_output_dir)
    if requested.name == "model_eval_database":
        return THIS_DIR / "results"
    if "model_eval_database" in requested.parts:
        parts = list(requested.parts)
        idx = parts.index("model_eval_database")
        remapped = Path(*parts[:idx], "results", *parts[idx + 1 :])
        return remapped
    return requested


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def _fmt_ts(raw: str) -> str:
    try:
        return datetime.strptime(raw, "%Y%m%d_%H%M%S").strftime("%Y-%m-%d %H:%M:%S")
    except ValueError:
        return raw


def _fmt_hyperparams(hp: dict) -> str:
    return "  ·  ".join(f"{k}={v}" for k, v in hp.items())


def _metrics_row(label: str, m: dict | None) -> str:
    if m is None:
        return f"| **{label}** | — | — | — | — | — |\n"
    return (
        f"| **{label}** "
        f"| {m.get('mean_iou', 0):.3f} "
        f"| {m.get('median_iou', 0):.3f} "
        f"| {m.get('iou_at_50', 0) * 100:.1f}% "
        f"| {m.get('iou_at_75', 0) * 100:.1f}% "
        f"| {m.get('n_evaluated', '?')} |\n"
    )


def save_report(
    test_metrics: dict,
    metadata: dict,
    report_dir: Path,
    train_metrics: dict | None = None,
    visuals: dict | None = None,
) -> Path:
    report_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = report_dir / "report.json"

    report = {
        "report_timestamp": timestamp,
        "report_timestamp_human": _fmt_ts(timestamp),
        "metadata": metadata,
        "test_metrics": {k: v for k, v in test_metrics.items() if k != "per_image"},
        "test_per_image": test_metrics["per_image"],
    }
    if train_metrics is not None:
        report["train_metrics"] = {k: v for k, v in train_metrics.items() if k != "per_image"}
        report["train_per_image"] = train_metrics["per_image"]
    if visuals:
        report["visuals"] = visuals

    with report_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    return report_path


def update_reports_index(output_dir: Path) -> None:
    reports = sorted(output_dir.glob("*.json"), key=lambda p: p.stem, reverse=True)

    lines = [
        "# Model Evaluation Reports\n",
        f"_Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}_\n",
    ]

    for path in reports:
        try:
            report = json.loads(path.read_text(encoding="utf-8"))
            meta = report.get("metadata", {})
            model = meta.get("model", "unknown")
            ts = _fmt_ts(report.get("report_timestamp", path.stem[:15]))
            test_m = report.get("test_metrics")
            train_m = report.get("train_metrics")
            hp = meta.get("hyperparams", {})
            vis = report.get("visuals", {})

            lines.append(f"\n---\n\n## {ts} — {model}\n\n")
            lines.append("| Split | Mean IoU | Median IoU | % preds ≥ IoU 0.50 | % preds ≥ IoU 0.75 | n |\n")
            lines.append("|-------|----------|------------|--------------------|--------------------|-|\n")
            if train_m is not None:
                lines.append(_metrics_row("Train", train_m))
            lines.append(_metrics_row("Test", test_m))
            if hp:
                lines.append(f"\n**Hyperparams:** {_fmt_hyperparams(hp)}\n")

            if vis.get("iou_distribution"):
                lines.append(f"\n![IoU Distribution]({vis['iou_distribution']})\n")
            if vis.get("sample_bbox_train") and vis.get("prediction_visualization"):
                lines.append(
                    f"\n| Train sample | Test sample |\n"
                    f"|---|---|\n"
                    f"| ![Train]({vis['sample_bbox_train']}) | ![Test]({vis['prediction_visualization']}) |\n"
                )
            elif vis.get("prediction_visualization"):
                lines.append(f"\n![Prediction Visualization]({vis['prediction_visualization']})\n")

        except Exception:
            continue

    lines.append("\n---\n")
    (output_dir / "reports_index.md").write_text("".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------------
# Console output
# ---------------------------------------------------------------------------

def print_split_summary(label: str, metrics: dict) -> None:
    print(f"  [{label}]")
    print(f"    n evaluated : {metrics['n_evaluated']}")
    print(f"    mean IoU    : {metrics['mean_iou']:.4f}")
    print(f"    median IoU  : {metrics['median_iou']:.4f}")
    print(f"    std IoU     : {metrics['std_iou']:.4f}")
    for t in IOU_THRESHOLDS:
        print(f"    % preds >= {t:.2f} : {metrics[f'iou_at_{int(t * 100)}'] * 100:.1f}%")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate bounding box predictions against ground truth using IoU.")
    parser.add_argument("--test-predictions", required=True, help="Path to test-set predictions XML.")
    parser.add_argument("--train-predictions", default=None, help="Path to train-set predictions XML (optional).")
    parser.add_argument("--ground-truth", default=str(DEFAULT_GT_PATH), help="Path to ground truth bboxes .npy file.")
    parser.add_argument("--images-npy", default=str(DEFAULT_IMAGES_NPY), help="Path to preprocessed images.npy for bbox visualization.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="Directory to save report files.")
    args = parser.parse_args()

    output_dir = resolve_output_dir(args.output_dir)
    images_npy = Path(args.images_npy)
    test_predictions_path = Path(args.test_predictions)
    report_dir = build_report_dir(output_dir, test_predictions_path)

    print(f"Ground truth: {args.ground_truth}")
    ground_truth = load_ground_truth(Path(args.ground_truth))

    print(f"Test predictions: {args.test_predictions}")
    print(f"Report directory: {report_dir}")
    test_preds, metadata = parse_predictions_xml(test_predictions_path)
    test_metrics = compute_metrics(test_preds, ground_truth)

    train_preds, train_metrics = None, None
    if args.train_predictions:
        print(f"Train predictions: {args.train_predictions}")
        train_preds, _ = parse_predictions_xml(Path(args.train_predictions))
        train_metrics = compute_metrics(train_preds, ground_truth)

    # Visuals — save directly in the per-prediction report directory.
    model = metadata.get("model", "unknown")
    visuals_dir = report_dir

    print("Generating visualizations...")
    visuals = generate_visuals(
        test_metrics=test_metrics,
        test_predictions=test_preds,
        ground_truth=ground_truth,
        images_npy=images_npy,
        visuals_dir=visuals_dir,
        train_metrics=train_metrics,
        train_predictions=train_preds,
    )
    if not visuals.get("prediction_visualization"):
        print(f"  Note: images.npy not found ({images_npy}), skipping bbox visualization.")

    report_path = save_report(test_metrics, metadata, report_dir, train_metrics, visuals)
    print(f"Report saved: {report_path}")

    update_reports_index(report_dir)
    print(f"Index updated: {report_dir / 'reports_index.md'}")

    print(f"\n=== {model} ===")
    if train_metrics:
        print_split_summary("Train", train_metrics)
    print_split_summary("Test", test_metrics)


if __name__ == "__main__":
    main()
