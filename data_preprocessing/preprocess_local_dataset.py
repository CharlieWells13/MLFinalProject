import argparse
import os
import xml.etree.ElementTree as ET

import numpy as np
from PIL import Image
from tqdm import tqdm


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Preprocess already-downloaded Oxford Pets data into NumPy arrays."
    )
    parser.add_argument(
        "--image-dir",
        default=os.path.join(PROJECT_ROOT, "data", "images"),
        help="Path to image directory containing .jpg files.",
    )
    parser.add_argument(
        "--annotation-dir",
        default=os.path.join(PROJECT_ROOT, "data", "annotations", "xmls"),
        help="Path to annotation directory containing .xml files.",
    )
    parser.add_argument(
        "--output-dir",
        default=os.path.join(PROJECT_ROOT, "preprocessed_data"),
        help="Path where images.npy and bboxes.npy will be saved.",
    )
    parser.add_argument(
        "--target-size",
        type=int,
        default=224,
        help="Square target image size in pixels (default: 224).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    target_size = (args.target_size, args.target_size)

    img_filenames = sorted(f for f in os.listdir(args.image_dir) if f.endswith(".jpg"))
    print(f"Found {len(img_filenames)} images in {args.image_dir}")

    X = []
    Y = []
    skipped = []

    for fname in tqdm(img_filenames):
        annotation_path = os.path.join(
            args.annotation_dir, f"{os.path.splitext(fname)[0]}.xml"
        )
        if not os.path.exists(annotation_path):
            skipped.append(fname)
            continue

        img = Image.open(os.path.join(args.image_dir, fname)).convert("RGB")
        orig_w, orig_h = img.size
        img = img.resize(target_size)

        root = ET.parse(annotation_path).getroot()
        obj = root.find("object")
        bndbox = obj.find("bndbox") if obj is not None else None
        if bndbox is None:
            skipped.append(fname)
            continue

        xmin = int(bndbox.find("xmin").text)
        ymin = int(bndbox.find("ymin").text)
        xmax = int(bndbox.find("xmax").text)
        ymax = int(bndbox.find("ymax").text)

        x_scale = target_size[0] / orig_w
        y_scale = target_size[1] / orig_h
        x = xmin * x_scale
        y = ymin * y_scale
        w = (xmax - xmin) * x_scale
        h = (ymax - ymin) * y_scale

        X.append(np.array(img, dtype=np.float32) / 255.0)
        Y.append([x, y, w, h])

    X = np.array(X, dtype=np.float32)
    Y = np.array(Y, dtype=np.float32)

    os.makedirs(args.output_dir, exist_ok=True)
    np.save(os.path.join(args.output_dir, "images.npy"), X)
    np.save(os.path.join(args.output_dir, "bboxes.npy"), Y)

    print(f"X shape: {X.shape}")
    print(f"Y shape: {Y.shape}")
    print(f"Skipped {len(skipped)} images")
    print(f"Saved images.npy and bboxes.npy to {args.output_dir}")


if __name__ == "__main__":
    main()
