import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import tensorflow as tf


def parse_split_file(split_file: Path) -> list[str]:
    image_ids = []
    with split_file.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            image_ids.append(line.split()[0])
    return image_ids


def load_bbox_xywh(xml_file: Path) -> tuple[float, float, float, float]:
    root = ET.parse(xml_file).getroot()
    width = float(root.findtext("./size/width"))
    height = float(root.findtext("./size/height"))
    xmin = float(root.findtext("./object/bndbox/xmin"))
    ymin = float(root.findtext("./object/bndbox/ymin"))
    xmax = float(root.findtext("./object/bndbox/xmax"))
    ymax = float(root.findtext("./object/bndbox/ymax"))
    x_center = ((xmin + xmax) / 2.0) / width
    y_center = ((ymin + ymax) / 2.0) / height
    box_width = (xmax - xmin) / width
    box_height = (ymax - ymin) / height
    return x_center, y_center, box_width, box_height


def _make_dataset(
    data_dir: Path,
    split_file: Path,
    image_size: int,
    batch_size: int,
    shuffle: bool,
) -> tf.data.Dataset:
    image_ids = parse_split_file(split_file)
    image_paths = [str(data_dir / "images" / f"{image_id}.jpg") for image_id in image_ids]
    targets = np.array(
        [load_bbox_xywh(data_dir / "annotations" / "xmls" / f"{image_id}.xml") for image_id in image_ids],
        dtype=np.float32,
    )

    ds = tf.data.Dataset.from_tensor_slices((image_paths, targets))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(image_paths), reshuffle_each_iteration=True)

    def _load(image_path: tf.Tensor, target: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
        image = tf.io.read_file(image_path)
        image = tf.io.decode_jpeg(image, channels=3)
        image = tf.image.resize(image, [image_size, image_size])
        image = tf.cast(image, tf.float32) / 255.0
        return image, target

    ds = ds.map(_load, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds


def create_datasets(
    data_dir: Path,
    image_size: int,
    batch_size: int,
) -> tuple[tf.data.Dataset, tf.data.Dataset]:
    train_split = data_dir / "annotations" / "trainval.txt"
    val_split = data_dir / "annotations" / "test.txt"
    train_ds = _make_dataset(data_dir, train_split, image_size, batch_size, shuffle=True)
    val_ds = _make_dataset(data_dir, val_split, image_size, batch_size, shuffle=False)
    return train_ds, val_ds
