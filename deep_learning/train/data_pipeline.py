import xml.etree.ElementTree as ET
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms


def parse_split_file(split_file: Path) -> list[str]:
    if not split_file.is_file():
        raise FileNotFoundError(f"Split file not found: {split_file.resolve()}")
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


class OxfordPetBBoxDataset(Dataset):
    def __init__(self, data_dir: Path, split_file: Path, image_size: int = 224):
        self.data_dir = data_dir
        self.image_dir = self.data_dir / "images"
        self.xml_dir = self.data_dir / "annotations" / "xmls"
        parsed_ids = parse_split_file(split_file)
        self.image_ids = [
            image_id
            for image_id in parsed_ids
            if (self.image_dir / f"{image_id}.jpg").is_file() and (self.xml_dir / f"{image_id}.xml").is_file()
        ]
        missing_count = len(parsed_ids) - len(self.image_ids)
        if missing_count:
            print(f"Warning: skipped {missing_count} samples in {split_file.name} with missing image/xml files.")
        if not self.image_ids:
            raise ValueError(f"No valid samples found for split: {split_file.resolve()}")
        self.transform = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
            ]
        )

    def __len__(self) -> int:
        return len(self.image_ids)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        image_id = self.image_ids[idx]
        image_path = self.image_dir / f"{image_id}.jpg"
        xml_path = self.xml_dir / f"{image_id}.xml"
        image = Image.open(image_path).convert("RGB")
        image = self.transform(image)
        target = torch.tensor(load_bbox_xywh(xml_path), dtype=torch.float32)
        return image, target


def create_dataloaders(
    data_dir: Path,
    image_size: int,
    batch_size: int,
    num_workers: int,
    train_split_file: Path | None = None,
    val_split_file: Path | None = None,
) -> tuple[DataLoader, DataLoader]:
    train_split = train_split_file or (data_dir / "annotations" / "custom_split" / "train.txt")
    val_split = val_split_file or (data_dir / "annotations" / "custom_split" / "val.txt")
    if not data_dir.is_dir():
        raise FileNotFoundError(f"Data directory not found: {data_dir.resolve()}")
    train_ds = OxfordPetBBoxDataset(data_dir, train_split, image_size=image_size)
    try:
        val_ds = OxfordPetBBoxDataset(data_dir, val_split, image_size=image_size)
    except ValueError:
        if len(train_ds) < 2:
            raise ValueError("Need at least 2 valid training samples to create a validation split.")
        val_size = max(1, int(0.1 * len(train_ds)))
        train_size = len(train_ds) - val_size
        train_ds, val_ds = random_split(
            train_ds,
            [train_size, val_size],
            generator=torch.Generator().manual_seed(42),
        )
        print("Warning: test.txt has no valid bbox labels. Using a deterministic 90/10 split from trainval.txt.")

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    return train_loader, val_loader
