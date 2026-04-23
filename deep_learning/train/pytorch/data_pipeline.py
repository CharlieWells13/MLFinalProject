import xml.etree.ElementTree as ET
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


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


class OxfordPetBBoxDataset(Dataset):
    def __init__(self, data_dir: Path, split_file: Path, image_size: int = 224):
        self.data_dir = data_dir
        self.image_ids = parse_split_file(split_file)
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
        image_path = self.data_dir / "images" / f"{image_id}.jpg"
        xml_path = self.data_dir / "annotations" / "xmls" / f"{image_id}.xml"
        image = Image.open(image_path).convert("RGB")
        image = self.transform(image)
        target = torch.tensor(load_bbox_xywh(xml_path), dtype=torch.float32)
        return image, target


def create_dataloaders(
    data_dir: Path,
    image_size: int,
    batch_size: int,
    num_workers: int,
) -> tuple[DataLoader, DataLoader]:
    train_split = data_dir / "annotations" / "trainval.txt"
    val_split = data_dir / "annotations" / "test.txt"
    train_ds = OxfordPetBBoxDataset(data_dir, train_split, image_size=image_size)
    val_ds = OxfordPetBBoxDataset(data_dir, val_split, image_size=image_size)

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
