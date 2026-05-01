import numpy as np
import torch
from pathlib import Path
from torch.utils.data import DataLoader, Dataset, Subset


ArrayLikeOrPath = np.ndarray | str


def _load_npy(array_or_path: ArrayLikeOrPath) -> np.ndarray:
    if isinstance(array_or_path, str):
        return np.load(array_or_path)
    return array_or_path


def _load_index_array(
    indices_or_path: np.ndarray | str,
    dataset_size: int,
    split_name: str,
) -> np.ndarray:
    if isinstance(indices_or_path, str):
        indices = np.load(indices_or_path)
    else:
        indices = indices_or_path

    if indices.ndim != 1:
        raise ValueError(f"{split_name} indices must be a 1D array, got shape={indices.shape}")
    if len(indices) == 0:
        raise ValueError(f"{split_name} indices are empty.")
    if not np.issubdtype(indices.dtype, np.integer):
        raise ValueError(f"{split_name} indices must contain integers, got dtype={indices.dtype}")

    if int(indices.min()) < 0 or int(indices.max()) >= dataset_size:
        raise ValueError(
            f"{split_name} indices out of range for dataset size {dataset_size}: "
            f"min={int(indices.min())}, max={int(indices.max())}"
        )

    unique_count = len(np.unique(indices))
    if unique_count != len(indices):
        raise ValueError(
            f"{split_name} indices contain duplicates: {len(indices) - unique_count} repeated entries."
        )
    return indices.astype(np.int64, copy=False)


def _default_split_path(filename: str) -> str:
    project_root = Path(__file__).resolve().parent.parent.parent
    return str(project_root / "preprocessed_data" / filename)


class OxfordPetBBoxDataset(Dataset):
    """
    Dataset backed by preprocessed NumPy arrays from preprocess_dataset.ipynb.

    Expected inputs:
    - images: shape (N, H, W, 3), float in [0, 1] or uint8 in [0, 255]
    - bboxes: shape (N, 4), format (x, y, width, height) in pixel space
    """

    def __init__(
        self,
        images: ArrayLikeOrPath,
        bboxes: ArrayLikeOrPath,
    ):
        images_np = _load_npy(images)
        bboxes_np = _load_npy(bboxes)

        if images_np.ndim != 4 or images_np.shape[-1] != 3:
            raise ValueError(f"Expected images with shape (N, H, W, 3), got {images_np.shape}")
        if bboxes_np.ndim != 2 or bboxes_np.shape[1] != 4:
            raise ValueError(f"Expected bboxes with shape (N, 4), got {bboxes_np.shape}")
        if len(images_np) != len(bboxes_np):
            raise ValueError(
                "Image/box count mismatch: "
                f"images={len(images_np)} bboxes={len(bboxes_np)}"
            )

        self.images = images_np
        self.bboxes = bboxes_np.astype(np.float32)

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        image_np = self.images[idx]
        if image_np.dtype != np.float32:
            image_np = image_np.astype(np.float32)

        image = torch.from_numpy(image_np).permute(2, 0, 1).contiguous()
        if image.max() > 1.0:
            image = image / 255.0

        x, y, w, h = self.bboxes[idx]
        img_h, img_w = image.shape[1], image.shape[2]

        # Convert (x, y, w, h) pixel-space top-left format
        # to normalized (x_center, y_center, width, height).
        x_center = (x + (w / 2.0)) / float(img_w)
        y_center = (y + (h / 2.0)) / float(img_h)
        box_width = w / float(img_w)
        box_height = h / float(img_h)

        target = torch.tensor([x_center, y_center, box_width, box_height], dtype=torch.float32)
        return image, target


def create_dataloaders(
    images: ArrayLikeOrPath,
    bboxes: ArrayLikeOrPath,
    batch_size: int,
    num_workers: int,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
    train_indices: np.ndarray | str | None = None,
    val_indices: np.ndarray | str | None = None,
) -> tuple[DataLoader, DataLoader]:
    dataset = OxfordPetBBoxDataset(images, bboxes)
    dataset_size = len(dataset)

    # Backward-compatible args retained in signature for callers, but split ratios
    # are intentionally ignored in favor of exact precomputed index files.
    _ = (train_ratio, val_ratio, test_ratio)

    train_indices = train_indices or _default_split_path("train_indices.npy")
    val_indices = val_indices or _default_split_path("val_indices.npy")

    train_idx = _load_index_array(train_indices, dataset_size=dataset_size, split_name="train")
    val_idx = _load_index_array(val_indices, dataset_size=dataset_size, split_name="val")

    overlap = np.intersect1d(train_idx, val_idx)
    if len(overlap) > 0:
        raise ValueError(f"Train/val splits overlap on {len(overlap)} sample(s).")

    train_ds = Subset(dataset, train_idx.tolist())
    val_ds = Subset(dataset, val_idx.tolist())

    loader_generator = torch.Generator().manual_seed(seed)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        generator=loader_generator,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    return train_loader, val_loader
