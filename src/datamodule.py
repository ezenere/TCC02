"""Dataset and DataLoader built from manifest.csv.

Every experiment reads the manifest and nothing else. The processed image
files are never enumerated from the filesystem, so runs stay reproducible and
comparable across axes. The class list is derived from the manifest itself,
which keeps a single source of truth for the label ordering.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

# ImageNet statistics: the backbones are initialised with ImageNet weights.
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def manifest_sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_manifest(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    expected = {"image_path", "label", "user_id", "split"}
    missing = expected - set(df.columns)
    if missing:
        raise ValueError(f"manifest is missing columns: {sorted(missing)}")
    return df


def class_names(df: pd.DataFrame) -> list[str]:
    """Label ordering: sorted, so the class index is stable across runs."""
    return sorted(df["label"].unique())


class ManifestDataset(Dataset):
    """Images of one split, addressed by row order in the manifest."""

    def __init__(self, df: pd.DataFrame, classes: list[str], transform,
                 root: Path | None = None):
        self.paths = df["image_path"].tolist()
        index = {c: i for i, c in enumerate(classes)}
        self.targets = [index[label] for label in df["label"]]
        self.transform = transform
        self.root = Path(root) if root else None

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, i: int):
        path = self.paths[i]
        if self.root is not None:
            path = self.root / path
        with Image.open(path) as im:
            img = im.convert("RGB")
        return self.transform(img), self.targets[i]


def build_transforms(cfg: dict) -> tuple:
    """Train: random crop from the stored 256px image + optional hflip.

    Test: deterministic centre crop, so the metric never moves between runs.
    """
    size = int(cfg["data"]["image_size"])
    normalize = transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)

    train_ops = [transforms.RandomCrop(size)]
    if cfg["data"].get("hflip", True):
        train_ops.append(transforms.RandomHorizontalFlip(p=0.5))
    train_ops += [transforms.ToTensor(), normalize]

    test_ops = [transforms.CenterCrop(size), transforms.ToTensor(), normalize]
    return transforms.Compose(train_ops), transforms.Compose(test_ops)


def build_dataloaders(cfg: dict, mask_column: str | None = None):
    """Train/test loaders plus metadata that must be logged with the run.

    `mask_column` selects a boolean column of the manifest (axis 4 fractions);
    it only ever restricts the training split, the test split is untouchable.
    """
    manifest = cfg["data"]["manifest"]
    df = read_manifest(manifest)
    classes = class_names(df)

    train_df = df[df["split"] == "train"]
    test_df = df[df["split"] == "test"]

    if mask_column:
        if mask_column not in df.columns:
            raise ValueError(f"manifest has no column '{mask_column}'")
        train_df = train_df[train_df[mask_column].astype(bool)]

    # Subject disjointness is a precondition of every axis; re-check it here so
    # a hand-edited manifest can never silently leak into a run.
    overlap = set(train_df["user_id"]) & set(test_df["user_id"])
    if overlap:
        raise AssertionError(
            f"subject leakage: {len(overlap)} user_ids in both splits")

    train_tf, test_tf = build_transforms(cfg)
    root = cfg["data"].get("root")

    train_ds = ManifestDataset(train_df, classes, train_tf, root)
    test_ds = ManifestDataset(test_df, classes, test_tf, root)

    workers = int(cfg["data"]["workers"])
    batch = int(cfg["data"]["batch_size"])
    common = dict(
        num_workers=workers,
        pin_memory=True,
        persistent_workers=workers > 0,
        prefetch_factor=int(cfg["data"].get("prefetch_factor", 4)) if workers else None,
    )

    train_loader = DataLoader(
        train_ds, batch_size=batch, shuffle=True, drop_last=True,
        generator=torch.Generator().manual_seed(int(cfg["seed"])), **common)
    test_loader = DataLoader(
        test_ds, batch_size=int(cfg["data"].get("eval_batch_size", batch)),
        shuffle=False, drop_last=False, **common)

    meta = {
        "classes": classes,
        "n_train": len(train_ds),
        "n_test": len(test_ds),
        "n_users_train": train_df["user_id"].nunique(),
        "n_users_test": test_df["user_id"].nunique(),
        "manifest": str(manifest),
        "manifest_sha256": manifest_sha256(manifest),
        "mask_column": mask_column,
    }
    return train_loader, test_loader, meta
