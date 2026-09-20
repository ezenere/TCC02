"""Dataset and DataLoader built from a versioned manifest.

Every experiment reads the manifest and nothing else — the processed image
files are never enumerated from the filesystem. Splits:

  v1 manifest : split ∈ {train, test}
  v2+ manifest: split ∈ {train, test} and inner_split ∈ {fit, val, test}

Definitive runs train on `fit`, select on `val`, and touch `test` exactly once.
Axis 4 restricts `fit` further through a boolean mask column of the manifest.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from runinfo import manifest_version

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
BASE_COLUMNS = ["image_path", "label", "user_id", "split"]
TRUTHY = {"true", "1", "t", "yes"}


def manifest_sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_manifest(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    missing = set(BASE_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"manifest is missing columns: {sorted(missing)}")
    return df


def class_names(df: pd.DataFrame) -> list[str]:
    """Sorted, so the class index is stable across runs and versions."""
    return sorted(df["label"].unique())


def as_bool(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.lower().isin(TRUTHY)


def split_frames(df: pd.DataFrame, mask_column: str | None = None):
    """(fit, val, test) frames. val is None on a v1 manifest.

    The mask only ever restricts `fit`; val and test are never filtered.
    Subject disjointness is asserted here so a hand-edited manifest can never
    silently leak into a run.
    """
    test = df[df["split"] == "test"]
    if "inner_split" in df.columns:
        fit = df[df["inner_split"] == "fit"]
        val = df[df["inner_split"] == "val"]
    else:
        fit = df[df["split"] == "train"]
        val = None

    if mask_column:
        if mask_column not in df.columns:
            raise ValueError(f"manifest has no column '{mask_column}'")
        fit = fit[as_bool(fit[mask_column])]
        if len(fit) == 0:
            raise ValueError(f"mask column '{mask_column}' selects no fit rows")

    fit_users, test_users = set(fit["user_id"]), set(test["user_id"])
    assert not (fit_users & test_users), "subject leakage: fit ∩ test"
    if val is not None:
        val_users = set(val["user_id"])
        assert not (fit_users & val_users), "subject leakage: fit ∩ val"
        assert not (val_users & test_users), "subject leakage: val ∩ test"
    return fit, val, test


def frame_for_split(df: pd.DataFrame, split: str) -> pd.DataFrame:
    """Rows of one named split: fit | val | test | train."""
    if split == "train":
        return df[df["split"] == "train"]
    if split == "test":
        return df[df["split"] == "test"]
    if "inner_split" not in df.columns:
        raise ValueError(f"split '{split}' needs a v2+ manifest with inner_split")
    out = df[df["inner_split"] == split]
    if len(out) == 0:
        raise ValueError(f"split '{split}' is empty")
    return out


class ManifestDataset(Dataset):
    def __init__(self, df: pd.DataFrame, classes: list[str], transform,
                 root: str | Path | None = None):
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
    """Train: random crop 224 from the stored 256 + hflip. Nothing else —
    rotations are forbidden (they map classes onto their *_inverted pairs).
    Eval: deterministic centre crop."""
    size = int(cfg["data"]["image_size"])
    normalize = transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
    train_ops = [transforms.RandomCrop(size)]
    if cfg["data"].get("hflip", True):
        train_ops.append(transforms.RandomHorizontalFlip(p=0.5))
    train_ops += [transforms.ToTensor(), normalize]
    eval_ops = [transforms.CenterCrop(size), transforms.ToTensor(), normalize]
    return transforms.Compose(train_ops), transforms.Compose(eval_ops)


def _loader_kwargs(cfg: dict) -> dict:
    """Workers are NOT persistent by default: a persistent worker keeps its
    augmentation RNG across epochs, so a resumed run (fresh workers) would not
    replay the same crops/flips. Restarting workers costs ~1 s per epoch."""
    workers = int(cfg["data"]["workers"])
    return dict(
        num_workers=workers, pin_memory=True,
        persistent_workers=bool(cfg["data"].get("persistent_workers", False)) and workers > 0,
        prefetch_factor=int(cfg["data"].get("prefetch_factor", 4)) if workers else None,
    )


def build_dataloaders(cfg: dict, mask_column: str | None = None,
                      limit_fit: int | None = None, limit_val: int | None = None,
                      shuffle_labels: bool = False):
    """(fit_loader, val_loader | None, test_loader, meta, generator).

    `generator` drives the fit shuffle; re-seed it per epoch so a resumed run
    replays exactly the same batch order.
    """
    manifest = cfg["data"]["manifest"]
    df = read_manifest(manifest)
    classes = class_names(df)
    fit, val, test = split_frames(df, mask_column)

    seed = int(cfg["seed"])
    if limit_fit:
        fit = fit.sample(n=min(limit_fit, len(fit)), random_state=seed)
    if limit_val and val is not None:
        val = val.sample(n=min(limit_val, len(val)), random_state=seed)

    if shuffle_labels:
        # Negative control: permute the fit labels. A pipeline without label leakage
        # must then score at chance (1/18) on the untouched test split.
        fit = fit.copy()
        fit["label"] = fit["label"].sample(frac=1.0, random_state=seed).to_numpy()

    train_tf, eval_tf = build_transforms(cfg)
    root = cfg["data"].get("root")
    kw = _loader_kwargs(cfg)
    batch = int(cfg["data"]["batch_size"])
    eval_batch = int(cfg["data"].get("eval_batch_size", batch))

    generator = torch.Generator().manual_seed(seed)
    fit_loader = DataLoader(ManifestDataset(fit, classes, train_tf, root),
                            batch_size=batch, shuffle=True, drop_last=True,
                            generator=generator, **kw)
    val_loader = None if val is None else DataLoader(
        ManifestDataset(val, classes, eval_tf, root),
        batch_size=eval_batch, shuffle=False, **kw)
    test_loader = DataLoader(ManifestDataset(test, classes, eval_tf, root),
                             batch_size=eval_batch, shuffle=False, **kw)

    meta = {
        "classes": classes,
        "manifest": str(manifest),
        "manifest_version": manifest_version(manifest),
        "manifest_sha256": manifest_sha256(manifest),
        "mask_column": mask_column,
        "n_fit": len(fit), "n_val": 0 if val is None else len(val), "n_test": len(test),
        "n_users_fit": fit["user_id"].nunique(),
        "n_users_val": 0 if val is None else val["user_id"].nunique(),
        "n_users_test": test["user_id"].nunique(),
        "limit_fit": limit_fit, "limit_val": limit_val, "shuffle_labels": shuffle_labels,
    }
    return fit_loader, val_loader, test_loader, meta, generator


def build_eval_loader(cfg: dict, split: str, manifest: str | None = None):
    """Loader over one split with the deterministic eval transform."""
    manifest = manifest or cfg["data"]["manifest"]
    df = read_manifest(manifest)
    classes = class_names(df)
    frame = frame_for_split(df, split)
    _, eval_tf = build_transforms(cfg)
    loader = DataLoader(
        ManifestDataset(frame, classes, eval_tf, cfg["data"].get("root")),
        batch_size=int(cfg["data"].get("eval_batch_size", cfg["data"]["batch_size"])),
        shuffle=False, **_loader_kwargs(cfg))
    meta = {"classes": classes, "split": split, "n": len(frame),
            "n_users": frame["user_id"].nunique(), "manifest": str(manifest),
            "manifest_version": manifest_version(manifest),
            "manifest_sha256": manifest_sha256(manifest)}
    return loader, meta
