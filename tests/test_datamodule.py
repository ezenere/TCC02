"""Invariants of the data pipeline that every axis depends on."""

from pathlib import Path

import pandas as pd
import pytest
from torchvision import transforms

from datamodule import build_transforms, frame_for_split, split_frames

ROOT = Path(__file__).resolve().parents[1]
MANIFEST_V2 = ROOT / "data/processed/manifest_v2.csv"
CFG = {"data": {"image_size": 224, "hflip": True}}


def synthetic_manifest() -> pd.DataFrame:
    """20 users x 10 images; users 0-11 fit, 12-13 val, 14-19 test."""
    rows = []
    for u in range(20):
        inner = "fit" if u < 12 else ("val" if u < 14 else "test")
        split = "test" if inner == "test" else "train"
        for i in range(10):
            rows.append({"image_path": f"img_{u}_{i}.jpg", "label": f"c{i % 3}",
                         "user_id": f"u{u}", "split": split, "inner_split": inner,
                         "frac_50_s0": "True" if i < 5 else "False"})
    return pd.DataFrame(rows)


def test_train_transform_has_no_rotation_or_vflip():
    train_tf, eval_tf = build_transforms(CFG)
    kinds = {type(t) for t in train_tf.transforms}
    forbidden = {transforms.RandomRotation, transforms.RandomVerticalFlip,
                 transforms.RandomAffine, transforms.RandomPerspective}
    assert not (kinds & forbidden)
    assert transforms.RandomCrop in kinds and transforms.RandomHorizontalFlip in kinds
    assert transforms.CenterCrop in {type(t) for t in eval_tf.transforms}


def test_split_frames_are_subject_disjoint_synthetic():
    fit, val, test = split_frames(synthetic_manifest())
    assert not (set(fit.user_id) & set(val.user_id))
    assert not (set(fit.user_id) & set(test.user_id))
    assert not (set(val.user_id) & set(test.user_id))
    assert len(fit) == 120 and len(val) == 20 and len(test) == 60


def test_mask_restricts_only_fit():
    df = synthetic_manifest()
    fit, val, test = split_frames(df, mask_column="frac_50_s0")
    assert len(fit) == 60 and (fit["frac_50_s0"] == "True").all()
    assert len(val) == 20 and len(test) == 60          # untouched


def test_mask_column_missing_raises():
    with pytest.raises(ValueError):
        split_frames(synthetic_manifest(), mask_column="frac_nope")


def test_leakage_is_detected():
    df = synthetic_manifest()
    df.loc[df.user_id == "u0", "inner_split"] = "val"   # u0 now in fit? no: all rows moved
    df.loc[(df.user_id == "u0") & (df.image_path.str.endswith("_0.jpg")), "inner_split"] = "fit"
    with pytest.raises(AssertionError):
        split_frames(df)


def test_frame_for_split_names():
    df = synthetic_manifest()
    assert len(frame_for_split(df, "train")) == 140
    assert len(frame_for_split(df, "val")) == 20
    with pytest.raises(ValueError):
        frame_for_split(df.drop(columns=["inner_split"]), "val")


@pytest.mark.skipif(not MANIFEST_V2.exists(), reason="manifest v2 not on disk")
def test_real_manifest_v2_is_subject_disjoint():
    df = pd.read_csv(MANIFEST_V2, dtype=str, keep_default_na=False)
    fit, val, test = split_frames(df)
    assert len(test) == 83_613
    assert len(fit) + len(val) == 195_102
    assert not (set(fit.user_id) & set(val.user_id))
    assert not (set(fit.user_id) & set(test.user_id))
    assert not (set(val.user_id) & set(test.user_id))


def test_shuffle_labels_permutes_only_fit(tmp_path):
    """Negative-control flag: fit labels are permuted, val/test labels are untouched."""
    import numpy as np
    from PIL import Image
    from datamodule import build_dataloaders
    df = synthetic_manifest()
    for p in df.image_path:
        Image.fromarray(np.zeros((256, 256, 3), dtype=np.uint8)).save(tmp_path / p)
    df.to_csv(tmp_path / "manifest_v2.csv", index=False)
    cfg = {"seed": 0, "data": {"manifest": str(tmp_path / "manifest_v2.csv"), "root": str(tmp_path), "image_size": 224,
                               "hflip": True, "batch_size": 8, "workers": 0}}
    a = build_dataloaders(cfg)
    b = build_dataloaders(cfg, shuffle_labels=True)
    assert a[0].dataset.targets != b[0].dataset.targets                 # fit permuted
    assert sorted(a[0].dataset.targets) == sorted(b[0].dataset.targets)  # same multiset
    assert a[1].dataset.targets == b[1].dataset.targets                 # val untouched
    assert a[2].dataset.targets == b[2].dataset.targets                 # test untouched
