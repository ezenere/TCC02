"""Axis-4 masks: nested, stratified, confined to fit, deterministic."""

from pathlib import Path

import pandas as pd
import pytest

from make_masks import FRACTIONS, SEEDS, add_masks, check, column

ROOT = Path(__file__).resolve().parents[1]
V3 = ROOT / "data/processed/manifest_v3.csv"


def synthetic():
    rows = []
    for u in range(60):
        inner = "fit" if u < 40 else ("val" if u < 48 else "test")
        for i in range(20):
            rows.append({"image_path": f"{u}_{i}.jpg", "label": f"c{i % 4}", "user_id": f"u{u}",
                         "split": "test" if inner == "test" else "train", "inner_split": inner})
    return pd.DataFrame(rows)


def test_masks_nested_stratified_and_confined():
    df = add_masks(synthetic(), "deadbeef")
    check(df)                                          # asserts nesting, shares, stratification
    assert len(df.columns) == 5 + len(FRACTIONS) * len(SEEDS)


def test_masks_deterministic_and_seed_dependent():
    a = add_masks(synthetic(), "deadbeef")
    b = add_masks(synthetic(), "deadbeef")
    c = add_masks(synthetic(), "cafebabe")
    assert a.equals(b)
    assert not a[column(0.5, 0)].equals(c[column(0.5, 0)])
    assert not a[column(0.5, 0)].equals(a[column(0.5, 1)])


def test_val_and_test_untouched():
    src = synthetic()
    df = add_masks(src, "deadbeef")
    assert df[src.columns].equals(src)
    for k in SEEDS:
        for f in FRACTIONS:
            assert not df.loc[df.inner_split != "fit", column(f, k)].any()


@pytest.mark.skipif(not V3.exists(), reason="manifest v3 not on disk")
def test_real_manifest_v3():
    df = pd.read_csv(V3, dtype=str, keep_default_na=False)
    for k in SEEDS:
        for f in FRACTIONS:
            df[column(f, k)] = df[column(f, k)].str.lower().isin({"true", "1"})
    check(df)
