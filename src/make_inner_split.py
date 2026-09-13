"""Manifest v2: add `inner_split` (fit / val) to the training rows.

About 10% of the training *subjects* become the internal validation set used
for model selection and early stopping in the definitive runs. The test split
is never touched: every v1 row is carried over unchanged, in the same order,
and only the new column is appended.

The selection reuses the deterministic class-balanced user sampler from
preprocess.py, so the result depends only on (train users, seed).
"""

from __future__ import annotations

import argparse
import hashlib
from collections import defaultdict
from pathlib import Path

import pandas as pd
import yaml

from preprocess import select_users_balanced

V1_COLUMNS = ["image_path", "label", "user_id", "split"]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=Path("configs/preprocess.yaml"))
    ap.add_argument("--v1", type=Path, default=Path("data/processed/manifest.csv"))
    ap.add_argument("--out", type=Path, default=Path("data/processed/manifest_v2.csv"))
    ap.add_argument("--val-fraction", type=float, default=0.10)
    args = ap.parse_args()

    cfg = yaml.safe_load(args.config.read_text())
    seed = int(cfg["seed"]) + 2          # +0 subset, +1 train/test, +2 inner val
    tolerances = list(cfg["subset"]["tolerances"])

    df = pd.read_csv(args.v1, dtype=str, keep_default_na=False)
    assert list(df.columns) == V1_COLUMNS, f"unexpected v1 columns: {list(df.columns)}"

    train = df[df["split"] == "train"]
    counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for uid, label in zip(train["user_id"], train["label"]):
        counts[uid][label] += 1
    counts = {u: dict(c) for u, c in counts.items()}
    class_totals = train["label"].value_counts().to_dict()

    val_users = select_users_balanced(
        counts, class_totals, args.val_fraction, tolerances, seed)

    inner = []
    for split, uid in zip(df["split"], df["user_id"]):
        if split == "test":
            inner.append("test")
        else:
            inner.append("val" if uid in val_users else "fit")
    df["inner_split"] = inner

    # ---- invariants: fail loudly ------------------------------------------
    fit_users = set(df.loc[df.inner_split == "fit", "user_id"])
    test_users = set(df.loc[df.split == "test", "user_id"])
    assert not (val_users & fit_users), "val/fit share subjects"
    assert not (val_users & test_users), "val/test share subjects"
    assert not (fit_users & test_users), "fit/test share subjects"
    assert set(df.loc[df.inner_split == "val", "label"]) == set(df["label"]), \
        "val is missing at least one class"
    assert (df.loc[df.split == "test", "inner_split"] == "test").all()
    assert df.loc[df.split == "train", "inner_split"].isin(["fit", "val"]).all()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False, lineterminator="\n")

    n = df.inner_split.value_counts()
    users = df.groupby("inner_split")["user_id"].nunique()
    print(f"manifesto v2 -> {args.out}")
    print(f"sha256        : {sha256(args.out)}")
    print(f"seed          : {seed}")
    for s in ("fit", "val", "test"):
        print(f"  {s:<5} {int(n[s]):>8,} imgs  {int(users[s]):>6,} sujeitos")
    frac = (df[df.inner_split == "val"].label.value_counts()
            / df[df.split == "train"].label.value_counts() * 100)
    print(f"  val por classe: {frac.min():.2f}% .. {frac.max():.2f}% do treino")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
