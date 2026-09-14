"""Manifest v3: boolean data-fraction masks for axis 4, on top of manifest v2.

Columns frac_{75,50,25,10,5}_s{0..4}. For each seed K, every class of `fit`
is permuted once with a RNG seeded from (sha256 of manifest v2, K); fraction f
keeps the first round(f * n_class) images of that permutation. Hence the masks
are stratified by class (exact up to rounding) and nested (25% ⊂ 50% ⊂ 75%
⊂ fit) by construction. val and test rows are False in every mask column and
otherwise untouched: the validation set and the test set never change with
the training fraction.
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

FRACTIONS = (0.75, 0.50, 0.25, 0.10, 0.05)
SEEDS = (0, 1, 2, 3, 4)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def mask_seed(base_sha: str, k: int) -> int:
    return int(hashlib.sha256(f"{base_sha}:{k}".encode()).hexdigest()[:8], 16)


def column(frac: float, k: int) -> str:
    return f"frac_{int(round(100 * frac))}_s{k}"


def add_masks(df: pd.DataFrame, base_sha: str, fractions=FRACTIONS, seeds=SEEDS) -> pd.DataFrame:
    """Return a copy of df with the mask columns appended (row order unchanged)."""
    out = df.copy()
    fit_idx = np.flatnonzero((out["inner_split"] == "fit").to_numpy())
    labels = out["label"].to_numpy()
    for k in seeds:
        rng = np.random.default_rng(mask_seed(base_sha, k))
        keep = {f: np.zeros(len(out), dtype=bool) for f in fractions}
        for cls in sorted(set(labels[fit_idx])):
            idx = fit_idx[labels[fit_idx] == cls]
            perm = rng.permutation(idx)                # one permutation per (class, seed)
            for f in fractions:
                keep[f][perm[: int(round(f * len(idx)))]] = True
        for f in fractions:
            out[column(f, k)] = keep[f]
    return out


def check(df: pd.DataFrame, fractions=FRACTIONS, seeds=SEEDS, tol=1e-3) -> None:
    fit = df["inner_split"] == "fit"
    n_fit_cls = df[fit]["label"].value_counts()
    for k in seeds:
        prev = None
        for f in sorted(fractions, reverse=True):
            col = df[column(f, k)].astype(bool)
            assert not (col & ~fit).any(), f"{column(f, k)}: True outside fit"
            share = col.sum() / fit.sum()
            assert abs(share - f) < tol, f"{column(f, k)}: share {share:.5f} vs {f}"
            per_cls = df[col]["label"].value_counts() / n_fit_cls
            assert (per_cls - f).abs().max() < tol, f"{column(f, k)}: stratification off"
            if prev is not None:
                assert not (col & ~prev).any(), f"{column(f, k)} not nested in previous fraction"
            prev = col


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--v2", type=Path, default=Path("data/processed/manifest_v2.csv"))
    ap.add_argument("--out", type=Path, default=Path("data/processed/manifest_v3.csv"))
    args = ap.parse_args()

    base_sha = sha256(args.v2)
    df = pd.read_csv(args.v2, dtype=str, keep_default_na=False)
    v3 = add_masks(df, base_sha)
    check(v3)
    v3.to_csv(args.out, index=False, lineterminator="\n")
    print(f"manifesto v3 -> {args.out}")
    print(f"sha256        : {sha256(args.out)}")
    print(f"base (v2)     : {base_sha}")
    print(f"colunas       : {len(v3.columns)} ({len(FRACTIONS) * len(SEEDS)} máscaras)")
    n_fit = int((v3.inner_split == "fit").sum())
    for f in FRACTIONS:
        print(f"  frac_{int(100 * f):>2}: {int(v3[column(f, 0)].sum()):>8,} imgs (s0)  de {n_fit:,} fit")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
