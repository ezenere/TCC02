"""Figure 2 of the project plan: a 3x6 grid with one crop per class.

Cells are chosen to spread illumination and background across the grid: for
each class a deterministic pool of candidates is scored by mean luminance and
background variance, and the pick is the candidate that lies farthest from the
cells already chosen. Output is vector PDF (with embedded bitmaps) plus a
high-resolution PNG, as required by the plan.
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from PIL import Image

N_ROWS, N_COLS = 3, 6
POOL_PER_CLASS = 24


def cell_features(path: Path) -> tuple[float, float]:
    """(mean luminance, border colour spread) proxies for lighting/background."""
    with Image.open(path) as im:
        arr = np.asarray(im.convert("RGB"), dtype=np.float32) / 255.0
    lum = float((0.2126 * arr[..., 0] + 0.7152 * arr[..., 1]
                 + 0.0722 * arr[..., 2]).mean())
    border = np.concatenate([
        arr[:16].reshape(-1, 3), arr[-16:].reshape(-1, 3),
        arr[:, :16].reshape(-1, 3), arr[:, -16:].reshape(-1, 3),
    ])
    return lum, float(border.std())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=Path("configs/preprocess.yaml"))
    ap.add_argument("--out-dir", type=Path, default=Path("results/figures"))
    args = ap.parse_args()

    cfg = yaml.safe_load(args.config.read_text())
    classes = list(cfg["classes"])
    assert len(classes) == N_ROWS * N_COLS, (
        f"grid {N_ROWS}x{N_COLS} needs {N_ROWS * N_COLS} classes, got {len(classes)}")

    df = pd.read_csv(cfg["paths"]["manifest"])
    df = df[df.split == "train"]
    rng = random.Random(int(cfg["seed"]) + 7)

    # Deterministic candidate pool per class, one image per user_id to avoid
    # picking the same subject twice.
    pools: dict[str, list[tuple[float, float, str]]] = {}
    for cls in classes:
        paths = sorted(df[df.label == cls].drop_duplicates("user_id").image_path)
        cands = rng.sample(paths, min(POOL_PER_CLASS, len(paths)))
        pools[cls] = [(*cell_features(Path(p)), p) for p in cands]

    # Greedy spread: each pick maximises distance to the already chosen cells.
    chosen: dict[str, str] = {}
    picked: list[tuple[float, float]] = []
    for cls in classes:
        if not picked:
            lum, bg, path = max(pools[cls], key=lambda t: t[1])
        else:
            def dist(t):
                return min((t[0] - a) ** 2 + (t[1] - b) ** 2 for a, b in picked)
            lum, bg, path = max(pools[cls], key=dist)
        picked.append((lum, bg))
        chosen[cls] = path

    fig, axes = plt.subplots(N_ROWS, N_COLS, figsize=(N_COLS * 2.0, N_ROWS * 2.18))
    for idx, cls in enumerate(classes):
        ax = axes[idx // N_COLS, idx % N_COLS]
        with Image.open(chosen[cls]) as im:
            ax.imshow(im.convert("RGB"), interpolation="antialiased")
        ax.set_title(cls, fontsize=10, pad=4)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_linewidth(0.6)
            spine.set_color("0.7")

    fig.suptitle(
        "Amostras do subconjunto HaGRIDv2 com 18 classes e crops 256x256 "
        "(bbox + margem de 10%)", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.965))

    args.out_dir.mkdir(parents=True, exist_ok=True)
    pdf = args.out_dir / "fig2_amostras_classes.pdf"
    png = args.out_dir / "fig2_amostras_classes.png"
    fig.savefig(pdf)                       # vector page, bitmaps embedded
    fig.savefig(png, dpi=300)
    print(f"wrote {pdf}\nwrote {png}")

    pd.DataFrame(
        [{"label": c, "image_path": chosen[c]} for c in classes]
    ).to_csv(args.out_dir / "fig2_amostras_classes.csv", index=False)


if __name__ == "__main__":
    main()
