"""Visual check of crop margin and framing.

Renders N images as rows: original with the bbox (green) and the margin box
(orange) drawn, then the resulting crop in each configured mode. Lets the
`crop.mode` / `crop.margin` decision be made by looking, before the full run.
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import yaml
from PIL import Image

from preprocess import build_plan, crop_box

MODES = ["square", "stretch"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=Path("configs/preprocess.yaml"))
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--out", type=Path,
                    default=Path("results/figures/crop_check.png"))
    args = ap.parse_args()

    cfg = yaml.safe_load(args.config.read_text())
    img_root = Path(cfg["paths"]["images"])
    margin = float(cfg["crop"]["margin"])
    size = int(cfg["crop"]["size"])

    subset, _, _ = build_plan(cfg)
    # Same seed as --sample, so these rows come from the sampled batch.
    picks = random.Random(int(cfg["seed"]) + 99).sample(subset, 200)[: args.n]

    fig, axes = plt.subplots(args.n, 1 + len(MODES),
                             figsize=(3.1 * (1 + len(MODES)), 3.1 * args.n))
    for row, rec in enumerate(picks):
        src = img_root / rec["label"] / f"{rec['key']}.jpg"
        with Image.open(src) as im:
            im = im.convert("RGB")
            W, H = im.size

            ax = axes[row, 0]
            ax.imshow(im)
            x, y, w, h = rec["bbox"]
            ax.add_patch(patches.Rectangle((x * W, y * H), w * W, h * H,
                                           lw=2, edgecolor="lime", facecolor="none"))
            mx0, my0 = (x - w * margin) * W, (y - h * margin) * H
            ax.add_patch(patches.Rectangle(
                (mx0, my0), w * (1 + 2 * margin) * W, h * (1 + 2 * margin) * H,
                lw=1.6, edgecolor="orange", ls="--", facecolor="none"))
            ax.set_title(f"{rec['label']}  ({W}x{H})", fontsize=9)
            ax.axis("off")

            for col, mode in enumerate(MODES, start=1):
                box = crop_box(rec["bbox"], W, H, margin, mode)
                crop = im.crop(box).resize((size, size), Image.BICUBIC)
                axes[row, col].imshow(crop)
                axes[row, col].set_title(
                    f"{mode}  {box[2]-box[0]}x{box[3]-box[1]} -> {size}x{size}",
                    fontsize=9)
                axes[row, col].axis("off")

    fig.suptitle(f"Validacao de crop: bbox (verde) + margem {margin:.0%} (laranja)",
                 fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.99))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=110, bbox_inches="tight")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
