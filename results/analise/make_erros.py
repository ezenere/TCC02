"""Qualitative error analysis on the test split (seed-0 models).

    python results/analise/make_erros.py

Inputs: preds_test_<arch>_s0.csv written by `src/eval.py --save-preds`.
Outputs: confusion_pairs.csv (most frequent true->pred pairs per architecture),
erros_overlap.json (do both architectures fail on the same images?), and
erros_<arch>.{pdf,png}: the 24 most confident mistakes, labelled true / predicted.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
ARCHS = {"resnet50": "ResNet-50", "densenet121": "DenseNet-121"}


def grid(err: pd.DataFrame, title: str, stem: str) -> None:
    show = err.sort_values("confidence", ascending=False).head(24)
    fig, axes = plt.subplots(3, 8, figsize=(20, 8.4))
    for ax in axes.flat:
        ax.axis("off")
    for ax, r in zip(axes.flat, show.itertuples()):
        ax.imshow(Image.open(ROOT / r.image_path))
        ax.set_title(f"rótulo: {r.label}\npredito: {r.pred} ({100 * r.confidence:.0f}%)", fontsize=9)
    fig.suptitle(title, fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(OUT / f"{stem}.pdf")
    fig.savefig(OUT / f"{stem}.png", dpi=110)
    plt.close(fig)


def main() -> int:
    errs, pairs = {}, []
    for arch, label in ARCHS.items():
        d = pd.read_csv(OUT / f"preds_test_{arch}_s0.csv")
        e = d[d.label != d.pred]
        errs[arch] = e
        top = e.groupby(["label", "pred"]).size().sort_values(ascending=False).head(12)
        for (t, p), n in top.items():
            pairs.append({"arch": arch, "true": t, "pred": p, "n": int(n), "share_of_errors": round(n / len(e), 3)})
        grid(e, f"{label} (seed 0) — os 24 erros mais confiantes no teste ({len(e)} erros em {len(d):,})", f"erros_{arch}")
    pd.DataFrame(pairs).to_csv(OUT / "confusion_pairs.csv", index=False)

    a, b = set(errs["resnet50"].image_path), set(errs["densenet121"].image_path)
    both = errs["resnet50"][errs["resnet50"].image_path.isin(a & b)]
    same_wrong = int((both.set_index("image_path").pred ==
                      errs["densenet121"].set_index("image_path").pred.reindex(both.image_path).values).sum())
    overlap = {"errors_resnet50": len(a), "errors_densenet121": len(b), "errors_in_both": len(a & b),
               "share_of_resnet_errors_shared": round(len(a & b) / len(a), 3),
               "share_of_densenet_errors_shared": round(len(a & b) / len(b), 3),
               "shared_errors_with_same_wrong_class": same_wrong,
               "errors_union": len(a | b)}
    (OUT / "erros_overlap.json").write_text(json.dumps(overlap, indent=2) + "\n")
    grid(both, f"Erradas pelas DUAS arquiteturas ({len(a & b)} imagens) — candidatas a rótulo errado ou imagem ambígua", "erros_ambas")

    print(json.dumps(overlap, indent=2))
    print(pd.DataFrame(pairs).query("arch == 'resnet50'").head(8).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
