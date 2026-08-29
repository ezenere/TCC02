"""Figures for the baseline evidence: training curves and confusion matrix."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def curves(hist: pd.DataFrame, out_dir: Path, stem: str) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))

    ax1.plot(hist.epoch, hist.train_loss, "o-", label="treino")
    ax1.plot(hist.epoch, hist.test_loss, "s-", label="teste")
    ax1.set_xlabel("epoca")
    ax1.set_ylabel("perda (cross-entropy)")
    ax1.set_title("Perda")
    ax1.grid(alpha=0.3)
    ax1.legend()

    ax2.plot(hist.epoch, 100 * hist.train_acc, "o-", label="acuracia treino")
    ax2.plot(hist.epoch, 100 * hist.test_acc, "s-", label="acuracia teste")
    ax2.plot(hist.epoch, 100 * hist.test_f1_macro, "^--", label="F1 macro teste")
    ax2.set_xlabel("epoca")
    ax2.set_ylabel("%")
    ax2.set_title("Acuracia e F1 macro")
    ax2.grid(alpha=0.3)
    ax2.legend()

    fig.suptitle("Baseline ResNet-50 (ImageNet) - 5 epocas, holdout 70/30 "
                 "por sujeito", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    for ext in ("pdf", "png"):
        fig.savefig(out_dir / f"{stem}_curvas.{ext}", dpi=300)
    plt.close(fig)


def confusion(cm: np.ndarray, classes: list[str], out_dir: Path, stem: str) -> None:
    norm = cm.astype(float) / np.maximum(cm.sum(axis=1, keepdims=True), 1)

    fig, ax = plt.subplots(figsize=(9.5, 8.4))
    im = ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(classes)), classes, rotation=90, fontsize=8)
    ax.set_yticks(range(len(classes)), classes, fontsize=8)
    ax.set_xlabel("predito")
    ax.set_ylabel("verdadeiro")
    ax.set_title("Matriz de confusao normalizada por linha (teste com 83.613 imagens)")

    # Annotate only the cells that carry information, to keep the grid readable.
    for i in range(len(classes)):
        for j in range(len(classes)):
            if norm[i, j] >= 0.01:
                ax.text(j, i, f"{100 * norm[i, j]:.0f}", ha="center", va="center",
                        fontsize=6.5,
                        color="white" if norm[i, j] > 0.5 else "black")
    fig.colorbar(im, ax=ax, fraction=0.046, label="fracao da classe verdadeira")
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(out_dir / f"{stem}_confusao.{ext}", dpi=300)
    plt.close(fig)


def per_class_table(cm: np.ndarray, classes: list[str], out: Path) -> pd.DataFrame:
    tp = np.diag(cm).astype(float)
    support = cm.sum(axis=1)
    precision = tp / np.maximum(cm.sum(axis=0), 1)
    recall = tp / np.maximum(support, 1)
    f1 = 2 * precision * recall / np.maximum(precision + recall, 1e-12)
    df = pd.DataFrame({
        "classe": classes, "suporte": support,
        "precisao": precision.round(4), "revocacao": recall.round(4),
        "f1": f1.round(4),
    }).sort_values("f1")
    df.to_csv(out, index=False)
    return df


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, default=Path("results/figures"))
    args = ap.parse_args()

    hist = pd.read_csv(args.run_dir / "metrics.csv")
    meta = json.loads((args.run_dir / "run_meta.json").read_text())
    classes = meta["classes"]
    cm = np.load(args.run_dir / "confusion_matrix.npy")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    stem = args.run_dir.name
    curves(hist, args.out_dir, stem)
    confusion(cm, classes, args.out_dir, stem)
    table = per_class_table(cm, classes, Path("results") / f"{stem}_per_class.csv")

    print(f"figuras -> {args.out_dir}/{stem}_curvas.[pdf|png], "
          f"{args.out_dir}/{stem}_confusao.[pdf|png]")
    print(f"\npiores 5 classes por F1:\n{table.head(5).to_string(index=False)}")
    print(f"\nmelhores 5 classes por F1:\n{table.tail(5).to_string(index=False)}")


if __name__ == "__main__":
    main()
