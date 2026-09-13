"""Training curves per architecture (seeds overlaid) and a comparison figure.

    python scripts/plot_curves.py --prefix eixo1 --out results/eixo1/figures
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
LABEL = {"resnet50": "ResNet-50", "densenet121": "DenseNet-121"}


def load_runs(runs_dir: Path, prefix: str) -> dict[str, dict[int, pd.DataFrame]]:
    """Runs with at least one finished epoch; in-progress runs are included
    (their curves are partial and labelled as such)."""
    out: dict[str, dict[int, pd.DataFrame]] = {}
    for csv in sorted(runs_dir.glob(f"{prefix}_*/metrics.csv")):
        name = csv.parent.name                       # eixo1_resnet50_s0
        arch, seed = name[len(prefix) + 1:].rsplit("_s", 1)
        h = pd.read_csv(csv)
        if len(h) == 0:
            continue
        h["partial"] = not (csv.parent / "metrics.json").exists()
        out.setdefault(arch, {})[int(seed)] = h
    return out


def save(fig, out: Path, stem: str) -> None:
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / f"{stem}.pdf")
    fig.savefig(out / f"{stem}.png", dpi=200)
    plt.close(fig)


def per_arch(arch: str, seeds: dict[int, pd.DataFrame], out: Path, prefix: str) -> None:
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
    for seed, h in sorted(seeds.items()):
        tag = f"s{seed}" + (" (parcial)" if h["partial"].iloc[0] else "")
        ax[0].plot(h.epoch, h.train_loss, "-", alpha=.8, label=f"fit {tag}")
        ax[0].plot(h.epoch, h.val_loss, "--", alpha=.8, label=f"val {tag}")
        ax[1].plot(h.epoch, 100 * h.val_error_rate, "-o", ms=3, label=f"val {tag}")
        ax[2].plot(h.epoch, h.lr, "-", label=f"s{seed}")
    ax[0].set(title="Perda (cross-entropy)", xlabel="época", yscale="log")
    ax[1].set(title="Taxa de erro em val (%)", xlabel="época")
    ax[2].set(title="Learning rate", xlabel="época")
    for a in ax:
        a.grid(alpha=.3)
        a.legend(fontsize=8)
    fig.suptitle(f"{LABEL.get(arch, arch)} — {len(seeds)} seed(s), 30 épocas, fit/val do manifesto v2")
    fig.tight_layout(rect=(0, 0, 1, .94))
    save(fig, out, f"{prefix}_{arch}_curvas")


def comparison(runs: dict, out: Path, prefix: str) -> None:
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    for arch, seeds in sorted(runs.items()):
        hs = [h for h in seeds.values() if not h["partial"].iloc[0]] or list(seeds.values())
        n = min(len(h) for h in hs)
        err = pd.concat([100 * h.val_error_rate.iloc[:n].reset_index(drop=True).astype(float)
                         for h in hs], axis=1, ignore_index=True)
        f1 = pd.concat([100 * h.val_f1_macro.iloc[:n].reset_index(drop=True).astype(float)
                        for h in hs], axis=1, ignore_index=True)
        ep = range(1, n + 1)
        ax[0].plot(ep, err.mean(1), "-o", ms=3, label=LABEL.get(arch, arch))
        ax[0].fill_between(ep, err.mean(1) - err.std(1).fillna(0), err.mean(1) + err.std(1).fillna(0), alpha=.2)
        ax[1].plot(ep, f1.mean(1), "-o", ms=3, label=LABEL.get(arch, arch))
        ax[1].fill_between(ep, f1.mean(1) - f1.std(1).fillna(0), f1.mean(1) + f1.std(1).fillna(0), alpha=.2)
    ax[0].set(title="Taxa de erro em val (%) — média ± std entre seeds", xlabel="época")
    ax[1].set(title="F1 macro em val (%) — média ± std entre seeds", xlabel="época")
    for a in ax:
        a.grid(alpha=.3)
        a.legend()
    fig.tight_layout()
    save(fig, out, f"{prefix}_comparativo")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=Path, default=ROOT / "runs")
    ap.add_argument("--prefix", default="eixo1")
    ap.add_argument("--out", type=Path, default=ROOT / "results/eixo1/figures")
    args = ap.parse_args()
    runs = load_runs(args.runs, args.prefix)
    if not runs:
        print("nenhum metrics.csv encontrado")
        return 1
    for arch, seeds in runs.items():
        per_arch(arch, seeds, args.out, args.prefix)
    comparison(runs, args.out, args.prefix)
    print(f"figuras -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
