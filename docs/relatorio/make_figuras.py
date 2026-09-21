"""Plain-language figures of the report, generated from results/*.csv.

    python docs/relatorio/make_figuras.py

Colour follows the network everywhere: ResNet-50 blue, DenseNet-121 orange
(palette validated for colour-blind separation). Text stays in ink colours.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent / "figuras"
C = {"resnet50": "#2a78d6", "densenet121": "#eb6834"}
NAME = {"resnet50": "ResNet-50", "densenet121": "DenseNet-121"}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10.5, "axes.edgecolor": GRID, "axes.labelcolor": MUTED,
                     "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
                     "figure.facecolor": "white", "axes.facecolor": "white", "legend.frameon": False})


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{name}.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def fig_poda():
    s = pd.read_csv(ROOT / "results/eixo2/pruning_summary.csv")
    fig, ax = plt.subplots(figsize=(7.2, 3.9))
    for arch, g in s.groupby("arch"):
        g = g.sort_values("sparsity_target")
        x = range(len(g))
        ax.fill_between(x, g.err_ratio_mean - g.err_ratio_std, g.err_ratio_mean + g.err_ratio_std, color=C[arch], alpha=.10, lw=0)
        ax.plot(x, g.err_ratio_mean, "-o", color=C[arch], lw=2, ms=7, mec="white", mew=2, label=NAME[arch])
        ax.annotate(f"{g.err_ratio_mean.iloc[-1]:.2f}×".replace(".", ","), (len(g) - 1, g.err_ratio_mean.iloc[-1]),
                    xytext=(8, 0), textcoords="offset points", va="center", color=INK, fontsize=10)
    ax.axhline(1.5, color=MUTED, lw=1)
    ax.text(0, 1.53, "limite do critério: 1,5×", color=MUTED, fontsize=9)
    ax.axhline(1.0, color=GRID, lw=1)
    ax.set_xticks(range(5), ["50%", "70%", "90%", "95%", "98%"])
    ax.set_xlim(-.3, 4.6)
    ax.set_xlabel("poda (% dos pesos zerados) + fine-tuning de 5 épocas")
    ax.set_ylabel("razão de erro vs baseline")
    ax.set_yticks([1.0, 1.25, 1.5, 1.75, 2.0], ["igual", "1,25×", "1,5×", "1,75×", "2×"])
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper left", bbox_to_anchor=(0, .88))
    ax.set_title("Poda de 90% não aumenta os erros; o joelho está em 98%", loc="left", color=INK, fontsize=12, pad=10)
    save(fig, "poda")


def fig_velocidade():
    rows = [("CPU FP32\n(16 threads)", 10.75, 16.64), ("CPU int8\n(fbgemm)", 1.38, 2.86),
            ("GPU TensorRT\nFP32", 2.65, 4.23), ("GPU TensorRT\nFP16", 0.94, 2.76),
            ("GPU TensorRT\nINT8", 0.80, 3.17)]
    fig, ax = plt.subplots(figsize=(7.2, 4.3))
    h = 0.34
    for i, (lab, r, d) in enumerate(rows):
        for off, arch, v in ((-h / 2 - .02, "resnet50", r), (h / 2 + .02, "densenet121", d)):
            ax.barh(i + off, v, height=h, color=C[arch], label=NAME[arch] if i == 0 else None)
            ax.text(v + .25, i + off, (f"{v:.1f}" if v >= 5 else f"{v:.2f}").replace(".", ","),
                    va="center", color=INK, fontsize=9.5)
    ax.set_yticks(range(len(rows)), [r[0] for r in rows], color=INK)
    ax.invert_yaxis()
    ax.set_xlabel("latência p50, batch 1 (ms) — menor é melhor")
    ax.grid(axis="y", visible=False)
    ax.set_xlim(0, 19)
    ax.set_xticks([0, 5, 10, 15])
    ax.legend(loc="lower right")
    ax.set_title("int8 é até 8 vezes mais rápido em CPU", loc="left", color=INK, fontsize=12, pad=10)
    save(fig, "velocidade")


def fig_dados():
    s = pd.read_csv(ROOT / "results/eixo4/eixo4_summary.csv")
    g = s[s.stage == "dense"].sort_values("frac")
    x = range(len(g))
    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    ax.fill_between(x, 100 * (g.err_mean - g.err_std), 100 * (g.err_mean + g.err_std), color=C["resnet50"], alpha=.10, lw=0)
    ax.plot(x, 100 * g.err_mean, "-o", color=C["resnet50"], lw=2, ms=7, mec="white", mew=2)
    for i in x:
        ax.annotate(f"{100 * g.err_mean.iloc[i]:.2f}%".replace(".", ","), (i, 100 * g.err_mean.iloc[i]),
                    xytext=(0, 10), textcoords="offset points", ha="center", color=INK, fontsize=9.5)
    ax.set_xticks(list(x), [f"{f}%\n{int(round(n / 1000))} mil" for f, n in zip(g.frac, g.n_fit)])
    ax.set_xlabel("fração do treino (imagens de fit)")
    ax.set_ylabel("taxa de erro no teste")
    ax.set_yticks([0, .2, .4, .6, .8], ["0", "0,2%", "0,4%", "0,6%", "0,8%"])
    ax.set_ylim(0, .85)
    ax.grid(axis="x", visible=False)
    ax.set_title("Com 25% do treino o erro sobe pouco; abaixo disso, sobe rápido", loc="left", color=INK, fontsize=12, pad=10)
    save(fig, "dados")


def fig_poda_metodos():
    s = pd.read_csv(ROOT / "results/eixo2/prune_compare_summary.csv")
    s = s[s.sparsity == 98]
    order = [("depois", "poda depois\n+ fine-tuning\n5 ép."), ("antes", "poda antes\n+ treino\n30 ép."),
             ("contro", "poda depois\n+ treino\n30 ép.")]
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.7), sharey=True)
    for ax, arch in zip(axes, ("resnet50", "densenet121")):
        g = s[s.arch == arch]
        for i, (key, lab) in enumerate(order):
            r = g[g.method.str.startswith(key)]
            if r.empty:
                continue
            v = float(r.ratio_mean.iloc[0])
            ax.bar(i, v, width=.42, color=C[arch])
            ax.text(i, v + .04, f"{v:.2f}×".replace(".", ","), ha="center", color=INK, fontsize=10)
        ax.axhline(1.5, color=MUTED, lw=1)
        ax.set_xticks(range(3), [o[1] for o in order], fontsize=8.5, color=INK)
        ax.set_ylim(0, 2.2)
        ax.set_title(NAME[arch], color=INK, fontsize=11)
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel("razão de erro vs baseline")
    axes[0].set_yticks([0, .5, 1.0, 1.5, 2.0], ["0", "0,5×", "igual", "1,5×", "2×"])
    axes[1].text(2.45, 1.54, "limite 1,5×", color=MUTED, fontsize=8.5, ha="right")
    fig.suptitle("Poda de 98%: o que decide é o número de épocas depois da poda", x=.02, ha="left", color=INK, fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, .93))
    save(fig, "poda_metodos")


def fig_gestos():
    """Smaller copy of the 18-gesture grid (the original is a 3600 px print figure)."""
    from PIL import Image
    im = Image.open(ROOT / "results/figures/fig2_amostras_classes.png").convert("RGB")
    im = im.crop((0, 190, im.width, im.height))                 # drop the technical title
    im.resize((1500, int(im.height * 1500 / im.width)), Image.LANCZOS).save(OUT / "gestos.jpg", quality=88)


def fig_rotulos():
    """Six test photos that BOTH networks get 'wrong' with full confidence — questionable labels."""
    from PIL import Image
    a = pd.read_csv(ROOT / "results/analise/preds_test_resnet50_s0.csv")
    b = pd.read_csv(ROOT / "results/analise/preds_test_densenet121_s0.csv")
    m = a.merge(b, on=["image_path", "label"], suffixes=("_r", "_d"))
    m = m[(m.pred_r != m.label) & (m.pred_r == m.pred_d)]
    m["conf"] = m[["confidence_r", "confidence_d"]].min(axis=1)
    pick = m.sort_values("conf", ascending=False).drop_duplicates(["label", "pred_r"]).head(6)
    fig, axes = plt.subplots(1, 6, figsize=(9.6, 2.35))
    for ax, r in zip(axes, pick.itertuples()):
        ax.imshow(Image.open(ROOT / r.image_path))
        ax.set_title(f"rótulo: {r.label}\npredito: {r.pred_r}", fontsize=8.5, color=INK)
        ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
        for sp in ax.spines.values():
            sp.set_visible(False)
    fig.tight_layout(w_pad=.6)
    save(fig, "rotulos")


if __name__ == "__main__":
    fig_poda(); fig_velocidade(); fig_dados(); fig_poda_metodos(); fig_gestos(); fig_rotulos()
    print("figuras ->", OUT)
