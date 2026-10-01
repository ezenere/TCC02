"""Extra B: does compression hurt more when the model was trained on less data?

Compares, for ResNet-50 seed 0, the error ratio of each compression step relative
to the dense model of the SAME data regime: 100% of the training data (axis-3
cells) vs 25% (runs/eixo4_resnet50_f025_s0 and runs/extraB_resnet50_f025_p90_s0).
Writes results/analise/interacao.csv and figures.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent


def errs(run: str, fname: str = "metrics.json") -> int | None:
    p = ROOT / "runs" / run / fname
    return json.loads(p.read_text())["n_errors"] if p.exists() else None


def main() -> int:
    regimes = {"100%": {"dense": "eixo1_resnet50_s0", "pruned": "eixo2_prune_resnet50_p90_s0"},
               "25%": {"dense": "eixo4_resnet50_f025_s0", "pruned": "extraB_resnet50_f025_p90_s0"}}
    cells = [("baseline FP32", "dense", "metrics.json"), ("int8 CPU", "dense", "metrics_int8_fbgemm.json"),
             ("int8 TensorRT", "dense", "metrics_trt_int8.json"), ("poda 90% + fine-tuning", "pruned", "metrics.json"),
             ("poda 90% + int8 CPU", "pruned", "metrics_int8_fbgemm.json"), ("poda 90% + int8 TensorRT", "pruned", "metrics_trt_int8.json")]
    rows = []
    for regime, runs in regimes.items():
        base = errs(runs["dense"])
        for label, which, fname in cells:
            e = errs(runs[which], fname)
            rows.append({"dados": regime, "celula": label, "erros": e, "razao_vs_denso_mesmo_regime": (e / base) if e else None})
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "interacao.csv", index=False)
    piv = df.pivot(index="celula", columns="dados", values="razao_vs_denso_mesmo_regime").reindex([c[0] for c in cells])
    print(df.pivot(index="celula", columns="dados", values="erros").reindex([c[0] for c in cells]).to_string())
    print(); print(piv.round(2).to_string())

    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    x = range(len(piv))
    for i, (col, color) in enumerate((("100%", "#2a78d6"), ("25%", "#eb6834"))):
        vals = piv[col].fillna(0)
        ax.bar([p + (i - .5) * .36 for p in x], vals, width=.34, color=color, label=f"treinado com {col} dos dados")
        for p, v in zip(x, vals):
            if v:
                ax.text(p + (i - .5) * .36, v + .02, f"{v:.2f}".replace(".", ","), ha="center", fontsize=8.5)
    ax.axhline(1.0, color="gray", lw=.8, ls=":")
    ax.axhline(1.5, color="red", lw=.8, ls="--")
    ax.set_xticks(list(x), [c.replace(" + ", "\n+ ") for c in piv.index], fontsize=8.5)
    ax.set_ylabel("razão de erro vs denso do mesmo regime")
    ax.set_title("ResNet-50, seed 0: a compressão custa o mesmo com 100% ou 25% dos dados?", fontsize=11, loc="left")
    ax.grid(axis="y", alpha=.3); ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT / "interacao.pdf"); fig.savefig(OUT / "interacao.png", dpi=200)
    print(f"-> {OUT / 'interacao.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
