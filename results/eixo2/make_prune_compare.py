"""Eixo 2 vs 2b: prune AFTER training (+5-epoch fine-tuning) vs prune BEFORE training
(ImageNet weights pruned, then the full 30-epoch recipe with fixed masks).

    python results/eixo2/make_prune_compare.py

Error ratio is against the dense axis-1 baseline of the same seed and architecture.
Writes prune_compare_runs.csv, prune_compare_summary.csv, figures/prune_compare.{pdf,png}
and the marked block of README.md.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
LABEL = {"resnet50": "ResNet-50", "densenet121": "DenseNet-121"}
PATS = {"depois (treino → poda → fine-tuning 5 ép.)": re.compile(r"eixo2_prune_(?P<arch>\w+?)_p(?P<pct>\d+)_s(?P<seed>\d+)$"),
        "antes (poda ImageNet → treino 30 ép.)": re.compile(r"eixo2b_prunefirst_(?P<arch>\w+?)_p(?P<pct>\d+)_s(?P<seed>\d+)$"),
        "controle (treino → poda → 30 ép., LR rewinding)": re.compile(r"eixo2c_rewind_(?P<arch>\w+?)_p(?P<pct>\d+)_s(?P<seed>\d+)$")}


def collect() -> pd.DataFrame:
    base = {}
    for m in (ROOT / "runs").glob("eixo1_*/metrics.json"):
        d = json.loads(m.read_text())
        base[(d["arch"], int(d["seed"]))] = d["error_rate"]
    rows = []
    for method, pat in PATS.items():
        for m in sorted((ROOT / "runs").glob("eixo2*_p*_s*/metrics.json")):
            g = pat.match(m.parent.name)
            if not g:
                continue
            d = json.loads(m.read_text())
            hist = pd.read_csv(m.parent / "metrics.csv")
            b = base.get((g["arch"], int(g["seed"])), np.nan)
            rows.append({"method": method, "arch": g["arch"], "sparsity": int(g["pct"]), "seed": int(g["seed"]),
                         "run": m.parent.name, "acc": d["acc"], "f1_macro": d["f1_macro"], "error_rate": d["error_rate"],
                         "n_errors": d["n_errors"], "error_ratio": d["error_rate"] / b,
                         "sparsity_achieved": d.get("sparsity_achieved"), "best_epoch": d["epoch"],
                         "epochs": len(hist), "gpu_min": hist.epoch_time_s.sum() / 60})
    return pd.DataFrame(rows)


def main() -> int:
    df = collect()
    if df.empty:
        print("sem runs")
        return 1
    df.to_csv(OUT / "prune_compare_runs.csv", index=False)
    g = df.groupby(["arch", "sparsity", "method"])
    s = pd.DataFrame({"n_seeds": g.size(), "err_mean": g.error_rate.mean(), "ratio_mean": g.error_ratio.mean(),
                      "ratio_std": g.error_ratio.std(ddof=1), "errors": g.n_errors.apply(lambda x: ", ".join(map(str, x))),
                      "gpu_min": g.gpu_min.mean()}).reset_index()
    s.to_csv(OUT / "prune_compare_summary.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4), sharey=True)
    for ax, (arch, grp) in zip(axes, s.groupby("arch")):
        for method, sub in grp.groupby("method"):
            sub = sub.sort_values("sparsity")
            ax.errorbar(sub.sparsity, sub.ratio_mean, yerr=sub.ratio_std.fillna(0), fmt="-o", capsize=3, label=method)
        ax.axhline(1.0, color="gray", ls=":", lw=.8)
        ax.axhline(1.5, color="red", ls="--", lw=.8)
        ax.set(title=LABEL.get(arch, arch), xlabel="esparsidade global (%)")
        ax.grid(alpha=.3)
    axes[0].set_ylabel("razão de erro vs baseline denso (mesma seed)")
    axes[0].legend(fontsize=8, title="quando podar")
    fig.tight_layout()
    (OUT / "figures").mkdir(exist_ok=True)
    fig.savefig(OUT / "figures/prune_compare.pdf")
    fig.savefig(OUT / "figures/prune_compare.png", dpi=200)

    pd.set_option("display.width", 200)
    piv = s.pivot_table(index=["arch", "sparsity"], columns="method", values="ratio_mean").round(2)
    print(piv.to_string())
    lines = ["| arquitetura | esparsidade | poda depois, 5 ép.: razão (erros) | poda antes, 30 ép.: razão (erros) | controle: poda depois, 30 ép. (erros) | min de GPU |", "|---|---|---|---|---|---|"]
    for (arch, sp), grp in s.groupby(["arch", "sparsity"]):
        cell = {r.method[:5]: r for r in grp.itertuples()}
        fmt = lambda r: (f"{r.ratio_mean:.2f}" + (f" ± {r.ratio_std:.2f}" if pd.notna(r.ratio_std) else "") + f" ({r.errors})") if r is not None else "—"
        gm = lambda r: f"{r.gpu_min:.0f}" if r is not None else "—"
        a, b, c = cell.get("depoi"), cell.get("antes"), cell.get("contr")
        lines.append(f"| {LABEL.get(arch, arch)} | {sp}% | {fmt(a)} | {fmt(b)} | {fmt(c)} | {gm(a)} / {gm(b)} / {gm(c)} |")
    readme = OUT / "README.md"
    txt = readme.read_text()
    a, b = "<!-- eixo2:compare:start -->", "<!-- eixo2:compare:end -->"
    if a in txt and b in txt:
        pre, rest = txt.split(a, 1)
        readme.write_text(pre + a + "\n" + "\n".join(lines) + "\n" + b + rest.split(b, 1)[1])
    print(f"-> {OUT / 'prune_compare_summary.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
