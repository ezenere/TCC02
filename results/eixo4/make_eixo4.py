"""Axis 4: error vs training fraction, mean ± std over seeds, per architecture.

    python results/eixo4/make_eixo4.py                    # all archs found in runs/eixo4_*
    python results/eixo4/make_eixo4.py --powerlaw          # + fit error = a * N^-alpha (Hestness)

Reads runs/eixo4_<arch>_f<FFF>_s<K>[_p<SS>]/metrics.json. The reported cell of
a pruned winner is the fine-tuned pruned run (_p<SS>); the dense run of the
same fraction is kept as a secondary row. Error ratio is vs the 100% point of
the same seed and arch (and same stage).
"""

from __future__ import annotations

import argparse
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
PAT = re.compile(r"eixo4_(?P<arch>\w+?)_f(?P<frac>\d{3})_s(?P<seed>\d+)(?:_p(?P<pct>\d+))?$")


def collect(runs_dir: Path) -> pd.DataFrame:
    rows = []
    for m in sorted(runs_dir.glob("eixo4_*/metrics.json")):
        g = PAT.match(m.parent.name)
        if not g:
            continue
        d = json.loads(m.read_text())
        meta = json.loads((m.parent / "run_meta.json").read_text())
        hist = pd.read_csv(m.parent / "metrics.csv")
        rows.append({"run": m.parent.name, "arch": g["arch"], "frac": int(g["frac"]), "seed": int(g["seed"]),
                     "stage": f"prune{g['pct']}" if g["pct"] else "dense",
                     "n_fit": meta.get("n_fit"), "epochs_run": len(hist), "best_epoch": d["epoch"],
                     "stopped_early": meta.get("stopped_early"), "acc": d["acc"], "f1_macro": d["f1_macro"],
                     "error_rate": d["error_rate"], "n_errors": d["n_errors"],
                     "per_class": {c["label"]: c["f1"] for c in d["per_class"]}})
        t = m.parent / "metrics_trt_int8.json"          # winner of axis 3: int8 TensorRT of the same run
        if t.exists():
            q = json.loads(t.read_text())
            rows.append({**rows[-1], "stage": rows[-1]["stage"] + "+trt-int8", "acc": q["acc"], "f1_macro": q["f1_macro"],
                         "error_rate": q["error_rate"], "n_errors": q["n_errors"],
                         "per_class": {c["label"]: c["f1"] for c in q["per_class"]}})
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    ref = df[df.frac == 100].set_index(["arch", "seed", "stage"]).error_rate
    df["error_ratio_vs_100"] = [
        r.error_rate / ref.get((r.arch, r.seed, r.stage), np.nan) for r in df.itertuples()]
    return df


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby(["arch", "stage", "frac"])
    return pd.DataFrame({
        "n_seeds": g.size(), "n_fit": g.n_fit.first(),
        "acc_mean": g.acc.mean(), "acc_std": g.acc.std(ddof=1),
        "f1_mean": g.f1_macro.mean(), "f1_std": g.f1_macro.std(ddof=1),
        "err_mean": g.error_rate.mean(), "err_std": g.error_rate.std(ddof=1),
        "err_ratio_mean": g.error_ratio_vs_100.mean(), "err_ratio_std": g.error_ratio_vs_100.std(ddof=1),
        "epochs_mean": g.epochs_run.mean(), "best_epoch_mean": g.best_epoch.mean(),
    }).reset_index()


def powerlaw(s: pd.DataFrame) -> dict:
    """error = a * N^-alpha on the mean curve (log-log least squares)."""
    out = {}
    for (arch, stage), grp in s.groupby(["arch", "stage"]):
        grp = grp.dropna(subset=["n_fit", "err_mean"])
        if len(grp) < 3:
            continue
        x, y = np.log(grp.n_fit.astype(float)), np.log(grp.err_mean.astype(float))
        slope, intercept = np.polyfit(x, y, 1)
        pred = slope * x + intercept
        r2 = 1 - ((y - pred) ** 2).sum() / ((y - y.mean()) ** 2).sum()
        out[f"{arch}/{stage}"] = {"alpha": float(-slope), "a": float(np.exp(intercept)), "r2": float(r2), "n_points": len(grp)}
    return out


def figures(s: pd.DataFrame, df: pd.DataFrame) -> None:
    (OUT / "figures").mkdir(exist_ok=True)
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.6))
    for (arch, stage), grp in s.groupby(["arch", "stage"]):
        grp = grp.sort_values("frac")
        lab = LABEL.get(arch, arch) + ("" if stage == "dense" else f" ({stage})")
        ax[0].errorbar(grp.frac, 100 * grp.err_mean, yerr=100 * grp.err_std.fillna(0), fmt="-o", capsize=3, label=lab)
        ax[1].errorbar(grp.frac, grp.err_ratio_mean, yerr=grp.err_ratio_std.fillna(0), fmt="-o", capsize=3, label=lab)
    for a, t in zip(ax, ("taxa de erro no teste (%)", "razão de erro vs 100% (mesma seed)")):
        a.set(xscale="log", xlabel="fração do treino (%)", ylabel=t)
        a.set_xticks([5, 10, 25, 50, 75, 100], [5, 10, 25, 50, 75, 100])
        a.grid(alpha=.3, which="both")
        a.legend()
    fig.suptitle("Eixo 4 — redução de dados anotados (30 épocas + early stopping em val; média ± std entre seeds)")
    fig.tight_layout(rect=(0, 0, 1, .94))
    fig.savefig(OUT / "figures/eixo4_curva.pdf")
    fig.savefig(OUT / "figures/eixo4_curva.png", dpi=200)
    plt.close(fig)

    # which classes degrade first: per-class F1 at the extremes (mean over seeds)
    for (arch, stage), grp in df.groupby(["arch", "stage"]):
        fr = sorted(grp.frac.unique())
        if len(fr) < 2:
            continue
        lo, hi = fr[0], fr[-1]
        pc = {f: pd.DataFrame(list(grp[grp.frac == f].per_class)).mean() for f in (lo, hi)}
        delta = (pc[lo] - pc[hi]).sort_values()
        fig, ax = plt.subplots(figsize=(9, 4))
        ax.bar(delta.index, 100 * delta.values)
        ax.set(ylabel=f"Δ F1 por classe (pontos p.), {lo}% − {hi}%", title=f"{LABEL.get(arch, arch)} ({stage}) — quais classes degradam primeiro")
        ax.tick_params(axis="x", rotation=90)
        ax.grid(alpha=.3, axis="y")
        fig.tight_layout()
        fig.savefig(OUT / f"figures/eixo4_por_classe_{arch}_{stage}.pdf")
        fig.savefig(OUT / f"figures/eixo4_por_classe_{arch}_{stage}.png", dpi=200)
        plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=Path, default=ROOT / "runs")
    ap.add_argument("--powerlaw", action="store_true")
    args = ap.parse_args()
    df = collect(args.runs)
    if df.empty:
        print("nenhum run do eixo 4")
        return 1
    df.drop(columns=["per_class"]).to_csv(OUT / "eixo4_runs.csv", index=False)
    s = summarize(df)
    s.to_csv(OUT / "eixo4_summary.csv", index=False)
    figures(s, df)
    pd.set_option("display.width", 220)
    print(s.to_string(index=False))
    if args.powerlaw:
        pl = powerlaw(s)
        (OUT / "eixo4_powerlaw.json").write_text(json.dumps(pl, indent=2) + "\n")
        for k, v in pl.items():
            print(f"lei de potência {k}: erro ∝ N^-{v['alpha']:.3f} (R² {v['r2']:.3f}, {v['n_points']} pontos)")
    print(f"-> {OUT / 'eixo4_runs.csv'}\n-> {OUT / 'eixo4_summary.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
