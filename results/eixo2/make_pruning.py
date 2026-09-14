"""Axis 2 (pruning): consolidate runs/eixo2_prune_*/ against the axis-1 baselines.

    python results/eixo2/make_pruning.py            # CSVs + figure + knee per arch
    python results/eixo2/make_pruning.py --extend   # print whether 95/98% are triggered

Error ratio = error_rate(pruned, seed k) / error_rate(baseline eixo1, seed k).
Knee = first sparsity level whose mean error ratio exceeds --knee (default 1.5).
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
PAT = re.compile(r"eixo2_prune_(?P<arch>\w+?)_p(?P<pct>\d+)_s(?P<seed>\d+)$")


def baselines(runs_dir: Path) -> dict[tuple[str, int], dict]:
    out = {}
    for m in runs_dir.glob("eixo1_*/metrics.json"):
        d = json.loads(m.read_text())
        c = json.loads((m.parent / "cost.json").read_text()) if (m.parent / "cost.json").exists() else {}
        out[(d["arch"], int(d["seed"]))] = {**d, **c}
    return out


def collect(runs_dir: Path) -> pd.DataFrame:
    base = baselines(runs_dir)
    rows = []
    for m in sorted(runs_dir.glob("eixo2_prune_*/metrics.json")):
        g = PAT.match(m.parent.name)
        if not g:
            continue
        d = json.loads(m.read_text())
        cost_p = m.parent / "cost.json"
        c = json.loads(cost_p.read_text()) if cost_p.exists() else {}
        b = base.get((g["arch"], int(g["seed"])))
        rows.append({
            "run": m.parent.name, "arch": g["arch"], "seed": int(g["seed"]),
            "sparsity_target": int(g["pct"]) / 100, "sparsity_achieved": d.get("sparsity_achieved"),
            "acc": d["acc"], "f1_macro": d["f1_macro"], "error_rate": d["error_rate"], "n_errors": d["n_errors"],
            "baseline_error_rate": b["error_rate"] if b else np.nan,
            "error_ratio": d["error_rate"] / b["error_rate"] if b else np.nan,
            "params_nonzero": d.get("params_nonzero"), "params_total": c.get("params_total"),
            "macs": c.get("macs"), "fp32_bytes": c.get("state_dict_fp32_bytes"),
            "gzip_bytes": c.get("state_dict_fp32_gzip_bytes"),
            "baseline_gzip_bytes": b.get("state_dict_fp32_gzip_bytes") if b else np.nan,
            "best_epoch": d["epoch"],
        })
    return pd.DataFrame(rows)


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby(["arch", "sparsity_target"])
    s = pd.DataFrame({
        "n_seeds": g.size(),
        "acc_mean": g.acc.mean(), "f1_mean": g.f1_macro.mean(),
        "err_mean": g.error_rate.mean(), "err_std": g.error_rate.std(ddof=1),
        "err_ratio_mean": g.error_ratio.mean(), "err_ratio_std": g.error_ratio.std(ddof=1),
        "params_nonzero": g.params_nonzero.mean(), "gzip_mib": g.gzip_bytes.mean() / 2**20,
        "gzip_ratio_vs_baseline": (g.gzip_bytes.mean() / g.baseline_gzip_bytes.mean()),
    }).reset_index()
    return s


def knees(s: pd.DataFrame, thr: float) -> dict[str, str]:
    out = {}
    for arch, grp in s.groupby("arch"):
        grp = grp.sort_values("sparsity_target")
        over = grp[grp.err_ratio_mean > thr]
        out[arch] = (f"{int(100 * over.sparsity_target.iloc[0])}%" if len(over)
                     else f"nenhum ate {int(100 * grp.sparsity_target.max())}%")
    return out


def figure(s: pd.DataFrame, thr: float) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    for arch, grp in s.groupby("arch"):
        grp = grp.sort_values("sparsity_target")
        ax.errorbar(100 * grp.sparsity_target, grp.err_ratio_mean, yerr=grp.err_ratio_std.fillna(0),
                    fmt="-o", capsize=3, label=LABEL.get(arch, arch))
    ax.axhline(1.0, color="gray", lw=.8, ls=":")
    ax.axhline(thr, color="red", lw=.8, ls="--", label=f"limiar {thr}×")
    ax.set(xlabel="esparsidade global (%)", ylabel="razão de erro vs baseline (mesma seed)",
           title="Poda por magnitude + fine-tuning (5 épocas) — média ± std entre seeds")
    ax.grid(alpha=.3)
    ax.legend()
    fig.tight_layout()
    (OUT / "figures").mkdir(exist_ok=True)
    fig.savefig(OUT / "figures/pruning_error_ratio.pdf")
    fig.savefig(OUT / "figures/pruning_error_ratio.png", dpi=200)
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=Path, default=ROOT / "runs")
    ap.add_argument("--knee", type=float, default=1.5)
    ap.add_argument("--extend", action="store_true", help="report the 95/98% extension rule")
    args = ap.parse_args()

    df = collect(args.runs)
    if df.empty:
        print("nenhum run de poda concluido")
        return 1
    df.to_csv(OUT / "pruning_runs.csv", index=False)
    s = summarize(df)
    s.to_csv(OUT / "pruning_summary.csv", index=False)
    figure(s, args.knee)

    pd.set_option("display.width", 220)
    print("=== runs ===")
    print(df[["run", "sparsity_achieved", "acc", "f1_macro", "error_rate", "n_errors",
              "baseline_error_rate", "error_ratio", "best_epoch"]].to_string(index=False))
    print("\n=== summary ===")
    print(s.to_string(index=False))
    print(f"\njoelho (razao de erro > {args.knee}x): {knees(s, args.knee)}")

    if args.extend:
        at90 = s[s.sparsity_target == 0.9]
        trig = at90[at90.err_ratio_mean < 2.0]
        print("extensao 95/98%: " + ("SIM — " + ", ".join(f"{a} {r:.2f}x @90%" for a, r in
                                    zip(trig.arch, trig.err_ratio_mean)) if len(trig) else "NAO"))
    print(f"\n-> {OUT / 'pruning_runs.csv'}\n-> {OUT / 'pruning_summary.csv'}\n-> {OUT / 'figures/pruning_error_ratio.pdf'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
