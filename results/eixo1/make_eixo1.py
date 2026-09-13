"""Axis 1 results: consolidate runs/eixo1_*/ into CSVs and print the summary.

    python results/eixo1/make_eixo1.py

Works with any number of finished runs (1..6). Every number in the README of
this axis comes from the two CSVs written here — nothing is typed by hand.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent


def collect(runs_dir: Path, prefix: str) -> pd.DataFrame:
    rows = []
    for mpath in sorted(runs_dir.glob(f"{prefix}_*/metrics.json")):
        run = mpath.parent
        m = json.loads(mpath.read_text())
        meta = json.loads((run / "run_meta.json").read_text())
        hist = pd.read_csv(run / "metrics.csv")
        rows.append({
            "run": run.name, "arch": m["arch"], "seed": m["seed"],
            "epochs_run": len(hist), "best_epoch": m["epoch"],
            "test_acc": m["acc"], "test_f1_macro": m["f1_macro"],
            "test_error_rate": m["error_rate"], "test_n_errors": m["n_errors"], "test_n": m["n"],
            "val_f1_best": hist.val_f1_macro.max(), "val_acc_best": hist.val_acc.max(),
            "train_acc_last": hist.train_acc.iloc[-1],
            "epoch_time_s": hist.epoch_time_s.mean(), "throughput_img_s": hist.throughput_img_s.mean(),
            "manifest_version": m["manifest_version"], "manifest_sha256": m["manifest_sha256"][:16],
            "git_commit": (m.get("git_commit") or "")[:10], "stopped_early": meta.get("stopped_early"),
        })
    return pd.DataFrame(rows)


def summarize(runs: pd.DataFrame) -> pd.DataFrame:
    g = runs.groupby("arch")
    s = pd.DataFrame({
        "n_seeds": g.size(),
        "acc_mean": g.test_acc.mean(), "acc_std": g.test_acc.std(ddof=1),
        "f1_mean": g.test_f1_macro.mean(), "f1_std": g.test_f1_macro.std(ddof=1),
        "err_mean": g.test_error_rate.mean(), "err_std": g.test_error_rate.std(ddof=1),
        "epoch_time_s": g.epoch_time_s.mean(), "throughput_img_s": g.throughput_img_s.mean(),
    })
    if "resnet50" in s.index:
        s["err_ratio_vs_resnet50"] = s.err_mean / s.loc["resnet50", "err_mean"]
    return s


def signal_check(s: pd.DataFrame) -> str:
    """Is the architecture difference larger than 2x the between-seed spread?"""
    if len(s) < 2 or s.n_seeds.min() < 2:
        return "sinal: indeterminado (precisa de >= 2 seeds em ambas as arquiteturas)"
    diff = abs(s.err_mean.iloc[0] - s.err_mean.iloc[1])
    spread = 2 * s.err_std.max()
    verdict = "SIM" if diff > spread else "NAO"
    return (f"sinal na dimensao arquitetura: {verdict} — |Δ erro| = {diff:.4%} vs 2×std max = {spread:.4%}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=Path, default=ROOT / "runs")
    ap.add_argument("--prefix", default="eixo1")
    args = ap.parse_args()

    runs = collect(args.runs, args.prefix)
    if runs.empty:
        print("nenhum run concluido (metrics.json ausente)")
        return 1
    runs.to_csv(OUT / "eixo1_runs.csv", index=False)
    summary = summarize(runs)
    summary.to_csv(OUT / "eixo1_summary.csv")

    pd.set_option("display.width", 200)
    print("=== runs ===")
    print(runs[["run", "seed", "epochs_run", "best_epoch", "test_acc", "test_f1_macro",
                "test_error_rate", "test_n_errors", "val_f1_best", "epoch_time_s"]].to_string(index=False))
    print("\n=== summary (teste, media ± std entre seeds) ===")
    print(summary.to_string())
    print("\n" + signal_check(summary))
    print(f"\n-> {OUT / 'eixo1_runs.csv'}\n-> {OUT / 'eixo1_summary.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
