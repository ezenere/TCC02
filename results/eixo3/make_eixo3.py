"""Axis 3: cross table of every cell (quality x cost x latency) and the decision.

    python results/eixo3/make_eixo3.py            # eixo3_table.csv, eixo3_summary.csv, figures
    python results/eixo3/make_eixo3.py --decide   # + decision.md (criterion fixed on 2026-09-14)

Cells are discovered from runs/: baseline (eixo1_*), pruned (eixo2_prune_*),
int8 CPU (metrics_int8_fbgemm.json), TensorRT (metrics_trt_*.json) and
pruned+int8. Missing measurements are NaN, never invented.
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
P_BASE = re.compile(r"eixo1_(?P<arch>\w+?)_s(?P<seed>\d+)$")
P_PRUNE = re.compile(r"eixo2_prune_(?P<arch>\w+?)_p(?P<pct>\d+)_s(?P<seed>\d+)$")


def jload(p: Path) -> dict:
    return json.loads(p.read_text()) if p.exists() else {}


def latency_of(run: Path, kind: str, device: str, precision: str, threads: int | None, batch: int):
    """p50 ms of one latency JSON, or NaN."""
    t = f"t{threads}" if threads is not None else "t1"
    f = run / f"latency_{kind}_{device}_{precision}_{t}.json"
    d = jload(f)
    return d.get("results", {}).get(f"batch_{batch}", {}).get("p50_ms", np.nan)


def cell_row(arch, seed, cell, run: Path, metrics: dict, cost: dict, base: dict, artifact_bytes, lat: dict) -> dict:
    return {
        "arch": arch, "seed": seed, "cell": cell, "run": run.name,
        "acc": metrics.get("acc"), "f1_macro": metrics.get("f1_macro"),
        "error_rate": metrics.get("error_rate"), "n_errors": metrics.get("n_errors"),
        "error_ratio": (metrics["error_rate"] / base["error_rate"]) if metrics and base else np.nan,
        "params_total": cost.get("params_total"), "params_nonzero": cost.get("params_nonzero"),
        "macs": cost.get("macs"), "artifact_bytes": artifact_bytes, **lat,
    }


def collect(runs_dir: Path) -> pd.DataFrame:
    rows = []
    bases = {}
    for run in sorted(runs_dir.glob("eixo1_*")):
        g = P_BASE.match(run.name)
        if not g or not (run / "metrics.json").exists():
            continue
        arch, seed = g["arch"], int(g["seed"])
        base = jload(run / "metrics.json")
        cost = jload(run / "cost.json")
        bases[(arch, seed)] = base
        lat_cpu = lambda k, p, t, b: latency_of(run, k, "cpu", p, t, b)
        lat_gpu = lambda k, p, b: latency_of(run, k, "cuda", p, None, b)
        # baseline
        rows.append(cell_row(arch, seed, "baseline", run, base, cost, base, cost.get("state_dict_fp32_bytes"), {
            "lat_cpu_t1_b1": lat_cpu("eager", "fp32", 1, 1), "lat_cpu_t16_b1": lat_cpu("eager", "fp32", 16, 1),
            "lat_cpu_t16_b32": lat_cpu("eager", "fp32", 16, 32),
            "lat_gpu_eager_fp32_b1": lat_gpu("eager", "fp32", 1), "lat_gpu_eager_fp16_b1": lat_gpu("eager", "fp16", 1),
            "lat_gpu_trt_b1": lat_gpu("trt", "fp32", 1), "lat_gpu_trt_b32": lat_gpu("trt", "fp32", 32),
            "lat_gpu_trt_fp16_b1": lat_gpu("trt", "fp16", 1)}))
        # int8 CPU
        m8 = jload(run / "metrics_int8_fbgemm.json")
        if m8:
            rows.append(cell_row(arch, seed, "int8-cpu", run, m8, cost, base, m8.get("artifact_bytes"), {
                "lat_cpu_t1_b1": lat_cpu("torchscript", "int8", 1, 1), "lat_cpu_t16_b1": lat_cpu("torchscript", "int8", 16, 1),
                "lat_cpu_t16_b32": lat_cpu("torchscript", "int8", 16, 32)}))
        # TensorRT cells
        for prec in ("fp32", "fp16", "int8"):
            mt = jload(run / f"metrics_trt_{prec}.json")
            if mt:
                rows.append(cell_row(arch, seed, f"trt-{prec}", run, mt, cost, base, mt.get("artifact_bytes"), {
                    "lat_gpu_trt_b1": lat_gpu("trt", prec, 1), "lat_gpu_trt_b32": lat_gpu("trt", prec, 32)}))
    for run in sorted(runs_dir.glob("eixo2_prune_*")):
        g = P_PRUNE.match(run.name)
        if not g or not (run / "metrics.json").exists():
            continue
        arch, seed, pct = g["arch"], int(g["seed"]), int(g["pct"])
        base = bases.get((arch, seed), {})
        m = jload(run / "metrics.json")
        cost = jload(run / "cost.json")
        rows.append(cell_row(arch, seed, f"prune-{pct}", run, m, cost, base, cost.get("state_dict_fp32_gzip_bytes"), {
            "lat_cpu_t16_b1": latency_of(run, "eager", "cpu", "fp32", 16, 1),
            "lat_gpu_eager_fp32_b1": latency_of(run, "eager", "cuda", "fp32", None, 1)}))
        m8 = jload(run / "metrics_int8_fbgemm.json")
        if m8:
            rows.append(cell_row(arch, seed, f"prune-{pct}+int8-cpu", run, m8, cost, base, m8.get("artifact_bytes"), {
                "lat_cpu_t16_b1": latency_of(run, "torchscript", "cpu", "int8", 16, 1)}))
        for prec in ("int8",):
            mt = jload(run / f"metrics_trt_{prec}.json")
            if mt:
                rows.append(cell_row(arch, seed, f"prune-{pct}+trt-{prec}", run, mt, cost, base, mt.get("artifact_bytes"), {
                    "lat_gpu_trt_b1": latency_of(run, "trt", "cuda", prec, None, 1)}))
    return pd.DataFrame(rows)


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    num = [c for c in df.columns if c not in ("arch", "seed", "cell", "run")]
    g = df.groupby(["arch", "cell"])
    s = g[num].mean()
    s.columns = [f"{c}_mean" for c in s.columns]
    std = g[["error_rate", "error_ratio"]].std(ddof=1)
    std.columns = ["error_rate_std", "error_ratio_std"]
    s = s.join(std)
    s.insert(0, "n_seeds", g.size())
    return s.reset_index()


def decide(s: pd.DataFrame, thr: float) -> str:
    lines = [f"# Decisão do eixo 3\n",
             f"Critério (fixado em 2026-09-14): entre as células com razão de erro média ≤ {thr}×, vence a de menor "
             "latência p50 em GPU (TensorRT, lote 1); empate → menor artefato em disco.\n"]
    ok = s[s.error_ratio_mean <= thr].copy()
    if ok.lat_gpu_trt_b1_mean.notna().sum() == 0:
        lines.append("**Latência GPU (TensorRT) ainda não medida — decisão indisponível.** Células elegíveis "
                     f"pela razão de erro ({len(ok)}):\n")
        lines.append(ok[["arch", "cell", "n_seeds", "error_ratio_mean", "artifact_bytes_mean"]]
                     .sort_values(["arch", "error_ratio_mean"]).to_markdown(index=False))
        return "\n".join(lines)
    ok = ok.dropna(subset=["lat_gpu_trt_b1_mean"]).sort_values(["lat_gpu_trt_b1_mean", "artifact_bytes_mean"])
    win = ok.iloc[0]
    lines.append(f"## Vencedora: **{LABEL.get(win.arch, win.arch)} — {win.cell}**\n")
    lines.append(f"razão de erro {win.error_ratio_mean:.2f}× · latência TRT b1 {win.lat_gpu_trt_b1_mean:.3f} ms · "
                 f"artefato {win.artifact_bytes_mean / 2**20:.1f} MiB\n")
    lines.append("## Finalistas\n")
    lines.append(ok.head(4)[["arch", "cell", "error_ratio_mean", "lat_gpu_trt_b1_mean", "artifact_bytes_mean"]]
                 .to_markdown(index=False))
    lines.append("\n## Excluídas pela razão de erro\n")
    lines.append(s[s.error_ratio_mean > thr][["arch", "cell", "error_ratio_mean"]].to_markdown(index=False))
    return "\n".join(lines)


def pareto_figs(s: pd.DataFrame) -> None:
    (OUT / "figures").mkdir(exist_ok=True)
    panels = [("lat_cpu_t16_b1_mean", "latência CPU p50, 16 threads, lote 1 (ms)", "pareto_cpu"),
              ("lat_gpu_trt_b1_mean", "latência GPU TensorRT p50, lote 1 (ms)", "pareto_gpu"),
              ("artifact_bytes_mean", "artefato em disco (bytes)", "pareto_disk"),
              ("macs_mean", "MACs @224", "pareto_macs")]
    for col, xlabel, stem in panels:
        d = s.dropna(subset=[col, "error_ratio_mean"])
        if d.empty:
            continue
        fig, ax = plt.subplots(figsize=(7.5, 4.8))
        for arch, grp in d.groupby("arch"):
            ax.errorbar(grp[col], grp.error_ratio_mean, yerr=grp.error_ratio_std.fillna(0), fmt="o", capsize=3,
                        label=LABEL.get(arch, arch))
            for _, r in grp.iterrows():
                ax.annotate(r.cell, (r[col], r.error_ratio_mean), fontsize=7, xytext=(3, 3), textcoords="offset points")
        ax.axhline(1.5, color="red", ls="--", lw=.8, label="limiar 1,5×")
        ax.set(xlabel=xlabel, ylabel="razão de erro vs baseline", xscale="log")
        ax.grid(alpha=.3, which="both")
        ax.legend()
        fig.tight_layout()
        fig.savefig(OUT / f"figures/{stem}.pdf")
        fig.savefig(OUT / f"figures/{stem}.png", dpi=200)
        plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=Path, default=ROOT / "runs")
    ap.add_argument("--knee", type=float, default=1.5)
    ap.add_argument("--decide", action="store_true")
    args = ap.parse_args()
    df = collect(args.runs)
    if df.empty:
        print("nenhuma célula")
        return 1
    df.to_csv(OUT / "eixo3_table.csv", index=False)
    s = summarize(df)
    s.to_csv(OUT / "eixo3_summary.csv", index=False)
    pareto_figs(s)
    pd.set_option("display.width", 240)
    cols = ["arch", "cell", "n_seeds", "error_ratio_mean", "error_ratio_std", "params_nonzero_mean",
            "artifact_bytes_mean", "lat_cpu_t16_b1_mean", "lat_gpu_trt_b1_mean"]
    print(s[cols].sort_values(["arch", "cell"]).to_string(index=False))
    if args.decide:
        (OUT / "decision.md").write_text(decide(s, args.knee) + "\n")
        print(f"\n-> {OUT / 'decision.md'}")
    print(f"-> {OUT / 'eixo3_table.csv'}\n-> {OUT / 'eixo3_summary.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
