"""Axis 2 (quantization): consolidate int8 cells against the axis-1 baselines.

    python results/eixo2/make_quant.py

Reads runs/eixo1_<arch>_s<seed>/metrics_int8_fbgemm.json (CPU PTQ) and, when
present, metrics_trt_{fp32,fp16,int8}.json (GPU TensorRT). Error ratio is
against the PyTorch baseline of the same seed (metrics.json). Accuracy is
reported per backend — parity between backends is never assumed.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
PAT = re.compile(r"eixo1_(?P<arch>\w+?)_s(?P<seed>\d+)$")
CELLS = {"metrics_int8_fbgemm.json": ("cpu-fbgemm", "int8"),
         "metrics_trt_fp32.json": ("gpu-tensorrt", "fp32"),
         "metrics_trt_fp16.json": ("gpu-tensorrt", "fp16"),
         "metrics_trt_int8.json": ("gpu-tensorrt", "int8")}


def collect(runs_dir: Path) -> pd.DataFrame:
    rows = []
    for run in sorted(runs_dir.glob("eixo1_*")):
        g = PAT.match(run.name)
        if not g or not (run / "metrics.json").exists():
            continue
        base = json.loads((run / "metrics.json").read_text())
        cost = json.loads((run / "cost.json").read_text()) if (run / "cost.json").exists() else {}
        for fname, (backend, precision) in CELLS.items():
            f = run / fname
            if not f.exists():
                continue
            m = json.loads(f.read_text())
            rows.append({
                "run": run.name, "arch": g["arch"], "seed": int(g["seed"]), "backend": backend,
                "precision": precision, "acc": m["acc"], "f1_macro": m["f1_macro"],
                "error_rate": m["error_rate"], "n_errors": m["n_errors"],
                "baseline_error_rate": base["error_rate"], "baseline_n_errors": base["n_errors"],
                "error_ratio": m["error_rate"] / base["error_rate"],
                "artifact_bytes": m.get("artifact_bytes"), "fp32_bytes": cost.get("state_dict_fp32_bytes"),
                "size_ratio": (m.get("size_ratio_vs_fp32")
                               or (m.get("artifact_bytes") / cost["state_dict_fp32_bytes"]
                                   if m.get("artifact_bytes") and cost.get("state_dict_fp32_bytes") else np.nan)),
                "eval_img_s": m.get("eval_img_s"),
                "threads": m.get("threads"), "calib_n": (m.get("calibration") or {}).get("n"),
            })
    return pd.DataFrame(rows)


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby(["arch", "backend", "precision"])
    return pd.DataFrame({
        "n_seeds": g.size(), "acc_mean": g.acc.mean(), "f1_mean": g.f1_macro.mean(),
        "err_mean": g.error_rate.mean(), "err_std": g.error_rate.std(ddof=1),
        "err_ratio_mean": g.error_ratio.mean(), "err_ratio_std": g.error_ratio.std(ddof=1),
        "artifact_mib": g.artifact_bytes.mean() / 2**20, "size_ratio": g.size_ratio.mean(),
        "eval_img_s": g.eval_img_s.mean(),
    }).reset_index()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=Path, default=ROOT / "runs")
    args = ap.parse_args()
    df = collect(args.runs)
    if df.empty:
        print("nenhuma célula de quantização encontrada")
        return 1
    df.to_csv(OUT / "quant_runs.csv", index=False)
    s = summarize(df)
    s.to_csv(OUT / "quant_summary.csv", index=False)
    pd.set_option("display.width", 220)
    print("=== células ===")
    print(df[["run", "backend", "precision", "acc", "f1_macro", "error_rate", "n_errors",
              "baseline_n_errors", "error_ratio", "size_ratio", "eval_img_s"]].to_string(index=False))
    print("\n=== summary (por arquitetura × backend × precisão; acurácia por backend, sem assumir paridade) ===")
    print(s.to_string(index=False))
    lines = ["| arquitetura | backend | precisão | seeds | razão de erro | erros / baseline (por seed) | artefato (× FP32) | img/s (aval.) |",
             "|---|---|---|---|---|---|---|---|"]
    LABEL = {"resnet50": "ResNet-50", "densenet121": "DenseNet-121"}
    for r in s.sort_values(["arch", "backend", "precision"]).itertuples():
        cells = df[(df.arch == r.arch) & (df.backend == r.backend) & (df.precision == r.precision)].sort_values("seed")
        errs = ", ".join(f"{int(c.n_errors)}/{int(c.baseline_n_errors)}" for c in cells.itertuples())
        std = f" ± {r.err_ratio_std:.2f}" if pd.notna(r.err_ratio_std) else ""
        lines.append(f"| {LABEL.get(r.arch, r.arch)} | {r.backend} | {r.precision} | {int(r.n_seeds)} | {r.err_ratio_mean:.2f}{std} | {errs} | "
                     f"{r.artifact_mib:.1f} MiB ({r.size_ratio:.2f})" + f" | {r.eval_img_s:.0f} |")
    render_block(OUT / "README.md", "<!-- eixo2:quant:start -->", "<!-- eixo2:quant:end -->", "\n".join(lines))
    print(f"\n-> {OUT / 'quant_runs.csv'}\n-> {OUT / 'quant_summary.csv'}")
    return 0


def render_block(readme, start: str, end: str, block: str) -> None:
    """Rewrite the text between two marker comments in README.md."""
    from pathlib import Path as _P
    readme = _P(readme)
    if not readme.exists():
        return
    txt = readme.read_text()
    if start in txt and end in txt:
        pre, rest = txt.split(start, 1)
        _, post = rest.split(end, 1)
        readme.write_text(f"{pre}{start}\n{block}\n{end}{post}")


if __name__ == "__main__":
    raise SystemExit(main())

