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
                "size_ratio": m.get("size_ratio_vs_fp32"), "eval_img_s": m.get("eval_img_s"),
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
    print(f"\n-> {OUT / 'quant_runs.csv'}\n-> {OUT / 'quant_summary.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
