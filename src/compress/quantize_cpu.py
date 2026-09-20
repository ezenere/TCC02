"""Post-training static int8 quantization for CPU (PyTorch FX graph mode, x86/fbgemm).

    python src/compress/quantize_cpu.py --checkpoint runs/eixo1_resnet50_s0/checkpoints/best.pt

Steps: eval-mode FP32 model -> prepare_fx (conv+bn+relu fusion, observers) ->
calibration on N images of `fit` (never test) -> convert_fx -> TorchScript
artifact <run>/model_int8_fbgemm.pt -> accuracy on the test split in CPU.
Writes <run>/metrics_int8_fbgemm.json in the shared metrics format.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.ao.quantization import get_default_qconfig_mapping
from torch.ao.quantization.quantize_fx import convert_fx, prepare_fx
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from datamodule import (ManifestDataset, as_bool, build_transforms, class_names, frame_for_split,  # noqa: E402
                        read_manifest, run_mask_column)
from metrics import compute_metrics, write_json                                                     # noqa: E402
from runinfo import env_info, git_info                                                              # noqa: E402
from train import build_model                                                                       # noqa: E402


def loader_for(df: pd.DataFrame, cfg: dict, classes, batch: int, workers: int) -> DataLoader:
    _, eval_tf = build_transforms(cfg)
    return DataLoader(ManifestDataset(df, classes, eval_tf, cfg["data"].get("root")),
                      batch_size=batch, shuffle=False, num_workers=workers, pin_memory=False)


@torch.no_grad()
def evaluate_cpu(model, loader, classes, desc: str) -> dict:
    preds, targets = [], []
    t0 = time.perf_counter()
    for images, labels in tqdm(loader, desc=desc, leave=False, mininterval=30):
        preds.append(model(images).argmax(1))
        targets.append(labels)
    dt = time.perf_counter() - t0
    m = compute_metrics(torch.cat(targets).numpy(), torch.cat(preds).numpy(), classes)
    m["eval_time_s"] = round(dt, 1)
    m["eval_img_s"] = round(m["n"] / dt, 1)
    return m


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--backend", default="x86", choices=["x86", "fbgemm"])
    ap.add_argument("--calib-n", type=int, default=1024)
    ap.add_argument("--calib-seed", type=int, default=0)
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--limit-test", type=int, default=None, help="smoke only")
    ap.add_argument("--eval-fp32", action="store_true", help="also evaluate the FP32 model on CPU")
    ap.add_argument("--tag", default="int8_fbgemm")
    args = ap.parse_args()

    torch.set_num_threads(args.threads)
    torch.backends.quantized.engine = "x86" if args.backend == "x86" else "fbgemm"

    ck = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    cfg = ck["config"]
    run_dir = args.checkpoint.parent.parent
    df = read_manifest(cfg["data"]["manifest"])
    classes = class_names(df)
    assert classes == ck["classes"]

    model = build_model(cfg["model"]["arch"], len(classes), pretrained=False)
    model.load_state_dict(ck["model"])
    model.eval()

    # --- calibration set: N images of fit, deterministic ----------------------
    fit_df = frame_for_split(df, "fit")
    mask = run_mask_column(run_dir)
    if mask:                                   # axis 4: calibrate only on the run's data fraction
        fit_df = fit_df[as_bool(fit_df[mask])]
    calib_df = fit_df.sample(n=min(args.calib_n, len(fit_df)), random_state=args.calib_seed)
    calib_loader = loader_for(calib_df, cfg, classes, args.batch, args.workers)
    test_df = frame_for_split(df, "test")
    if args.limit_test:
        test_df = test_df.sample(n=args.limit_test, random_state=0)
    test_loader = loader_for(test_df, cfg, classes, args.batch, args.workers)

    # --- prepare / calibrate / convert ---------------------------------------
    example = torch.zeros(1, 3, int(cfg["data"]["image_size"]), int(cfg["data"]["image_size"]))
    qconfig_mapping = get_default_qconfig_mapping(args.backend)
    t0 = time.perf_counter()
    prepared = prepare_fx(copy.deepcopy(model), qconfig_mapping, example_inputs=(example,))
    with torch.no_grad():
        for images, _ in tqdm(calib_loader, desc="calibração", leave=False, mininterval=30):
            prepared(images)
    quantized = convert_fx(prepared)
    t_quant = time.perf_counter() - t0

    artifact = run_dir / f"model_{args.tag}.pt"
    scripted = torch.jit.script(quantized)
    torch.jit.save(scripted, str(artifact))
    size = artifact.stat().st_size
    fp32_size = json.loads((run_dir / "cost.json").read_text())["state_dict_fp32_bytes"] \
        if (run_dir / "cost.json").exists() else None

    # --- accuracy on test (CPU) ------------------------------------------------
    loaded = torch.jit.load(str(artifact))
    loaded.eval()
    m_int8 = evaluate_cpu(loaded, test_loader, classes, "teste int8")
    payload = {"split": "test", "artifact": str(artifact), "artifact_bytes": size,
               "fp32_state_dict_bytes": fp32_size,
               "size_ratio_vs_fp32": (size / fp32_size) if fp32_size else None,
               "method": "PTQ static int8, FX graph mode", "backend": torch.backends.quantized.engine,
               "qconfig": args.backend, "calibration": {"split": "fit", "n": len(calib_df), "seed": args.calib_seed, "mask_column": mask},
               "quantize_time_s": round(t_quant, 1), "threads": args.threads,
               "limit_test": args.limit_test, "checkpoint": str(args.checkpoint),
               "arch": cfg["model"]["arch"], "seed": ck.get("seed"),
               "prune_sparsity": (cfg.get("prune") or {}).get("sparsity"),
               "manifest_sha256": ck.get("manifest_sha256"), **git_info(), **env_info(), **m_int8}
    if args.eval_fp32:
        m_fp32 = evaluate_cpu(model, test_loader, classes, "teste fp32 cpu")
        payload["fp32_cpu"] = {k: m_fp32[k] for k in ("acc", "f1_macro", "error_rate", "n_errors", "eval_img_s")}
    out = run_dir / f"metrics_{args.tag}{'_smoke' if args.limit_test else ''}.json"
    write_json(out, payload)
    print(f"{cfg['model']['arch']} int8/{torch.backends.quantized.engine}: artefato {size / 2**20:.1f} MiB"
          f"{f' ({size / fp32_size:.2f}x FP32)' if fp32_size else ''} | quantização {t_quant:.0f}s | "
          f"teste n={m_int8['n']:,}: acc {m_int8['acc']:.5f} f1 {m_int8['f1_macro']:.5f} "
          f"erro {m_int8['error_rate']:.4%} ({m_int8['n_errors']}) | {m_int8['eval_img_s']} img/s "
          f"({args.threads} threads)\n-> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
