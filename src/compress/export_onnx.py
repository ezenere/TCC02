"""Export a checkpoint to ONNX and verify parity against PyTorch.

    python src/compress/export_onnx.py --checkpoint runs/eixo1_resnet50_s0/checkpoints/best.pt

Writes <run>/model.onnx (opset 17, dynamic batch, FP32, eval mode, BN folded by
the exporter) plus <run>/onnx_parity.json with max |Δlogit| over N manifest
images run through ONNX Runtime CPU vs PyTorch CPU FP32. Pruned checkpoints
export unchanged: zeros are ordinary weights.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from datamodule import build_eval_loader   # noqa: E402
from metrics import write_json             # noqa: E402
from runinfo import env_info               # noqa: E402
from train import build_model              # noqa: E402


def export(model: torch.nn.Module, out: Path, size: int, opset: int, dynamo: bool) -> None:
    dummy = torch.zeros(1, 3, size, size)
    torch.onnx.export(
        model, dummy, str(out), opset_version=opset, dynamo=dynamo,
        input_names=["input"], output_names=["logits"],
        dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
        do_constant_folding=True)


def parity(model: torch.nn.Module, onnx_path: Path, cfg: dict, n: int, split: str) -> dict:
    cfg = json.loads(json.dumps(cfg))
    cfg["data"]["workers"] = 2
    cfg["data"]["eval_batch_size"] = 16
    loader, meta = build_eval_loader(cfg, split)
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    max_abs, argmax_mismatch, seen = 0.0, 0, 0
    t_ort = t_pt = 0.0
    with torch.no_grad():
        for images, _ in loader:
            t0 = time.perf_counter()
            ref = model(images).numpy()
            t_pt += time.perf_counter() - t0
            t0 = time.perf_counter()
            out = sess.run(["logits"], {"input": images.numpy()})[0]
            t_ort += time.perf_counter() - t0
            max_abs = max(max_abs, float(np.abs(out - ref).max()))
            argmax_mismatch += int((out.argmax(1) != ref.argmax(1)).sum())
            seen += images.shape[0]
            if seen >= n:
                break
    return {"n_images": seen, "split": split, "max_abs_diff_logits": max_abs,
            "argmax_mismatches": argmax_mismatch,
            "pytorch_cpu_img_s": round(seen / t_pt, 1), "ort_cpu_img_s": round(seen / t_ort, 1)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--opset", type=int, default=17)
    ap.add_argument("--dynamo", action="store_true", help="use the torch.export-based exporter")
    ap.add_argument("--parity-n", type=int, default=64)
    ap.add_argument("--parity-split", default="val")
    ap.add_argument("--tol", type=float, default=1e-4)
    args = ap.parse_args()

    ck = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    cfg = ck["config"]
    model = build_model(cfg["model"]["arch"], len(ck["classes"]), pretrained=False)
    model.load_state_dict(ck["model"])
    model.eval()
    torch.set_num_threads(8)

    run_dir = args.checkpoint.parent.parent
    out = args.out or (run_dir / "model.onnx")
    t0 = time.perf_counter()
    export(model, out, int(cfg["data"]["image_size"]), args.opset, args.dynamo)
    t_export = time.perf_counter() - t0
    onnx.checker.check_model(str(out))
    m = onnx.load(str(out))

    par = parity(model, out, cfg, args.parity_n, args.parity_split)
    ok = par["max_abs_diff_logits"] < args.tol and par["argmax_mismatches"] == 0
    payload = {"checkpoint": str(args.checkpoint), "onnx": str(out), "arch": cfg["model"]["arch"],
               "opset": args.opset, "exporter": "dynamo" if args.dynamo else "torchscript",
               "ir_version": m.ir_version, "n_nodes": len(m.graph.node),
               "onnx_bytes": out.stat().st_size, "export_time_s": round(t_export, 1),
               "tolerance": args.tol, "parity_ok": ok, **par, **env_info(),
               "onnx_version": onnx.__version__, "onnxruntime_version": ort.__version__}
    write_json(run_dir / "onnx_parity.json", payload)
    print(f"{cfg['model']['arch']}: {out} ({out.stat().st_size / 2**20:.1f} MiB, {len(m.graph.node)} nós, "
          f"opset {args.opset}) | paridade ORT-CPU vs PyTorch em {par['n_images']} imgs: "
          f"max|Δ| = {par['max_abs_diff_logits']:.2e}, argmax divergentes = {par['argmax_mismatches']} "
          f"-> {'OK' if ok else 'FALHOU'} | ORT {par['ort_cpu_img_s']} img/s vs PyTorch {par['pytorch_cpu_img_s']} img/s")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
