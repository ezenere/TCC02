"""Inference latency of one artifact on CPU or GPU.

    python src/measure/latency.py --artifact runs/eixo1_resnet50_s0/checkpoints/best.pt \\
        --kind eager --device cpu --threads 1 --batch 1 32
    python src/measure/latency.py --artifact runs/eixo1_resnet50_s0/model_int8_fbgemm.pt \\
        --kind torchscript --device cpu --threads 16
    python src/measure/latency.py --artifact runs/eixo1_resnet50_s0/model.onnx --kind onnx --device cpu
    python src/measure/latency.py --artifact runs/eixo1_resnet50_s0/checkpoints/best.pt \\
        --kind eager --device cuda --precision fp16 --batch 1 32

Protocol: synthetic input (latency does not depend on pixel content), 50 warmup
iterations, 300 timed iterations per batch size; p50 / p95 / mean of the
per-batch wall time and images/s. On GPU every iteration is bracketed by
cuda.synchronize(). Writes <run>/latency_<kind>_<device>_<precision>_t<threads>.json
with hardware, versions and the manual `--note` (e.g. "sessão gráfica fechada").
"""

from __future__ import annotations

import argparse
import json
import platform
import re
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from metrics import write_json      # noqa: E402
from runinfo import env_info        # noqa: E402


def cpu_name() -> str:
    try:
        txt = Path("/proc/cpuinfo").read_text()
        m = re.search(r"model name\s*:\s*(.+)", txt)
        return m.group(1).strip() if m else platform.processor()
    except OSError:
        return platform.processor()


def measure(fn, x, warmup: int, iters: int, sync) -> dict:
    """Time fn(x) `iters` times after `warmup`; returns ms statistics."""
    for _ in range(warmup):
        fn(x)
    sync()
    times = np.empty(iters)
    for i in range(iters):
        t0 = time.perf_counter()
        fn(x)
        sync()
        times[i] = time.perf_counter() - t0
    ms = 1000 * times
    return {"p50_ms": float(np.percentile(ms, 50)), "p95_ms": float(np.percentile(ms, 95)),
            "mean_ms": float(ms.mean()), "std_ms": float(ms.std()), "min_ms": float(ms.min()),
            "iters": iters, "warmup": warmup}


def load_artifact(path: Path, kind: str, device: str, precision: str):
    """Returns (callable taking a torch tensor or numpy array, input_is_numpy, arch)."""
    if kind == "eager":
        from train import build_model
        ck = torch.load(path, map_location="cpu", weights_only=False)
        cfg = ck["config"]
        model = build_model(cfg["model"]["arch"], len(ck["classes"]), pretrained=False)
        model.load_state_dict(ck["model"])
        model.eval().to(device)
        if device == "cuda":
            model = model.to(memory_format=torch.channels_last)
        if precision == "fp16" and device == "cuda":
            def fn(x):
                with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
                    return model(x)
        else:
            def fn(x):
                with torch.no_grad():
                    return model(x)
        return fn, False, cfg["model"]["arch"]

    if kind == "torchscript":
        model = torch.jit.load(str(path), map_location=device)
        model.eval()

        def fn(x):
            with torch.no_grad():
                return model(x)
        return fn, False, None

    if kind == "onnx":
        import onnxruntime as ort
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if device == "cuda" else ["CPUExecutionProvider"]
        so = ort.SessionOptions()
        sess = ort.InferenceSession(str(path), so, providers=providers)
        name = sess.get_inputs()[0].name
        return (lambda x: sess.run(None, {name: x})), True, None

    if kind == "trt":
        from compress.trt_runtime import TRTRunner       # provided with the TensorRT step
        runner = TRTRunner(path)
        return runner, False, None

    raise ValueError(kind)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifact", type=Path, required=True)
    ap.add_argument("--kind", choices=["eager", "torchscript", "onnx", "trt"], required=True)
    ap.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    ap.add_argument("--precision", choices=["fp32", "fp16", "int8"], default="fp32",
                    help="label (and autocast for eager+cuda); int8 artifacts are already int8")
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--batch", type=int, nargs="+", default=[1, 32])
    ap.add_argument("--size", type=int, default=224)
    ap.add_argument("--warmup", type=int, default=50)
    ap.add_argument("--iters", type=int, default=300)
    ap.add_argument("--note", default="", help="manual context, e.g. 'sessão gráfica fechada (TTY)'")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    torch.set_num_threads(args.threads)
    if args.device == "cuda":
        torch.backends.cudnn.benchmark = True
    if args.kind == "onnx":
        import onnxruntime as ort
        ort.set_default_logger_severity(3)

    fn, numpy_in, arch = load_artifact(args.artifact, args.kind, args.device, args.precision)
    sync = torch.cuda.synchronize if args.device == "cuda" else (lambda: None)

    results = {}
    for b in args.batch:
        x = torch.randn(b, 3, args.size, args.size)
        if numpy_in:
            x = x.numpy()
        else:
            x = x.to(args.device)
            if args.device == "cuda":
                x = x.to(memory_format=torch.channels_last)
        r = measure(fn, x, args.warmup, args.iters, sync)
        r["img_s_p50"] = 1000 * b / r["p50_ms"]
        r["img_s_mean"] = 1000 * b / r["mean_ms"]
        results[f"batch_{b}"] = r
        print(f"batch {b:>3}: p50 {r['p50_ms']:8.3f} ms  p95 {r['p95_ms']:8.3f} ms  "
              f"mean {r['mean_ms']:8.3f} ms  -> {r['img_s_p50']:8.1f} img/s (p50)")

    payload = {"artifact": str(args.artifact), "artifact_bytes": args.artifact.stat().st_size,
               "kind": args.kind, "device": args.device, "precision": args.precision,
               "threads": args.threads if args.device == "cpu" else None, "arch": arch,
               "input_size": args.size, "cpu": cpu_name(), "note": args.note,
               "measured_at": time.strftime("%Y-%m-%dT%H:%M:%S"), **env_info(),
               "onnxruntime": __import__("onnxruntime").__version__ if args.kind == "onnx" else None,
               "results": results}
    run_dir = args.artifact.parent.parent if args.artifact.parent.name == "checkpoints" else args.artifact.parent
    out = args.out or (run_dir / f"latency_{args.kind}_{args.device}_{args.precision}_t{args.threads}.json")
    write_json(out, payload)
    print(f"-> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
