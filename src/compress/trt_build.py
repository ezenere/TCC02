"""Build TensorRT engines (FP32 / FP16 / INT8) from a run's ONNX exports.

    python src/compress/trt_build.py --run runs/eixo1_resnet50_s0 --precision fp32 fp16 int8

TensorRT 11 has no precision builder flags (FP16/INT8 were removed with weakly
typed networks): the engine's precision is the precision of the graph, built as
a *strongly typed* network. So each precision comes from its own ONNX:
  fp32 -> model.onnx            (TF32 disabled, so it is genuinely FP32)
  fp16 -> model_fp16.onnx       (onnxconverter-common, keep_io_types=True)
  int8 -> model_qdq_int8.onnx   (Q/DQ nodes from quantize_onnx_qdq.py)
One optimisation profile: batch min 1 / opt 32 / max 32. After the build the
engine is inspected and the per-layer precision histogram is saved with it.
Build with the GPU otherwise idle: tactic selection is timing-based.
Writes <run>/trt/model_<precision>.engine and <run>/trt/build_<precision>.json.
"""

from __future__ import annotations

import argparse
import collections
import json
import sys
import time
from pathlib import Path

import onnx
import tensorrt as trt
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from metrics import write_json      # noqa: E402
from runinfo import env_info        # noqa: E402

LOGGER = trt.Logger(trt.Logger.WARNING)
SOURCES = {"fp32": "model.onnx", "fp16": "model_fp16.onnx", "int8": "model_qdq_int8.onnx"}


def make_fp16_onnx(src: Path, dst: Path) -> None:
    """FP16 weights and activations, FP32 graph inputs/outputs."""
    from onnxconverter_common import float16
    model = onnx.load(str(src))
    model16 = float16.convert_float_to_float16(model, keep_io_types=True)
    onnx.save(model16, str(dst))
    onnx.checker.check_model(str(dst))


def inspect_engine(engine_bytes: bytes) -> dict:
    with trt.Runtime(LOGGER) as rt:
        eng = rt.deserialize_cuda_engine(engine_bytes)
    info = json.loads(eng.create_engine_inspector().get_engine_information(trt.LayerInformationFormat.JSON))
    layers = info.get("Layers", [])
    hist = collections.Counter()
    for layer in layers:
        if not isinstance(layer, dict):          # engine built without DETAILED verbosity: names only
            hist["(sem detalhe)"] += 1
            continue
        outs = layer.get("Outputs") or []
        hist[outs[0].get("Datatype", "?") if outs else "?"] += 1
    return {"n_layers": len(layers), "output_datatype_histogram": dict(hist)}


def build(onnx_path: Path, out: Path, precision: str, size: int, args) -> dict:
    builder = trt.Builder(LOGGER)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
    parser = trt.OnnxParser(network, LOGGER)
    if not parser.parse_from_file(str(onnx_path)):
        errs = [str(parser.get_error(i)) for i in range(parser.num_errors)]
        raise RuntimeError("ONNX parse failed:\n" + "\n".join(errs))

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, args.workspace_gib << 30)
    config.profiling_verbosity = trt.ProfilingVerbosity.DETAILED   # per-layer types in the inspector
    if precision == "fp32":
        config.clear_flag(trt.BuilderFlag.TF32)          # genuine FP32 (TF32 is on by default on Ampere)
    profile = builder.create_optimization_profile()
    name = network.get_input(0).name
    profile.set_shape(name, (1, 3, size, size), (args.max_batch, 3, size, size), (args.max_batch, 3, size, size))
    config.add_optimization_profile(profile)

    t0 = time.perf_counter()
    serialized = builder.build_serialized_network(network, config)
    dt = time.perf_counter() - t0
    if serialized is None:
        raise RuntimeError(f"engine build failed ({precision})")
    out.parent.mkdir(parents=True, exist_ok=True)
    data = bytes(serialized)
    out.write_bytes(data)

    qdq_json = onnx_path.parent / "qdq_int8.json"
    return {"engine": str(out), "engine_bytes": len(data), "precision": precision,
            "strongly_typed": True, "tf32": precision != "fp32", "build_time_s": round(dt, 1),
            "onnx": str(onnx_path), "onnx_bytes": onnx_path.stat().st_size,
            "opt_profile": {"min": 1, "opt": args.max_batch, "max": args.max_batch},
            "workspace_gib": args.workspace_gib, "inspector": inspect_engine(data),
            "calibration": json.loads(qdq_json.read_text()) if precision == "int8" and qdq_json.exists() else None,
            "tensorrt": trt.__version__, **env_info(), "built_at": time.strftime("%Y-%m-%dT%H:%M:%S")}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--precision", nargs="+", default=["fp32", "fp16", "int8"], choices=list(SOURCES))
    ap.add_argument("--max-batch", type=int, default=32)
    ap.add_argument("--workspace-gib", type=int, default=4)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    ck = torch.load(args.run / "checkpoints/best.pt", map_location="cpu", weights_only=False)
    size = int(ck["config"]["data"]["image_size"])
    if not (args.run / "model.onnx").exists():
        raise SystemExit(f"{args.run / 'model.onnx'} missing — run src/compress/export_onnx.py first")

    for prec in args.precision:
        src = args.run / SOURCES[prec]
        if prec == "fp16" and not src.exists():
            make_fp16_onnx(args.run / "model.onnx", src)
            print(f"fp16 onnx -> {src} ({src.stat().st_size / 2**20:.1f} MiB)")
        if prec == "int8" and not src.exists():
            raise SystemExit(f"{src} missing — run src/compress/quantize_onnx_qdq.py first")
        out = args.run / "trt" / f"model_{prec}.engine"
        if out.exists() and not args.force:
            print(f"skip {out} (existe)")
            continue
        info = build(src, out, prec, size, args)
        write_json(args.run / "trt" / f"build_{prec}.json", info)
        print(f"{prec}: {out} ({info['engine_bytes'] / 2**20:.1f} MiB, build {info['build_time_s']}s) "
              f"| camadas por tipo: {info['inspector']['output_datatype_histogram']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
