"""Explicit int8 quantization of a run's ONNX (Q/DQ nodes) for TensorRT.

    python src/compress/quantize_onnx_qdq.py --run runs/eixo1_resnet50_s0

ONNX Runtime quantize_static, QDQ format, symmetric int8 for activations and
weights (what TensorRT consumes), per-channel weights.

Calibration: **MinMax** over the same 1.024 `fit` images (seed 0) used by the
CPU PTQ, flushed every `--calib-flush` batches (CalibMaxIntermediateOutputs).
The entropy / percentile calibrators of ONNX Runtime have no incremental mode:
they keep every intermediate activation of the whole calibration set in RAM
(30-53 GB for ResNet-50 -> OOM-killed twice on 2026-09-14). They are not
offered here. Run under a memory cap anyway:

    systemd-run --user --scope -p MemoryMax=16G python src/compress/quantize_onnx_qdq.py ...

Writes <run>/model_qdq_int8.onnx and <run>/qdq_int8.json, and checks the QDQ
model runs in ONNX Runtime CPU on a few val images (smoke, not the metric).
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import onnx
import onnxruntime as ort
import torch
from onnxruntime.quantization import (CalibrationDataReader, CalibrationMethod, QuantFormat,
                                      QuantType, calibrate, quantize_static)
from onnxruntime.quantization.calibrate import MinMaxCalibrater
from onnxruntime.quantization.shape_inference import quant_pre_process
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from datamodule import ManifestDataset, build_transforms, class_names, frame_for_split, read_manifest  # noqa: E402
from metrics import write_json                                                                     # noqa: E402
from runinfo import env_info                                                                       # noqa: E402


class IncrementalMinMax(MinMaxCalibrater):
    """ORT bug workaround: with max_intermediate_outputs set, MinMaxCalibrater
    .clear_collected_data() discards the collected batches WITHOUT folding their
    ranges into calibrate_tensors_range, so every flush loses data and the run
    ends with "No data is collected". Fold first, then clear."""

    def clear_collected_data(self):
        if self.intermediate_outputs:
            self.compute_data()                   # merges into self.calibrate_tensors_range
        self.intermediate_outputs = []


calibrate.MinMaxCalibrater = IncrementalMinMax    # quantize_static instantiates via this name


class Reader(CalibrationDataReader):
    def __init__(self, loader: DataLoader, input_name: str):
        self.it = iter(loader)
        self.name = input_name

    def get_next(self):
        try:
            images, _ = next(self.it)
        except StopIteration:
            return None
        return {self.name: images.numpy()}


def loader_for(cfg: dict, split: str, n: int, seed: int, batch: int, workers: int):
    df = read_manifest(cfg["data"]["manifest"])
    classes = class_names(df)
    frame = frame_for_split(df, split).sample(n=n, random_state=seed)
    _, eval_tf = build_transforms(cfg)
    return DataLoader(ManifestDataset(frame, classes, eval_tf, cfg["data"].get("root")),
                      batch_size=batch, shuffle=False, num_workers=workers), classes


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--calib-n", type=int, default=1024)
    ap.add_argument("--calib-seed", type=int, default=0)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--calib-flush", type=int, default=2,
                    help="batches kept in RAM before the MinMax ranges are merged and memory freed")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--smoke-n", type=int, default=256)
    args = ap.parse_args()

    src = args.run / "model.onnx"
    pre = args.run / "model_preprocessed.onnx"
    out = args.run / "model_qdq_int8.onnx"
    ck = torch.load(args.run / "checkpoints/best.pt", map_location="cpu", weights_only=False)
    cfg = ck["config"]
    torch.set_num_threads(args.threads)

    t0 = time.perf_counter()
    quant_pre_process(str(src), str(pre), skip_symbolic_shape=False)
    input_name = onnx.load(str(pre)).graph.input[0].name
    calib_loader, classes = loader_for(cfg, "fit", args.calib_n, args.calib_seed, args.batch, args.workers)
    quantize_static(
        model_input=str(pre), model_output=str(out), calibration_data_reader=Reader(calib_loader, input_name),
        quant_format=QuantFormat.QDQ, activation_type=QuantType.QInt8, weight_type=QuantType.QInt8,
        per_channel=True, reduce_range=False, calibrate_method=CalibrationMethod.MinMax,
        extra_options={"ActivationSymmetric": True, "WeightSymmetric": True, "AddQDQPairToWeight": True,
                       "CalibMovingAverage": False, "CalibMaxIntermediateOutputs": args.calib_flush})
    dt = time.perf_counter() - t0
    pre.unlink(missing_ok=True)
    onnx.checker.check_model(str(out))
    m = onnx.load(str(out))
    n_q = sum(1 for n in m.graph.node if n.op_type == "QuantizeLinear")

    # smoke: the QDQ graph runs and is not broken (accuracy on a few val images)
    smoke_loader, _ = loader_for(cfg, "val", args.smoke_n, 0, 32, args.workers)
    sess = ort.InferenceSession(str(out), providers=["CPUExecutionProvider"])
    ref = ort.InferenceSession(str(src), providers=["CPUExecutionProvider"])
    correct = correct_ref = seen = 0
    for images, labels in smoke_loader:
        x = images.numpy()
        correct += int((sess.run(None, {input_name: x})[0].argmax(1) == labels.numpy()).sum())
        correct_ref += int((ref.run(None, {input_name: x})[0].argmax(1) == labels.numpy()).sum())
        seen += len(labels)

    info = {"onnx_qdq": str(out), "onnx_bytes": out.stat().st_size, "source_onnx": str(src),
            "method": "onnxruntime quantize_static, QDQ, MinMax calibration, symmetric int8, per-channel weights",
            "calibration": {"split": "fit", "n": args.calib_n, "seed": args.calib_seed, "batch": args.batch,
                            "max_intermediate_outputs": args.calib_flush},
            "quantize_linear_nodes": n_q, "time_s": round(dt, 1),
            "smoke": {"split": "val", "n": seen, "acc_qdq_ort_cpu": correct / seen, "acc_fp32_ort_cpu": correct_ref / seen},
            "onnxruntime": ort.__version__, "onnx": onnx.__version__, **env_info(),
            "built_at": time.strftime("%Y-%m-%dT%H:%M:%S")}
    write_json(args.run / "qdq_int8.json", info)
    print(f"{cfg['model']['arch']}: {out} ({out.stat().st_size / 2**20:.1f} MiB, {n_q} QuantizeLinear, {dt:.0f}s) | "
          f"smoke val n={seen}: acc qdq {correct / seen:.4f} vs fp32 {correct_ref / seen:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
