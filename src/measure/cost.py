"""Static cost of a checkpoint: parameters (total / non-zero), MACs, disk size.

    python src/measure/cost.py --checkpoint runs/eixo1_resnet50_s0/checkpoints/best.pt

Writes cost.json next to the run. Non-zero parameters and the gzip size are the
quantities that move under pruning; the FP32 state_dict size does not.
"""

from __future__ import annotations

import argparse
import gzip
import io
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from metrics import write_json          # noqa: E402
from train import build_model           # noqa: E402


def count_params(model: torch.nn.Module) -> dict:
    total = nonzero = 0
    per_layer = []
    for name, p in model.named_parameters():
        n = p.numel()
        nz = int((p != 0).sum().item())
        total += n
        nonzero += nz
        per_layer.append({"name": name, "numel": n, "nonzero": nz, "sparsity": 1 - nz / n})
    return {"params_total": total, "params_nonzero": nonzero,
            "sparsity_global": 1 - nonzero / total, "per_layer": per_layer}


def count_macs(model: torch.nn.Module, size: int) -> dict:
    from fvcore.nn import FlopCountAnalysis
    x = torch.zeros(1, 3, size, size)
    fca = FlopCountAnalysis(model.eval(), x)
    fca.unsupported_ops_warnings(False)
    fca.uncalled_modules_warnings(False)
    # fvcore counts one fused multiply-add as one "flop" -> its total is MACs.
    return {"macs": int(fca.total()), "macs_input": [1, 3, size, size],
            "macs_by_module_top": {k: int(v) for k, v in
                                   sorted(fca.by_module().items(), key=lambda kv: -kv[1])[:10]}}


def disk_sizes(state_dict: dict) -> dict:
    buf = io.BytesIO()
    torch.save(state_dict, buf)
    raw = buf.getvalue()
    gz = gzip.compress(raw, compresslevel=6)
    return {"state_dict_fp32_bytes": len(raw), "state_dict_fp32_gzip_bytes": len(gz)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--image-size", type=int, default=224)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    ck = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    cfg = ck["config"]
    classes = ck.get("classes") or []
    model = build_model(cfg["model"]["arch"], len(classes), pretrained=False)
    model.load_state_dict(ck["model"])

    t0 = time.perf_counter()
    payload = {"checkpoint": str(args.checkpoint), "arch": cfg["model"]["arch"],
               "num_classes": len(classes), "epoch": ck.get("epoch", -1) + 1,
               **count_params(model), **count_macs(model, args.image_size),
               **disk_sizes(ck["model"]), "measured_at": time.strftime("%Y-%m-%dT%H:%M:%S")}
    out = args.out or (args.checkpoint.parent.parent / "cost.json")
    write_json(out, payload)
    print(f"{payload['arch']}: params {payload['params_total'] / 1e6:.2f} M "
          f"(nao-nulos {payload['params_nonzero'] / 1e6:.2f} M, esparsidade {payload['sparsity_global']:.1%}) | "
          f"MACs {payload['macs'] / 1e9:.2f} G @ {args.image_size}px | "
          f"FP32 {payload['state_dict_fp32_bytes'] / 2**20:.1f} MiB, gzip {payload['state_dict_fp32_gzip_bytes'] / 2**20:.1f} MiB "
          f"[{time.perf_counter() - t0:.1f}s]\n-> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
