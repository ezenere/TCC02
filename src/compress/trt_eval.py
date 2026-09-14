"""Evaluate a TensorRT engine on a split; writes metrics_trt_<precision>.json.

    python src/compress/trt_eval.py --run runs/eixo1_resnet50_s0 --precision int8
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from compress.trt_runtime import TRTRunner        # noqa: E402
from datamodule import build_eval_loader          # noqa: E402
from metrics import compute_metrics, write_json   # noqa: E402
from runinfo import env_info, git_info            # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--precision", choices=["fp32", "fp16", "int8"], required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--engine-tag", default="")
    args = ap.parse_args()

    engine = args.run / "trt" / f"model_{args.precision}{args.engine_tag}.engine"
    ck = torch.load(args.run / "checkpoints/best.pt", map_location="cpu", weights_only=False)
    cfg = json.loads(json.dumps(ck["config"]))
    cfg["data"]["eval_batch_size"] = args.batch
    cfg["data"]["workers"] = args.workers
    loader, meta = build_eval_loader(cfg, args.split)
    assert meta["classes"] == ck["classes"]
    if args.split == "test":
        assert ck["manifest_sha256"] == meta["manifest_sha256"], "manifest mismatch"

    runner = TRTRunner(engine)
    preds, targets = [], []
    t0 = time.perf_counter()
    for images, labels in tqdm(loader, desc=f"trt {args.precision} {args.split}", leave=False, mininterval=30):
        out = runner(images.to("cuda", non_blocking=True))
        preds.append(out.argmax(1).cpu())
        targets.append(labels)
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    m = compute_metrics(torch.cat(targets).numpy(), torch.cat(preds).numpy(), meta["classes"])
    build = json.loads((args.run / "trt" / f"build_{args.precision}{args.engine_tag}.json").read_text())
    payload = {"split": args.split, "artifact": str(engine), "artifact_bytes": engine.stat().st_size,
               "backend": "tensorrt", "precision": args.precision, "tensorrt": build.get("tensorrt"),
               "calibration": build.get("calibration"), "checkpoint": str(args.run / "checkpoints/best.pt"),
               "arch": cfg["model"]["arch"], "seed": ck.get("seed"),
               "prune_sparsity": (cfg.get("prune") or {}).get("sparsity"),
               "manifest_version": meta["manifest_version"], "manifest_sha256": meta["manifest_sha256"],
               "eval_time_s": round(dt, 1), "eval_img_s": round(m["n"] / dt, 1),
               "evaluated_at": time.strftime("%Y-%m-%dT%H:%M:%S"), **git_info(), **env_info(), **m}
    out = args.out or (args.run / f"metrics_trt_{args.precision}.json")
    write_json(out, payload)
    print(f"{cfg['model']['arch']} TRT {args.precision} {args.split}: n={m['n']:,} acc {m['acc']:.5f} "
          f"f1 {m['f1_macro']:.5f} erro {m['error_rate']:.4%} ({m['n_errors']}) | {payload['eval_img_s']} img/s\n-> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
