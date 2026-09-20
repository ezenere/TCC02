"""Evaluate a checkpoint on one split, independently of training.

    python src/eval.py --checkpoint runs/eixo1_resnet50_s0/checkpoints/best.pt --split test

Writes metrics_<split>.json next to the run (or --out) in the same format as
train.py, so results scripts consume both interchangeably.
"""

from __future__ import annotations

import argparse
import hashlib
import time
from pathlib import Path

import torch
import torch.nn as nn
import yaml

from datamodule import build_eval_loader
from metrics import write_json
from runinfo import env_info, git_info
from train import build_model, evaluate


def file_sha256(path: Path, limit_bytes: int = 64 << 20) -> str:
    """sha256 of the first 64 MB — enough to identify a checkpoint cheaply."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        h.update(fh.read(limit_bytes))
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--split", choices=["fit", "val", "test", "train"], default="test")
    ap.add_argument("--manifest", type=str, default=None, help="override the checkpoint's manifest")
    ap.add_argument("--config", type=Path, default=None, help="override the checkpoint's config")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--save-preds", type=Path, default=None, help="CSV with image_path,label,pred,confidence")
    args = ap.parse_args()

    ck = torch.load(args.checkpoint, map_location="cuda", weights_only=False)
    cfg = yaml.safe_load(args.config.read_text()) if args.config else ck["config"]
    if args.batch_size:
        cfg["data"]["eval_batch_size"] = args.batch_size
    if args.workers is not None:
        cfg["data"]["workers"] = args.workers

    loader, meta = build_eval_loader(cfg, args.split, args.manifest)
    if ck.get("classes") and ck["classes"] != meta["classes"]:
        raise SystemExit("class list of the checkpoint differs from the manifest")
    if args.split == "test" and ck.get("manifest_sha256") and args.manifest is None:
        assert ck["manifest_sha256"] == meta["manifest_sha256"], \
            "checkpoint was trained on a different manifest version than the one on disk"

    device = torch.device("cuda")
    amp_dtype = {"fp16": torch.float16, "bf16": torch.bfloat16}[cfg["train"]["amp"]]
    model = build_model(cfg["model"]["arch"], len(meta["classes"]), pretrained=False).to(device)
    model.load_state_dict(ck["model"])
    model = model.to(memory_format=torch.channels_last)
    torch.backends.cudnn.benchmark = True

    t0 = time.perf_counter()
    m, pred_idx, conf = evaluate(model, loader, device, amp_dtype, nn.CrossEntropyLoss(), meta["classes"],
                                 desc=f"eval {args.split}", return_preds=True)
    if args.save_preds:
        import pandas as pd
        ds = loader.dataset
        pd.DataFrame({"image_path": ds.paths, "label": [meta["classes"][t] for t in ds.targets],
                      "pred": [meta["classes"][i] for i in pred_idx], "confidence": conf.round(4)}
                     ).to_csv(args.save_preds, index=False)
    dt = time.perf_counter() - t0

    payload = {"split": args.split, "checkpoint": str(args.checkpoint),
               "checkpoint_sha256_64mb": file_sha256(args.checkpoint),
               "epoch": ck.get("epoch", -1) + 1, "run_name": cfg.get("run_name"),
               "arch": cfg["model"]["arch"], "seed": ck.get("seed", cfg.get("seed")),
               "manifest_version": meta["manifest_version"], "manifest_sha256": meta["manifest_sha256"],
               "eval_time_s": round(dt, 1), "eval_img_s": round(meta["n"] / dt, 1),
               "evaluated_at": time.strftime("%Y-%m-%dT%H:%M:%S"), **git_info(), **env_info(), **m}
    out = args.out or (args.checkpoint.parent.parent / f"metrics_{args.split}.json")
    write_json(out, payload)
    print(f"{args.split}: n={m['n']:,} acc={m['acc']:.4f} f1_macro={m['f1_macro']:.4f} "
          f"erro={m['error_rate']:.4%} ({m['n_errors']} erros)  [{dt:.0f}s]\n-> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
