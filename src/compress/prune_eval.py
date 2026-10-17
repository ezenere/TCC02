"""Eixo 2d: prune a trained model and evaluate it WITHOUT any retraining.

    PYTHONPATH=src python src/compress/prune_eval.py --arch resnet50 --seed 0 --sparsity 0.9
    PYTHONPATH=src python src/compress/prune_eval.py --all      # 2 archs x 3 seeds x 5 levels

Takes the axis-1 best.pt of the same architecture and seed, applies global
unstructured magnitude pruning (same masks as axis 2: head included), measures
the achieved sparsity and evaluates directly on the test split. No optimizer
step is taken, so the number answers "how much does fine-tuning recover?".

Writes runs/eixo2d_noft_<arch>_p<NN>_s<seed>/{metrics.json, run_meta.json,
confusion_matrix.npy, cost.json, checkpoints/best.pt}. Idempotent: a cell whose
metrics.json exists is skipped.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from compress.pruning import apply_masks, global_magnitude_masks, masks_to_cpu, sparsity_report
from datamodule import build_eval_loader
from metrics import write_json
from runinfo import env_info, git_info
from train import build_model, evaluate, set_seed

ROOT = Path(__file__).resolve().parents[2]
ARCHS = ("resnet50", "densenet121")
SEEDS = (0, 1, 2)
LEVELS = (0.5, 0.7, 0.9, 0.95, 0.98)


def run_name(arch: str, sparsity: float, seed: int) -> str:
    return f"eixo2d_noft_{arch}_p{int(round(100 * sparsity)):02d}_s{seed}"


def source_checkpoint(arch: str, seed: int) -> Path:
    return ROOT / "runs" / f"eixo1_{arch}_s{seed}" / "checkpoints" / "best.pt"


def prune_and_eval(arch: str, seed: int, sparsity: float, device, cache: dict) -> dict | None:
    """One cell. `cache` keeps the loaded source checkpoint and test loader per
    (arch, seed) so the five sparsity levels share them."""
    out_dir = ROOT / "runs" / run_name(arch, sparsity, seed)
    if (out_dir / "metrics.json").exists():
        print(f"[skip] {out_dir.name}: metrics.json existe")
        if not (out_dir / "cost.json").exists():
            measure_cost(out_dir / "checkpoints" / "best.pt")
        return None

    src = source_checkpoint(arch, seed)
    key = (arch, seed)
    if key not in cache:
        ck = torch.load(src, map_location=device, weights_only=False)
        loader, meta = build_eval_loader(ck["config"], "test")
        if ck.get("classes") and ck["classes"] != meta["classes"]:
            raise SystemExit("class list of the checkpoint differs from the manifest")
        assert ck["manifest_sha256"] == meta["manifest_sha256"], \
            "source checkpoint was trained on a different manifest version than the one on disk"
        cache[key] = (ck, loader, meta)
    ck, loader, meta = cache[key]
    cfg, classes = ck["config"], meta["classes"]
    amp_dtype = {"fp16": torch.float16, "bf16": torch.bfloat16}[cfg["train"]["amp"]]

    set_seed(seed)
    model = build_model(arch, len(classes), pretrained=False).to(device)
    model.load_state_dict(ck["model"])                   # fresh dense weights for every level
    masks = global_magnitude_masks(model, sparsity)      # head included, as in axis 2
    apply_masks(model, masks)
    rep = sparsity_report(model, masks)
    assert abs(rep["sparsity_prunable"] - sparsity) < 1e-3, \
        f"sparsity mismatch: {rep['sparsity_prunable']:.4%} vs target {sparsity:.2%}"
    model = model.to(memory_format=torch.channels_last)

    t0 = time.perf_counter()
    test = evaluate(model, loader, device, amp_dtype, nn.CrossEntropyLoss(), classes,
                    desc=f"test {out_dir.name}")
    dt = time.perf_counter() - t0

    init_from = str(src.relative_to(ROOT))
    out_dir.mkdir(parents=True, exist_ok=True)
    common = {"arch": arch, "seed": seed, "prune_sparsity": sparsity,
              "sparsity_achieved": rep["sparsity_prunable"], "params_nonzero": rep["params_nonzero"],
              "params_total": rep["params_total"], "init_from": init_from,
              "manifest_version": meta["manifest_version"], "manifest_sha256": meta["manifest_sha256"],
              **git_info(), **env_info()}
    payload = {"split": "test", "checkpoint": "best.pt", "selected_by": "none (no retraining)",
               "epoch": ck["epoch"] + 1, "run_name": out_dir.name, "frac_column": None,
               "prune_when": "no_retraining", "eval_time_s": round(dt, 1), **common, **test}
    write_json(out_dir / "metrics.json", payload)
    np.save(out_dir / "confusion_matrix.npy", np.array(test["confusion_matrix"]))

    now = time.strftime("%Y-%m-%dT%H:%M:%S")
    run_meta = {**{k: v for k, v in meta.items() if k != "classes"}, "classes": classes,
                "run_name": out_dir.name, "epochs": 0, "epochs_run": 0, "prune_when": "no_retraining",
                "prune_exclude": [], "manifest": meta["manifest"], "best_epoch": ck["epoch"] + 1,
                "started_at": now, "finished_at": now, **common,
                "note": "no optimizer step: masks computed on the axis-1 best.pt and evaluated as is"}
    write_json(out_dir / "run_meta.json", run_meta)

    cfg_out = {**cfg, "run_name": out_dir.name, "seed": seed,
               "model": {**cfg["model"], "init_from": init_from},
               "prune": {"sparsity": sparsity, "scope": "global_magnitude", "when": "no_retraining"}}
    ckpt_dir = out_dir / "checkpoints"
    ckpt_dir.mkdir(exist_ok=True)
    torch.save({"epoch": ck["epoch"], "model": model.state_dict(), "classes": classes,
                "manifest_sha256": meta["manifest_sha256"], "manifest_version": meta["manifest_version"],
                "config": cfg_out, "seed": seed, "prune_masks": masks_to_cpu(masks),
                "state": {"epoch": 0, "best_f1": None, "best_epoch": None}},
               ckpt_dir / "best.pt")

    print(f"{out_dir.name}: esparsidade {rep['sparsity_prunable']:.4%} | acc {test['acc']:.4f} "
          f"f1 {test['f1_macro']:.4f} erro {test['error_rate']:.4%} ({test['n_errors']} / {test['n']}) [{dt:.0f}s]",
          flush=True)
    measure_cost(ckpt_dir / "best.pt")
    return payload


def measure_cost(checkpoint: Path) -> None:
    """cost.json next to the run (params, MACs, disk size), path recorded relative to ROOT."""
    subprocess.run([sys.executable, str(ROOT / "src/measure/cost.py"),
                    "--checkpoint", str(checkpoint.relative_to(ROOT))], check=True, cwd=ROOT)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", choices=ARCHS)
    ap.add_argument("--seed", type=int)
    ap.add_argument("--sparsity", type=float)
    ap.add_argument("--all", action="store_true", help="every arch x seed x level cell")
    args = ap.parse_args()
    if args.all:
        cells = [(a, s, p) for a in ARCHS for s in SEEDS for p in LEVELS]
    elif args.arch is not None and args.seed is not None and args.sparsity is not None:
        cells = [(args.arch, args.seed, args.sparsity)]
    else:
        ap.error("pass --all or all of --arch --seed --sparsity")

    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda")
    cache: dict = {}
    for arch, seed, sp in cells:
        prune_and_eval(arch, seed, sp, device, cache)
        cache = {k: v for k, v in cache.items() if k == (arch, seed)}   # free the previous (arch, seed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
