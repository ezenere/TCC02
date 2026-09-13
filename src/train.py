"""Train a backbone on the HaGRIDv2 working subset (axes 1, 2-finetune, 4).

One script, any architecture, seed or data fraction:
  train on `fit`, evaluate on `val` every epoch, select best.pt by val macro F1,
  optional early stopping on val, touch `test` once at the end (--eval-test).
Resumable from last.pt; the fit shuffle is re-seeded per epoch so a resumed
run replays the same batch order.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import shutil
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from compress.pruning import (apply_masks, global_magnitude_masks, masks_to_cpu,
                              masks_to_device, sparsity_report)
from datamodule import build_dataloaders
from metrics import compute_metrics, write_json
from runinfo import env_info, git_info

ARCHS = {"resnet50": "ResNet50_Weights", "densenet121": "DenseNet121_Weights"}
CSV_FIELDS = ["epoch", "train_loss", "train_acc", "val_loss", "val_acc",
              "val_f1_macro", "val_error_rate", "lr", "epoch_time_s",
              "throughput_img_s", "is_best"]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_model(arch: str, num_classes: int, pretrained: bool) -> nn.Module:
    import torchvision.models as tvm
    if arch not in ARCHS:
        raise ValueError(f"unsupported arch '{arch}', expected one of {list(ARCHS)}")
    weights = getattr(tvm, ARCHS[arch]).DEFAULT if pretrained else None
    model = getattr(tvm, arch)(weights=weights)
    if arch == "resnet50":
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    else:
        model.classifier = nn.Linear(model.classifier.in_features, num_classes)
    return model


def build_optimizer(model: nn.Module, cfg: dict):
    o = cfg["optim"]
    if o["name"] == "sgd":
        return torch.optim.SGD(model.parameters(), lr=float(o["lr"]),
                               momentum=float(o["momentum"]),
                               weight_decay=float(o["weight_decay"]),
                               nesterov=bool(o.get("nesterov", True)))
    if o["name"] == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=float(o["lr"]),
                                 weight_decay=float(o["weight_decay"]))
    raise ValueError(f"unsupported optimizer '{o['name']}'")


def build_scheduler(optimizer, cfg: dict, steps_per_epoch: int, epochs: int):
    """Linear warmup (per step) followed by cosine to zero."""
    warmup = int(round(float(cfg["sched"].get("warmup_epochs", 0)) * steps_per_epoch))
    total = epochs * steps_per_epoch

    def lr_lambda(step: int) -> float:
        if step < warmup:
            return (step + 1) / warmup
        if cfg["sched"]["name"] != "cosine":
            return 1.0
        progress = (step - warmup) / max(1, total - warmup)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def to_device(images, labels, device):
    return (images.to(device, non_blocking=True).to(memory_format=torch.channels_last),
            labels.to(device, non_blocking=True))


@torch.no_grad()
def evaluate(model, loader, device, amp_dtype, criterion, classes, desc="eval"):
    model.eval()
    preds, targets = [], []
    loss_sum, n = 0.0, 0
    for images, labels in tqdm(loader, desc=desc, leave=False, smoothing=0.02):
        images, labels = to_device(images, labels, device)
        with torch.autocast("cuda", dtype=amp_dtype):
            logits = model(images)
            loss = criterion(logits, labels)
        loss_sum += loss.item() * labels.size(0)
        n += labels.size(0)
        preds.append(logits.argmax(1).cpu())
        targets.append(labels.cpu())
    m = compute_metrics(torch.cat(targets).numpy(), torch.cat(preds).numpy(), classes)
    m["loss"] = loss_sum / max(1, n)
    return m


def train_one_epoch(model, loader, device, amp_dtype, criterion, optimizer,
                    scaler, scheduler, desc, masks=None):
    model.train()
    loss_sum = correct = seen = 0
    t0 = time.perf_counter()
    pbar = tqdm(loader, desc=desc, smoothing=0.02, leave=False)
    for images, labels in pbar:
        images, labels = to_device(images, labels, device)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=amp_dtype):
            logits = model(images)
            loss = criterion(logits, labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        if masks:
            apply_masks(model, masks)          # keep pruned weights at zero
        bs = labels.size(0)
        loss_sum += loss.item() * bs
        correct += (logits.argmax(1) == labels).sum().item()
        seen += bs
        pbar.set_postfix(loss=f"{loss_sum / seen:.4f}", acc=f"{correct / seen:.4f}",
                         lr=f"{optimizer.param_groups[0]['lr']:.2e}", refresh=False)
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    return loss_sum / seen, correct / seen, dt, seen / dt


def sanity_check(model, loader, device, amp_dtype, criterion, optimizer, scaler,
                 scheduler, steps: int) -> int:
    model.train()
    torch.cuda.reset_peak_memory_stats()
    losses, lrs = [], []
    warmup = min(10, steps // 5)
    t0 = time.perf_counter()
    t_steady, seen, seen_steady = None, 0, 0
    for step, (images, labels) in enumerate(loader):
        if step >= steps:
            break
        if step == warmup:
            torch.cuda.synchronize()
            t_steady = time.perf_counter()
        images, labels = to_device(images, labels, device)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=amp_dtype):
            loss = criterion(model(images), labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        losses.append(loss.item())
        lrs.append(optimizer.param_groups[0]["lr"])
        seen += labels.size(0)
        if step >= warmup:
            seen_steady += labels.size(0)
        if (step + 1) % 10 == 0:
            print(f"  step {step + 1:>4}/{steps}  loss={losses[-1]:.4f}  lr={lrs[-1]:.2e}", flush=True)
    torch.cuda.synchronize()
    now = time.perf_counter()
    dt, dt_steady = now - t0, (now - t_steady) if t_steady else (now - t0)
    arr = np.array(losses)
    first, last = arr[:10].mean(), arr[-10:].mean()
    ips = seen_steady / dt_steady
    print("\n=== SANITY CHECK ===")
    print(f"steps                 : {len(losses)}")
    print(f"loss 10 primeiros/ultimos: {first:.4f} -> {last:.4f}  ({'CAINDO' if last < first else 'NAO CAIU'})")
    print(f"NaN / Inf             : {int(np.isnan(arr).sum())} / {int(np.isinf(arr).sum())}")
    print(f"lr primeiro -> ultimo : {lrs[0]:.2e} -> {lrs[-1]:.2e}  (warmup {'visivel' if lrs[-1] > lrs[0] else 'ausente'})")
    print(f"VRAM pico alloc/reserv: {torch.cuda.max_memory_allocated() / 2**30:.2f} / "
          f"{torch.cuda.max_memory_reserved() / 2**30:.2f} GiB")
    print(f"throughput (regime)   : {ips:.0f} img/s  ({dt_steady:.1f}s / {seen_steady} imgs)")
    print(f"epoca estimada        : {len(loader.dataset) / ips / 60:.1f} min ({len(loader.dataset):,} imgs)")
    ok = bool(np.isfinite(arr).all()) and last < first
    print(f"resultado             : {'OK' if ok else 'FALHOU'}")
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=None, help="override cfg seed")
    ap.add_argument("--run-name", type=str, default=None, help="override cfg run_name")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--frac-column", type=str, default=None,
                    help="boolean manifest column restricting the fit split (axis 4)")
    ap.add_argument("--resume", action="store_true", help="continue from <run>/checkpoints/last.pt")
    ap.add_argument("--eval-test", action="store_true",
                    help="evaluate best.pt on the test split at the end (once)")
    ap.add_argument("--init-from", type=Path, default=None,
                    help="checkpoint whose weights initialise the model (axis 2 fine-tuning)")
    ap.add_argument("--sparsity", type=float, default=None,
                    help="global magnitude pruning target applied before training (axis 2)")
    ap.add_argument("--sanity", type=int, default=0)
    ap.add_argument("--limit-fit", type=int, default=None, help="smoke tests only")
    ap.add_argument("--limit-val", type=int, default=None, help="smoke tests only")
    args = ap.parse_args()

    cfg = yaml.safe_load(args.config.read_text())
    if args.seed is not None:
        cfg["seed"] = args.seed
    if args.epochs is not None:
        cfg["train"]["epochs"] = args.epochs
    if args.run_name:
        cfg["run_name"] = args.run_name
    if args.init_from is not None:
        cfg["model"]["init_from"] = str(args.init_from)
    if args.sparsity is not None:
        cfg.setdefault("prune", {})["sparsity"] = args.sparsity
    prune_cfg = cfg.get("prune") or {}
    sparsity = prune_cfg.get("sparsity")
    sparsity = float(sparsity) if sparsity is not None else None
    cfg["run_name"] = cfg["run_name"].format(
        arch=cfg["model"]["arch"], seed=cfg["seed"], frac=args.frac_column or "full",
        sparsity=f"{int(round(100 * sparsity)):02d}" if sparsity is not None else "00")
    seed = int(cfg["seed"])
    set_seed(seed)
    bench = bool(cfg["train"].get("cudnn_benchmark", True))
    torch.backends.cudnn.benchmark = bench
    torch.backends.cudnn.deterministic = not bench

    device = torch.device("cuda")
    amp_dtype = {"fp16": torch.float16, "bf16": torch.bfloat16}[cfg["train"]["amp"]]

    fit_loader, val_loader, test_loader, meta, generator = build_dataloaders(
        cfg, args.frac_column, args.limit_fit, args.limit_val)
    classes = meta["classes"]
    if val_loader is None and not args.sanity:
        raise SystemExit("definitive runs need a v2+ manifest with inner_split (val)")

    model = build_model(cfg["model"]["arch"], len(classes), bool(cfg["model"]["pretrained"])).to(device)
    init_from = cfg["model"].get("init_from")
    if init_from:
        src_ck = torch.load(init_from, map_location=device, weights_only=False)
        model.load_state_dict(src_ck["model"])
        if src_ck.get("classes") and src_ck["classes"] != classes:
            raise SystemExit("init_from checkpoint has a different class list")
    masks = None
    if sparsity is not None:
        masks = global_magnitude_masks(model, sparsity)
        apply_masks(model, masks)
        rep0 = sparsity_report(model, masks)
        print(f"poda global por magnitude: alvo {sparsity:.2%} -> obtida "
              f"{rep0['sparsity_prunable']:.4%} dos pesos podáveis "
              f"({rep0['params_nonzero'] / 1e6:.2f} M não-nulos de {rep0['params_total'] / 1e6:.2f} M)")
    if cfg["train"].get("channels_last", True):
        model = model.to(memory_format=torch.channels_last)
    criterion = nn.CrossEntropyLoss(label_smoothing=float(cfg["optim"].get("label_smoothing", 0.0)))
    optimizer = build_optimizer(model, cfg)
    scaler = torch.amp.GradScaler("cuda", enabled=amp_dtype == torch.float16)
    epochs = int(cfg["train"]["epochs"])
    steps_per_epoch = len(fit_loader)
    scheduler = build_scheduler(optimizer, cfg, steps_per_epoch, epochs)
    patience = int(cfg["train"].get("early_stopping_patience", 0) or 0)

    print(f"run       : {cfg['run_name']}")
    print(f"arch      : {cfg['model']['arch']} (pretrained={cfg['model']['pretrained']})  seed={seed}")
    print(f"manifesto : v{meta['manifest_version']} {meta['manifest_sha256'][:16]}...  mask={args.frac_column}")
    print(f"fit/val/test: {meta['n_fit']:,} / {meta['n_val']:,} / {meta['n_test']:,} imgs  "
          f"({meta['n_users_fit']:,} / {meta['n_users_val']:,} / {meta['n_users_test']:,} sujeitos)")
    print(f"batch {cfg['data']['batch_size']} | steps/epoch {steps_per_epoch:,} | epochs {epochs} | "
          f"warmup {cfg['sched'].get('warmup_epochs', 0)} ep | early stop patience {patience or 'off'}\n")

    if args.sanity:
        return sanity_check(model, fit_loader, device, amp_dtype, criterion, optimizer,
                            scaler, scheduler, args.sanity)

    # ---- run directory ------------------------------------------------------
    run_dir = Path(cfg["paths"]["runs"]) / cfg["run_name"]
    ckpt_dir = run_dir / "checkpoints"
    last_path, best_path = ckpt_dir / "last.pt", ckpt_dir / "best.pt"
    if run_dir.exists() and not args.resume and (run_dir / "metrics.csv").exists():
        raise SystemExit(f"{run_dir} already exists; pass --resume or choose another --run-name")
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(args.config, run_dir / "config.yaml")
    (run_dir / "config_resolved.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))

    state = {"epoch": 0, "best_f1": -1.0, "best_epoch": None, "epochs_without_improvement": 0,
             "stopped_early": False}
    if args.resume and last_path.exists():
        ck = torch.load(last_path, map_location=device, weights_only=False)
        model.load_state_dict(ck["model"])
        optimizer.load_state_dict(ck["optimizer"])
        scaler.load_state_dict(ck["scaler"])
        scheduler.load_state_dict(ck["scheduler"])
        state = ck["state"]
        state["epoch"] = ck["epoch"] + 1
        if ck.get("prune_masks"):
            masks = masks_to_device(ck["prune_masks"], device)
            apply_masks(model, masks)
        print(f"retomando: epoca {state['epoch'] + 1}/{epochs}, best F1 {state['best_f1']:.4f} "
              f"(epoca {state['best_epoch']})")

    meta_path = run_dir / "run_meta.json"
    run_meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    run_meta.update({**meta, "run_name": cfg["run_name"], "seed": seed,
                     "arch": cfg["model"]["arch"], "epochs": epochs,
                     "batch_size": cfg["data"]["batch_size"], "frac_column": args.frac_column,
                     "init_from": init_from, "prune_sparsity": sparsity,
                     "cudnn_benchmark": bench,
                     "started_at": run_meta.get("started_at") or time.strftime("%Y-%m-%dT%H:%M:%S"),
                     **git_info(), **env_info(),
                     "note": "cudnn.benchmark=True leaves residual non-determinism in the "
                             "backward kernels; the data pipeline, init and augmentation are seeded."})
    if args.resume:
        run_meta.setdefault("resumed_at", []).append(time.strftime("%Y-%m-%dT%H:%M:%S"))
    write_json(meta_path, run_meta)

    writer = SummaryWriter(run_dir / "tb")
    csv_path = run_dir / "metrics.csv"
    if not csv_path.exists():
        with open(csv_path, "w", newline="") as fh:
            csv.DictWriter(fh, fieldnames=CSV_FIELDS).writeheader()

    # ---- epochs -------------------------------------------------------------
    for epoch in range(state["epoch"], epochs):
        generator.manual_seed(seed * 100_003 + epoch)     # replayable batch order
        tr_loss, tr_acc, dt, ips = train_one_epoch(
            model, fit_loader, device, amp_dtype, criterion, optimizer, scaler, scheduler,
            desc=f"epoch {epoch + 1}/{epochs}", masks=masks)
        val = evaluate(model, val_loader, device, amp_dtype, criterion, classes,
                       desc=f"val {epoch + 1}/{epochs}")

        is_best = val["f1_macro"] > state["best_f1"]
        if is_best:
            state.update(best_f1=val["f1_macro"], best_epoch=epoch + 1, epochs_without_improvement=0)
        else:
            state["epochs_without_improvement"] += 1

        row = {"epoch": epoch + 1, "train_loss": round(tr_loss, 6), "train_acc": round(tr_acc, 6),
               "val_loss": round(val["loss"], 6), "val_acc": round(val["acc"], 6),
               "val_f1_macro": round(val["f1_macro"], 6), "val_error_rate": round(val["error_rate"], 6),
               "lr": optimizer.param_groups[0]["lr"], "epoch_time_s": round(dt, 2),
               "throughput_img_s": round(ips, 1), "is_best": int(is_best)}
        with open(csv_path, "a", newline="") as fh:
            csv.DictWriter(fh, fieldnames=CSV_FIELDS).writerow(row)
        for k, v in row.items():
            if k != "epoch":
                writer.add_scalar(k.replace("_", "/", 1), v, epoch + 1)
        print(f"epoca {epoch + 1}/{epochs}: train {tr_loss:.4f}/{tr_acc:.4f} | "
              f"val loss {val['loss']:.4f} acc {val['acc']:.4f} f1 {val['f1_macro']:.4f} "
              f"err {val['error_rate']:.4%} {'*' if is_best else ''} | {dt:.0f}s {ips:.0f} img/s", flush=True)

        ck = {"epoch": epoch, "model": model.state_dict(), "optimizer": optimizer.state_dict(),
              "scaler": scaler.state_dict(), "scheduler": scheduler.state_dict(), "state": state,
              "val_metrics": {k: v for k, v in val.items() if k not in ("confusion_matrix", "per_class")},
              "classes": classes, "manifest_sha256": meta["manifest_sha256"],
              "manifest_version": meta["manifest_version"], "config": cfg, "seed": seed,
              "prune_masks": masks_to_cpu(masks) if masks else None}
        torch.save(ck, last_path)
        if is_best:
            torch.save(ck, best_path)
            write_json(run_dir / "metrics_val_best.json",
                       {"split": "val", "checkpoint": "best.pt", "epoch": epoch + 1, **val})

        if patience and state["epochs_without_improvement"] >= patience:
            state["stopped_early"] = True
            print(f"early stopping: {patience} epocas sem melhora em val (best epoca {state['best_epoch']})")
            torch.save(ck, last_path)
            break
    writer.close()

    if masks:
        final_rep = sparsity_report(model, masks)
        run_meta["sparsity_achieved"] = final_rep["sparsity_prunable"]
        run_meta["params_nonzero"] = final_rep["params_nonzero"]
        assert abs(final_rep["sparsity_prunable"] - sparsity) < 1e-3, \
            f"sparsity drifted: {final_rep['sparsity_prunable']:.4%} vs alvo {sparsity:.2%}"
    run_meta.update({"finished_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                     "epochs_run": state["epoch"] if state["stopped_early"] else epochs,
                     "best_epoch": state["best_epoch"], "best_val_f1_macro": state["best_f1"],
                     "stopped_early": state["stopped_early"]})
    write_json(meta_path, run_meta)

    # ---- test, once, on the val-selected checkpoint --------------------------
    if args.eval_test:
        ck = torch.load(best_path, map_location=device, weights_only=False)
        model.load_state_dict(ck["model"])
        test = evaluate(model, test_loader, device, amp_dtype, criterion, classes, desc="test")
        payload = {"split": "test", "checkpoint": "best.pt", "selected_by": "val f1_macro",
                   "epoch": ck["epoch"] + 1, "run_name": cfg["run_name"], "arch": cfg["model"]["arch"],
                   "seed": seed, "frac_column": args.frac_column,
                   "init_from": init_from, "prune_sparsity": sparsity,
                   "sparsity_achieved": run_meta.get("sparsity_achieved"),
                   "params_nonzero": run_meta.get("params_nonzero"),
                   "manifest_version": meta["manifest_version"], "manifest_sha256": meta["manifest_sha256"],
                   **git_info(), **test}
        write_json(run_dir / "metrics.json", payload)
        np.save(run_dir / "confusion_matrix.npy", np.array(test["confusion_matrix"]))
        print(f"\nTESTE (best.pt, epoca {ck['epoch'] + 1}): acc {test['acc']:.4f}  f1 {test['f1_macro']:.4f}  "
              f"erro {test['error_rate']:.4%} ({test['n_errors']} / {test['n']})")
        print(f"-> {run_dir / 'metrics.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
