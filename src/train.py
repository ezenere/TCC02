"""Train a classification backbone on the HaGRIDv2 working subset.

Runs are driven by a versioned YAML config. Every run directory records the
config, the seed, the manifest hash and per-epoch metrics, so a result can be
traced back to the exact data and settings that produced it.
"""

from __future__ import annotations

import argparse
import csv
import json
import platform
import random
import shutil
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml
from sklearn.metrics import confusion_matrix, f1_score
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from datamodule import build_dataloaders

ARCHS = {"resnet50": "ResNet50_Weights", "densenet121": "DenseNet121_Weights"}


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
    # Replace the ImageNet head with one sized for the gesture classes.
    if arch == "resnet50":
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    else:
        model.classifier = nn.Linear(model.classifier.in_features, num_classes)
    return model


def build_optimizer(model: nn.Module, cfg: dict):
    o = cfg["optim"]
    if o["name"] == "sgd":
        return torch.optim.SGD(
            model.parameters(), lr=float(o["lr"]), momentum=float(o["momentum"]),
            weight_decay=float(o["weight_decay"]), nesterov=bool(o.get("nesterov", True)))
    if o["name"] == "adamw":
        return torch.optim.AdamW(
            model.parameters(), lr=float(o["lr"]),
            weight_decay=float(o["weight_decay"]))
    raise ValueError(f"unsupported optimizer '{o['name']}'")


@torch.no_grad()
def evaluate(model, loader, device, amp_dtype, criterion, desc="eval"):
    model.eval()
    preds, targets = [], []
    loss_sum, n = 0.0, 0
    for images, labels in tqdm(loader, desc=desc, leave=False, smoothing=0.02):
        images = images.to(device, non_blocking=True).to(memory_format=torch.channels_last)
        labels = labels.to(device, non_blocking=True)
        with torch.autocast("cuda", dtype=amp_dtype):
            logits = model(images)
            loss = criterion(logits, labels)
        loss_sum += loss.item() * labels.size(0)
        n += labels.size(0)
        preds.append(logits.argmax(1).cpu())
        targets.append(labels.cpu())

    y_pred = torch.cat(preds).numpy()
    y_true = torch.cat(targets).numpy()
    return {
        "loss": loss_sum / n,
        "acc": float((y_pred == y_true).mean()),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro")),
    }, y_true, y_pred


def sanity_check(model, loader, device, amp_dtype, criterion, optimizer,
                 scaler, steps: int) -> int:
    """Short training burst: loss must move, stay finite, and fit in VRAM."""
    model.train()
    torch.cuda.reset_peak_memory_stats()
    losses: list[float] = []
    t0 = time.perf_counter()
    seen = 0
    # Steady state is measured after cuDNN autotuning and worker spin-up,
    # otherwise the warmup dominates a short burst and understates throughput.
    warmup = min(10, steps // 5)
    t_steady = None
    seen_steady = 0

    for step, (images, labels) in enumerate(loader):
        if step >= steps:
            break
        if step == warmup:
            torch.cuda.synchronize()
            t_steady = time.perf_counter()
        images = images.to(device, non_blocking=True).to(memory_format=torch.channels_last)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=amp_dtype):
            loss = criterion(model(images), labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        losses.append(loss.item())
        seen += labels.size(0)
        if step >= warmup:
            seen_steady += labels.size(0)
        if (step + 1) % 10 == 0:
            print(f"  step {step + 1:>3}/{steps}  loss={losses[-1]:.4f}", flush=True)

    torch.cuda.synchronize()
    now = time.perf_counter()
    dt = now - t0
    dt_steady = now - t_steady if t_steady else dt
    peak = torch.cuda.max_memory_allocated() / 2**30
    reserved = torch.cuda.max_memory_reserved() / 2**30
    free, total = torch.cuda.mem_get_info()
    arr = np.array(losses)
    first, last = arr[:10].mean(), arr[-10:].mean()

    print("\n=== SANITY CHECK ===")
    print(f"steps                : {len(losses)}")
    print(f"loss inicial (10 pri): {first:.4f}")
    print(f"loss final   (10 ult): {last:.4f}")
    print(f"delta                : {last - first:+.4f}  "
          f"({'CAINDO' if last < first else 'NAO CAIU'})")
    print(f"loss min / max       : {arr.min():.4f} / {arr.max():.4f}")
    print(f"NaN / Inf            : {int(np.isnan(arr).sum())} / {int(np.isinf(arr).sum())}")
    print(f"VRAM pico (alloc)    : {peak:.2f} GiB")
    print(f"VRAM pico (reserved) : {reserved:.2f} GiB")
    print(f"VRAM livre / total   : {free / 2**30:.2f} / {total / 2**30:.2f} GiB")
    print(f"throughput (c/ warmup): {seen / dt:.0f} img/s  ({dt:.1f}s / {seen} imgs)")
    print(f"throughput (regime)   : {seen_steady / dt_steady:.0f} img/s  "
          f"({dt_steady:.1f}s / {seen_steady} imgs, apos {warmup} steps de warmup)")
    n_train = len(loader.dataset)
    print(f"epoca estimada        : {n_train / (seen_steady / dt_steady) / 60:.1f} min "
          f"({n_train:,} imgs)")

    ok = bool(np.isfinite(arr).all()) and last < first
    print(f"resultado            : {'OK' if ok else 'FALHOU'}")
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--sanity", type=int, default=0,
                    help="run N training steps, report diagnostics, then exit")
    ap.add_argument("--epochs", type=int, default=None, help="override cfg epochs")
    ap.add_argument("--resume", type=Path, default=None)
    args = ap.parse_args()

    cfg = yaml.safe_load(args.config.read_text())
    if args.epochs is not None:
        cfg["train"]["epochs"] = args.epochs

    seed = int(cfg["seed"])
    set_seed(seed)
    torch.backends.cudnn.benchmark = bool(cfg["train"].get("cudnn_benchmark", True))

    device = torch.device("cuda")
    amp_dtype = {"fp16": torch.float16, "bf16": torch.bfloat16}[cfg["train"]["amp"]]

    train_loader, test_loader, meta = build_dataloaders(cfg)
    num_classes = len(meta["classes"])

    model = build_model(cfg["model"]["arch"], num_classes,
                        bool(cfg["model"]["pretrained"])).to(device)
    if cfg["train"].get("channels_last", True):
        model = model.to(memory_format=torch.channels_last)

    criterion = nn.CrossEntropyLoss(
        label_smoothing=float(cfg["optim"].get("label_smoothing", 0.0)))
    optimizer = build_optimizer(model, cfg)
    scaler = torch.amp.GradScaler("cuda", enabled=amp_dtype == torch.float16)

    epochs = int(cfg["train"]["epochs"])
    steps_per_epoch = len(train_loader)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs * steps_per_epoch) \
        if cfg["sched"]["name"] == "cosine" else None

    print(f"arch      : {cfg['model']['arch']} (pretrained={cfg['model']['pretrained']})")
    print(f"classes   : {num_classes}")
    print(f"treino    : {meta['n_train']:,} imgs / {meta['n_users_train']:,} sujeitos")
    print(f"teste     : {meta['n_test']:,} imgs / {meta['n_users_test']:,} sujeitos")
    print(f"batch     : {cfg['data']['batch_size']}  | steps/epoch: {steps_per_epoch:,}")
    print(f"manifesto : {meta['manifest_sha256'][:16]}...")
    print(f"seed      : {seed}\n")

    if args.sanity:
        return sanity_check(model, train_loader, device, amp_dtype, criterion,
                            optimizer, scaler, args.sanity)

    # ---- run directory -----------------------------------------------------
    run_dir = Path(cfg["paths"]["runs"]) / cfg["run_name"]
    (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    shutil.copy(args.config, run_dir / "config.yaml")
    (run_dir / "run_meta.json").write_text(json.dumps({
        **meta,
        "seed": seed,
        "arch": cfg["model"]["arch"],
        "epochs": epochs,
        "batch_size": cfg["data"]["batch_size"],
        "torch": torch.__version__,
        "gpu": torch.cuda.get_device_name(0),
        "python": platform.python_version(),
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "note": "cudnn.benchmark=True leaves residual non-determinism in the "
                "backward kernels; the data pipeline and init are seeded.",
    }, indent=2))

    writer = SummaryWriter(run_dir / "tb")
    csv_path = run_dir / "metrics.csv"
    fields = ["epoch", "train_loss", "train_acc", "test_loss", "test_acc",
              "test_f1_macro", "lr", "epoch_time_s", "throughput_img_s"]

    start_epoch = 0
    if args.resume and args.resume.is_file():
        ck = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ck["model"])
        optimizer.load_state_dict(ck["optimizer"])
        scaler.load_state_dict(ck["scaler"])
        if scheduler and ck.get("scheduler"):
            scheduler.load_state_dict(ck["scheduler"])
        start_epoch = ck["epoch"] + 1
        print(f"retomando da epoca {start_epoch}")
    if not csv_path.exists():
        with open(csv_path, "w", newline="") as fh:
            csv.DictWriter(fh, fieldnames=fields).writeheader()

    best_f1 = 0.0
    for epoch in range(start_epoch, epochs):
        model.train()
        t0 = time.perf_counter()
        loss_sum = correct = seen = 0
        pbar = tqdm(train_loader, desc=f"epoch {epoch + 1}/{epochs}", smoothing=0.02)
        for images, labels in pbar:
            images = images.to(device, non_blocking=True).to(memory_format=torch.channels_last)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=amp_dtype):
                logits = model(images)
                loss = criterion(logits, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            if scheduler:
                scheduler.step()

            bs = labels.size(0)
            loss_sum += loss.item() * bs
            correct += (logits.argmax(1) == labels).sum().item()
            seen += bs
            pbar.set_postfix(loss=f"{loss_sum / seen:.4f}", acc=f"{correct / seen:.4f}")

        torch.cuda.synchronize()
        epoch_time = time.perf_counter() - t0
        train_loss, train_acc = loss_sum / seen, correct / seen

        test_metrics, y_true, y_pred = evaluate(
            model, test_loader, device, amp_dtype, criterion,
            desc=f"eval {epoch + 1}/{epochs}")

        row = {
            "epoch": epoch + 1,
            "train_loss": round(train_loss, 6),
            "train_acc": round(train_acc, 6),
            "test_loss": round(test_metrics["loss"], 6),
            "test_acc": round(test_metrics["acc"], 6),
            "test_f1_macro": round(test_metrics["f1_macro"], 6),
            "lr": optimizer.param_groups[0]["lr"],
            "epoch_time_s": round(epoch_time, 2),
            "throughput_img_s": round(seen / epoch_time, 1),
        }
        with open(csv_path, "a", newline="") as fh:
            csv.DictWriter(fh, fieldnames=fields).writerow(row)
        for k, v in row.items():
            if k != "epoch":
                writer.add_scalar(k.replace("_", "/", 1), v, epoch + 1)

        print(f"epoca {epoch + 1}/{epochs}: train_loss={train_loss:.4f} "
              f"train_acc={train_acc:.4f} | test_acc={test_metrics['acc']:.4f} "
              f"test_f1={test_metrics['f1_macro']:.4f} | "
              f"{epoch_time:.1f}s  {seen / epoch_time:.0f} img/s", flush=True)

        ck = {"epoch": epoch, "model": model.state_dict(),
              "optimizer": optimizer.state_dict(), "scaler": scaler.state_dict(),
              "scheduler": scheduler.state_dict() if scheduler else None,
              "metrics": row, "classes": meta["classes"],
              "manifest_sha256": meta["manifest_sha256"], "config": cfg}
        torch.save(ck, run_dir / "checkpoints" / "last.pt")
        if test_metrics["f1_macro"] > best_f1:
            best_f1 = test_metrics["f1_macro"]
            torch.save(ck, run_dir / "checkpoints" / "best.pt")
            np.save(run_dir / "confusion_matrix.npy",
                    confusion_matrix(y_true, y_pred, labels=range(num_classes)))

    writer.close()

    import pandas as pd
    hist = pd.read_csv(csv_path)
    final = hist.iloc[-1]
    print("\n=== BASELINE: RESULTADO FINAL ===")
    print(hist.to_string(index=False))
    print(f"\nacuracia no teste (ultima epoca) : {final.test_acc:.4f}")
    print(f"F1 macro no teste (ultima epoca)  : {final.test_f1_macro:.4f}")
    print(f"melhor F1 macro                   : {hist.test_f1_macro.max():.4f} "
          f"(epoca {int(hist.loc[hist.test_f1_macro.idxmax(), 'epoch'])})")
    print(f"tempo medio por epoca             : {hist.epoch_time_s.mean():.1f} s")
    print(f"throughput medio                  : {hist.throughput_img_s.mean():.0f} img/s")
    print(f"\nrun dir      : {run_dir}")
    print(f"metricas     : {csv_path}")
    print(f"tensorboard  : {run_dir / 'tb'}")
    print(f"checkpoints  : {run_dir / 'checkpoints'} (last.pt, best.pt)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
