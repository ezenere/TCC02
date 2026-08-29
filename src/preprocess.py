"""Build the working subset: parse annotations, sample, split, crop.

Pipeline (all steps deterministic given `seed` in the config):
  1. parse annotation JSONs for the 18 target classes (train/val/test merged);
  2. keep the gesture hand only (no_gesture hands are dropped);
  3. sample `subset.fraction` of the pool by user_id, class-balanced;
  4. holdout split by user_id (subject-disjoint), class-balanced;
  5. crop bbox with margin, resize, save JPEG;
  6. assert invariants, write manifest.csv + descriptive statistics.

Determinism contract: user_ids are explicitly sorted before any shuffle, the
shuffle is seeded, and the manifest is written in sorted order. Two runs must
produce a byte-identical manifest (see scripts/verify_determinism.sh).
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import sys
from collections import defaultdict
from pathlib import Path

import yaml
from PIL import Image
from tqdm import tqdm

# Bimanual gestures use united_bbox/united_label. None belongs to the 18-class
# subset, but the branch is kept so the pipeline stays correct if it widens.
BIMANUAL_CLASSES = {
    "hand_heart", "hand_heart2", "thumb_index2", "timeout",
    "holy", "take_picture", "xsign",
}
SPLITS_RAW = ["train", "val", "test"]


# --------------------------------------------------------------------------
# annotation parsing
# --------------------------------------------------------------------------

def bbox_is_valid(bbox) -> bool:
    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        return False
    if not all(isinstance(v, (int, float)) for v in bbox):
        return False
    x, y, w, h = bbox
    if w <= 0 or h <= 0:
        return False
    return not (x >= 1.0 or y >= 1.0 or x + w <= 0.0 or y + h <= 0.0)


def pick_bbox(gesture: str, record: dict):
    """Bbox of the gesture hand, or None. no_gesture hands are ignored."""
    if gesture in BIMANUAL_CLASSES:
        bbox = record.get("united_bbox")
        label = record.get("united_label")
        if bbox is None or label is None or not bbox_is_valid(bbox):
            return None
        return bbox

    bboxes = record.get("bboxes") or []
    labels = record.get("labels") or []
    if len(bboxes) != len(labels):
        return None
    valid = [b for b, lab in zip(bboxes, labels)
             if lab == gesture and bbox_is_valid(b)]
    if not valid:
        return None
    # Largest-area box when the gesture appears more than once.
    return max(valid, key=lambda b: b[2] * b[3])


def load_records(cfg: dict) -> list[dict]:
    """All usable images of the target classes, in a deterministic order."""
    ann_dir = Path(cfg["paths"]["annotations"])
    records: list[dict] = []
    for gesture in cfg["classes"]:
        for split in SPLITS_RAW:
            with open(ann_dir / split / f"{gesture}.json") as fh:
                data = json.load(fh)
            for key in sorted(data):
                rec = data[key]
                bbox = pick_bbox(gesture, rec)
                uid = rec.get("user_id")
                if bbox is None or not uid:
                    continue
                records.append({
                    "key": key, "label": gesture, "user_id": uid, "bbox": bbox,
                })
    records.sort(key=lambda r: (r["label"], r["key"]))
    return records


# --------------------------------------------------------------------------
# deterministic class-balanced selection by user_id
# --------------------------------------------------------------------------

def select_users_balanced(
    user_counts: dict[str, dict[str, int]],
    class_totals: dict[str, int],
    fraction: float,
    tolerances: list[float],
    seed: int,
) -> set[str]:
    """Pick a subset of users holding ~`fraction` of the images of every class.

    Users are visited in a seeded shuffle of an explicitly sorted list, so the
    result depends only on (user set, seed), never on dict or filesystem order.
    A user is accepted when it does not push any of its classes past the class
    target inflated by the current tolerance; passes relax the tolerance.
    """
    targets = {c: n * fraction for c, n in class_totals.items()}
    total_target = sum(targets.values())

    order = sorted(user_counts)                 # explicit ordering first
    random.Random(seed).shuffle(order)          # then seeded shuffle

    selected: set[str] = set()
    got: dict[str, float] = defaultdict(float)
    total = 0.0

    for tol in tolerances:
        if total >= total_target:
            break
        for uid in order:
            if uid in selected or total >= total_target:
                continue
            counts = user_counts[uid]
            if all(got[c] + n <= targets[c] * (1.0 + tol) for c, n in counts.items()):
                selected.add(uid)
                for c, n in counts.items():
                    got[c] += n
                total += sum(counts.values())
    return selected


def user_class_counts(records: list[dict]) -> dict[str, dict[str, int]]:
    counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for r in records:
        counts[r["user_id"]][r["label"]] += 1
    return {u: dict(c) for u, c in counts.items()}


# --------------------------------------------------------------------------
# cropping
# --------------------------------------------------------------------------

def crop_box(bbox, img_w: int, img_h: int, margin: float, mode: str):
    """Pixel crop box (left, top, right, bottom) from a normalized bbox."""
    x, y, w, h = bbox
    x0, y0 = x * img_w, y * img_h
    bw, bh = w * img_w, h * img_h

    # Each side expands by `margin` of the corresponding dimension.
    x0 -= bw * margin
    y0 -= bh * margin
    bw *= 1.0 + 2.0 * margin
    bh *= 1.0 + 2.0 * margin

    if mode == "square":
        cx, cy = x0 + bw / 2.0, y0 + bh / 2.0
        side = min(max(bw, bh), img_w, img_h)
        x0, y0 = cx - side / 2.0, cy - side / 2.0
        bw = bh = side
        # Shift back inside the frame instead of clipping, to keep it square.
        x0 = min(max(x0, 0.0), img_w - side)
        y0 = min(max(y0, 0.0), img_h - side)

    left = max(int(round(x0)), 0)
    top = max(int(round(y0)), 0)
    right = min(int(round(x0 + bw)), img_w)
    bottom = min(int(round(y0 + bh)), img_h)
    if right - left < 2 or bottom - top < 2:
        return None
    return left, top, right, bottom


def process_one(task):
    """Crop + resize + save one image. Idempotent: existing outputs are kept."""
    src, dst, bbox, margin, mode, size, quality, force = task
    try:
        if not force and os.path.exists(dst):
            return None
        with Image.open(src) as im:
            im = im.convert("RGB")
            box = crop_box(bbox, im.width, im.height, margin, mode)
            if box is None:
                return f"degenerate crop: {src}"
            out = im.crop(box).resize((size, size), Image.BICUBIC)
        tmp = dst + ".tmp"
        out.save(tmp, "JPEG", quality=quality, subsampling=0)
        os.replace(tmp, dst)          # atomic: no truncated file on interrupt
        return None
    except Exception as exc:                                  # noqa: BLE001
        return f"{src}: {exc!r}"


# --------------------------------------------------------------------------
# reporting
# --------------------------------------------------------------------------

def write_stats(rows: list[dict], records_by_key: dict, out_dir: Path,
                classes: list[str]) -> "object":
    import pandas as pd

    df = pd.DataFrame(rows)
    per = (df.groupby(["label", "split"]).size().unstack(fill_value=0)
             .reindex(classes))
    users = (df.groupby(["label", "split"])["user_id"].nunique()
               .unstack(fill_value=0).reindex(classes))
    per.columns = [f"imgs_{c}" for c in per.columns]
    users.columns = [f"users_{c}" for c in users.columns]
    stats = per.join(users)
    stats["imgs_total"] = stats[[c for c in stats if c.startswith("imgs_")]].sum(axis=1)
    stats["pct_test"] = (100.0 * stats["imgs_test"] / stats["imgs_total"]).round(2)

    out_dir.mkdir(parents=True, exist_ok=True)
    stats.to_csv(out_dir / "subset_stats.csv")
    return stats


def write_manifest(subset: list[dict], proc_root: Path, out: Path) -> Path:
    """Write manifest.csv in a fully sorted order, which leads to a byte-identical execution across runs."""
    rows = [{
        "image_path": os.path.relpath(
            proc_root / r["split"] / r["label"] / f"{r['key']}.jpg", start="."),
        "label": r["label"], "user_id": r["user_id"], "split": r["split"],
    } for r in subset]
    rows.sort(key=lambda d: (d["split"], d["label"], d["image_path"]))

    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="\n") as fh:
        w = csv.DictWriter(fh, fieldnames=["image_path", "label", "user_id", "split"],
                           lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    return out


def manifest_hash(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------

def build_plan(cfg: dict):
    """Records annotated with their split, which is the full deterministic decision."""
    seed = int(cfg["seed"])
    records = load_records(cfg)

    counts = user_class_counts(records)
    class_totals: dict[str, int] = defaultdict(int)
    for r in records:
        class_totals[r["label"]] += 1

    subset_users = select_users_balanced(
        counts, class_totals, float(cfg["subset"]["fraction"]),
        list(cfg["subset"]["tolerances"]), seed,
    )
    subset = [r for r in records if r["user_id"] in subset_users]

    sub_counts = {u: counts[u] for u in subset_users}
    sub_totals: dict[str, int] = defaultdict(int)
    for r in subset:
        sub_totals[r["label"]] += 1

    train_users = select_users_balanced(
        sub_counts, sub_totals, float(cfg["subset"]["train_fraction"]),
        list(cfg["subset"]["tolerances"]), seed + 1,
    )
    for r in subset:
        r["split"] = "train" if r["user_id"] in train_users else "test"
    return subset, train_users, subset_users - train_users


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=Path("configs/preprocess.yaml"))
    ap.add_argument("--sample", type=int, default=0,
                    help="process only N images into a scratch dir (dry run)")
    ap.add_argument("--sample-out", type=Path, default=Path("results/figures/sample_check"))
    ap.add_argument("--force", action="store_true", help="rewrite existing crops")
    ap.add_argument("--plan-only", action="store_true",
                    help="compute the split and report, write no images")
    ap.add_argument("--manifest-out", type=Path, default=None,
                    help="write the manifest here instead of the config path")
    args = ap.parse_args()

    cfg = yaml.safe_load(args.config.read_text())
    img_root = Path(cfg["paths"]["images"])
    proc_root = Path(cfg["paths"]["processed"])
    results = Path(cfg["paths"]["results"])
    crop_cfg = cfg["crop"]

    print("[1/6] parsing annotations + planning split...", flush=True)
    subset, train_users, test_users = build_plan(cfg)
    print(f"      subset={len(subset):,} imgs  "
          f"users train={len(train_users):,} test={len(test_users):,}", flush=True)

    if args.plan_only:
        man = None
        if args.manifest_out:
            man = write_manifest(subset, proc_root, args.manifest_out)
        _report(subset, train_users, test_users, results, cfg, manifest=man)
        return 0

    # ---- sample mode: small dry run for visual validation -----------------
    if args.sample:
        rng = random.Random(int(cfg["seed"]) + 99)
        picks = rng.sample(subset, min(args.sample, len(subset)))
        out = args.sample_out
        out.mkdir(parents=True, exist_ok=True)
        tasks = [(
            str(img_root / r["label"] / f"{r['key']}.jpg"),
            str(out / f"{r['label']}__{r['key']}.jpg"),
            r["bbox"], float(crop_cfg["margin"]), crop_cfg["mode"],
            int(crop_cfg["size"]), int(crop_cfg["jpeg_quality"]), True,
        ) for r in picks]
        errs = [e for e in map(process_one, tqdm(tasks, desc="sample")) if e]
        print(f"      {len(tasks) - len(errs)} crops -> {out}")
        for e in errs[:10]:
            print("      ERR", e)
        return 0

    # ---- full run ---------------------------------------------------------
    print("[2/6] preparing output tree...", flush=True)
    for split in ("train", "test"):
        for c in cfg["classes"]:
            (proc_root / split / c).mkdir(parents=True, exist_ok=True)

    tasks = [(
        str(img_root / r["label"] / f"{r['key']}.jpg"),
        str(proc_root / r["split"] / r["label"] / f"{r['key']}.jpg"),
        r["bbox"], float(crop_cfg["margin"]), crop_cfg["mode"],
        int(crop_cfg["size"]), int(crop_cfg["jpeg_quality"]), args.force,
    ) for r in subset]

    print(f"[3/6] cropping {len(tasks):,} images "
          f"({cfg['runtime']['workers']} workers)...", flush=True)
    import multiprocessing as mp
    errors: list[str] = []
    with mp.Pool(int(cfg["runtime"]["workers"])) as pool:
        for err in tqdm(pool.imap_unordered(process_one, tasks, chunksize=64),
                        total=len(tasks), desc="crop", smoothing=0.02):
            if err:
                errors.append(err)
    if errors:
        print(f"      {len(errors)} failures; first 10:", file=sys.stderr)
        for e in errors[:10]:
            print("      ", e, file=sys.stderr)

    # ---- invariants (hard failures) ---------------------------------------
    print("[4/6] asserting invariants...", flush=True)
    assert not (train_users & test_users), (
        f"subject leakage: {len(train_users & test_users)} user_ids in both splits")

    failed = [t[1] for t in tasks if not os.path.exists(t[1])]
    assert not failed, (
        f"{len(failed)} manifest images missing on disk, e.g. {failed[:3]}")

    labels_seen = {r["label"] for r in subset}
    assert labels_seen == set(cfg["classes"]), (
        f"class mismatch: {set(cfg['classes']) - labels_seen} absent")
    for split in ("train", "test"):
        present = {r["label"] for r in subset if r["split"] == split}
        assert present == set(cfg["classes"]), (
            f"split '{split}' missing classes: {set(cfg['classes']) - present}")
    print("      OK: no user_id overlap, every image on disk, 18 classes in both splits")

    # ---- manifest ---------------------------------------------------------
    print("[5/6] writing manifest...", flush=True)
    manifest = write_manifest(
        subset, proc_root, args.manifest_out or Path(cfg["paths"]["manifest"]))

    print("[6/6] statistics...", flush=True)
    _report(subset, train_users, test_users, results, cfg, manifest=manifest)
    return 0


def _report(subset, train_users, test_users, results: Path, cfg: dict, manifest):
    rows = [{"label": r["label"], "split": r["split"], "user_id": r["user_id"]}
            for r in subset]
    stats = write_stats(rows, {}, results, cfg["classes"])

    n_train = sum(1 for r in subset if r["split"] == "train")
    n_test = len(subset) - n_train

    print("\n=== SUBCONJUNTO DE TRABALHO: 18 classes ===")
    print(stats.to_string())
    print("\n--- totais ---")
    print(f"total de imagens do subconjunto : {len(subset):,}")
    print(f"imagens de treino (70%)         : {n_train:,} "
          f"({100.0 * n_train / len(subset):.2f}%)")
    print(f"imagens de teste  (30%)         : {n_test:,} "
          f"({100.0 * n_test / len(subset):.2f}%)")
    print(f"sujeitos no treino              : {len(train_users):,}")
    print(f"sujeitos no teste               : {len(test_users):,}")
    print(f"sujeitos no total               : {len(train_users | test_users):,}")
    print(f"sobreposicao de sujeitos        : {len(train_users & test_users)}")
    if manifest is not None:
        print(f"manifesto                       : {manifest}")
        print(f"sha256 do manifesto             : {manifest_hash(manifest)}")
    print(f"seed                            : {cfg['seed']}")
    print(f"estatisticas                    : {results / 'subset_stats.csv'}")


if __name__ == "__main__":
    raise SystemExit(main())
