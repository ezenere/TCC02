"""Validate raw HaGRIDv2 data before writing the preprocessing pipeline.

Checks, for the 18 target classes across train/val/test:
  - annotation JSON files exist;
  - JSON keys resolve to existing image files;
  - bbox availability and well-formedness (unimanual: `bboxes`/`labels`;
    bimanual: `united_bbox`/`united_label`);
  - per-class image counts and unique user_id counts.

Writes a CSV report to results/ and prints a readable summary.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

# 18 original v1 classes (WACV paper).
TARGET_CLASSES = [
    "call", "dislike", "fist", "four", "like", "mute", "ok", "one",
    "palm", "peace", "peace_inverted", "rock", "stop", "stop_inverted",
    "three", "three2", "two_up", "two_up_inverted",
]

# Bimanual gestures use united_bbox/united_label. None of them is in the
# 18-class subset, but the check is kept so the pipeline stays correct if
# the subset is ever widened.
BIMANUAL_CLASSES = {
    "hand_heart", "hand_heart2", "thumb_index2", "timeout",
    "holy", "take_picture", "xsign",
}

SPLITS = ["train", "val", "test"]
IMAGE_EXT = ".jpg"


def bbox_is_valid(bbox) -> bool:
    """A bbox is [x, y, w, h] normalized, with positive extent inside [0, 1]."""
    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        return False
    if not all(isinstance(v, (int, float)) for v in bbox):
        return False
    x, y, w, h = bbox
    if w <= 0 or h <= 0:
        return False
    # Allow slight overflow outside the frame (clipped at crop time),
    # but reject boxes that fall entirely outside it.
    if x >= 1.0 or y >= 1.0 or x + w <= 0.0 or y + h <= 0.0:
        return False
    return True


def pick_annotation(gesture: str, record: dict):
    """Return (bbox, status) for the gesture hand of one image.

    status is one of: ok | missing | malformed | no_gesture_only
    """
    if gesture in BIMANUAL_CLASSES:
        bbox = record.get("united_bbox")
        label = record.get("united_label")
        if bbox is None or label is None:
            return None, "missing"
        return (bbox, "ok") if bbox_is_valid(bbox) else (None, "malformed")

    bboxes = record.get("bboxes") or []
    labels = record.get("labels") or []
    if len(bboxes) != len(labels):
        return None, "malformed"

    # Hands labeled no_gesture inside a gesture image are ignored.
    candidates = [b for b, lab in zip(bboxes, labels) if lab == gesture]
    if not candidates:
        return None, "no_gesture_only" if labels else "missing"
    valid = [b for b in candidates if bbox_is_valid(b)]
    if not valid:
        return None, "malformed"
    # Largest-area box when a gesture appears more than once.
    return max(valid, key=lambda b: b[2] * b[3]), "ok"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", type=Path,
                    default=Path("data/raw/HaGRIDv2_dataset_512"))
    ap.add_argument("--annotations", type=Path, default=Path("data/raw/annotations"))
    ap.add_argument("--out", type=Path, default=Path("results/raw_validation.csv"))
    args = ap.parse_args()

    hard_errors: list[str] = []

    # --- presence of annotation files -------------------------------------
    missing_json = [
        f"{split}/{g}.json"
        for split in SPLITS for g in TARGET_CLASSES
        if not (args.annotations / split / f"{g}.json").is_file()
    ]
    if missing_json:
        hard_errors.append(f"missing annotation JSONs: {missing_json}")

    missing_dirs = [g for g in TARGET_CLASSES if not (args.images / g).is_dir()]
    if missing_dirs:
        hard_errors.append(f"missing image directories: {missing_dirs}")

    if hard_errors:
        for e in hard_errors:
            print("ERROR:", e, file=sys.stderr)
        return 1

    rows = []
    users_by_class: dict[str, set[str]] = defaultdict(set)
    users_by_split: dict[str, set[str]] = defaultdict(set)
    user_split_conflicts = 0
    user_home: dict[str, str] = {}
    dup_keys_across_splits = 0
    total_usable = 0

    for gesture in TARGET_CLASSES:
        on_disk = {
            p.stem for p in (args.images / gesture).iterdir()
            if p.suffix.lower() == IMAGE_EXT
        }
        seen_keys: set[str] = set()

        for split in SPLITS:
            with open(args.annotations / split / f"{gesture}.json") as fh:
                data = json.load(fh)

            n_keys = len(data)
            n_missing_img = 0
            n_ok = n_missing_bbox = n_malformed = n_no_gesture_only = 0
            users: set[str] = set()

            for key, record in data.items():
                if key in seen_keys:
                    dup_keys_across_splits += 1
                seen_keys.add(key)

                if key not in on_disk:
                    n_missing_img += 1
                    continue

                _, status = pick_annotation(gesture, record)
                if status == "ok":
                    n_ok += 1
                elif status == "malformed":
                    n_malformed += 1
                elif status == "no_gesture_only":
                    n_no_gesture_only += 1
                else:
                    n_missing_bbox += 1

                uid = record.get("user_id")
                if uid:
                    users.add(uid)
                    users_by_class[gesture].add(uid)
                    users_by_split[split].add(uid)
                    if user_home.setdefault(uid, split) != split:
                        user_split_conflicts += 1

            total_usable += n_ok
            rows.append({
                "gesture": gesture, "split": split,
                "json_keys": n_keys,
                "images_on_disk_matched": n_keys - n_missing_img,
                "missing_image_file": n_missing_img,
                "bbox_ok": n_ok,
                "bbox_missing": n_missing_bbox,
                "bbox_malformed": n_malformed,
                "only_no_gesture_hands": n_no_gesture_only,
                "unique_user_ids": len(users),
            })

        extra = len(on_disk) - len(seen_keys & on_disk)
        rows.append({
            "gesture": gesture, "split": "ALL",
            "json_keys": len(seen_keys),
            "images_on_disk_matched": len(seen_keys & on_disk),
            "missing_image_file": len(seen_keys - on_disk),
            "bbox_ok": sum(r["bbox_ok"] for r in rows if r["gesture"] == gesture
                           and r["split"] != "ALL"),
            "bbox_missing": sum(r["bbox_missing"] for r in rows
                                if r["gesture"] == gesture and r["split"] != "ALL"),
            "bbox_malformed": sum(r["bbox_malformed"] for r in rows
                                  if r["gesture"] == gesture and r["split"] != "ALL"),
            "only_no_gesture_hands": sum(r["only_no_gesture_hands"] for r in rows
                                         if r["gesture"] == gesture
                                         and r["split"] != "ALL"),
            "unique_user_ids": len(users_by_class[gesture]),
        })
        rows[-1]["images_without_annotation"] = extra
        print(f"  [{gesture:>18}] keys={len(seen_keys):>6}  "
              f"usable={rows[-1]['bbox_ok']:>6}  users={len(users_by_class[gesture]):>5}",
              flush=True)

    import pandas as pd
    df = pd.DataFrame(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)

    per_class = df[df.split == "ALL"]
    all_users = set().union(*users_by_class.values())

    print("\n=== VALIDACAO DOS DADOS BRUTOS (18 classes) ===")
    print(per_class[[
        "gesture", "json_keys", "images_on_disk_matched", "missing_image_file",
        "bbox_ok", "bbox_missing", "bbox_malformed", "only_no_gesture_hands",
        "unique_user_ids",
    ]].to_string(index=False))

    print("\n--- por split (agregado nas 18 classes) ---")
    per_split = df[df.split != "ALL"].groupby("split")[[
        "json_keys", "images_on_disk_matched", "missing_image_file",
        "bbox_ok", "bbox_missing", "bbox_malformed", "only_no_gesture_hands",
    ]].sum()
    per_split["unique_user_ids"] = [len(users_by_split[s]) for s in per_split.index]
    print(per_split.to_string())

    print("\n--- totais ---")
    print(f"imagens anotadas (chaves JSON, 18 classes) : {int(per_class.json_keys.sum()):,}")
    print(f"arquivos de imagem ausentes                : {int(per_class.missing_image_file.sum()):,}")
    print(f"imagens sem bbox do gesto (ausente)        : {int(per_class.bbox_missing.sum()):,}")
    print(f"imagens com bbox malformada                : {int(per_class.bbox_malformed.sum()):,}")
    print(f"imagens so com maos no_gesture             : {int(per_class.only_no_gesture_hands.sum()):,}")
    print(f"imagens no disco sem anotacao              : {int(per_class.images_without_annotation.sum()):,}")
    print(f"IMAGENS UTILIZAVEIS (pool 100%)            : {total_usable:,}")
    print(f"user_ids unicos (18 classes)               : {len(all_users):,}")
    print(f"user_ids em mais de um split oficial       : {user_split_conflicts:,}")
    print(f"chaves duplicadas entre splits             : {dup_keys_across_splits:,}")
    print(f"\nrelatorio: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
