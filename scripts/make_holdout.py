"""External check set: every usable image of the HaGRID users that were NEVER
sampled into the working subset (the other ~50% of the 18-class pool).

These subjects appear in no manifest version, so no model, mask, calibration or
selection ever touched them. Cropped with the exact function of preprocess.py.
Writes data/holdout_unused/{images, manifest_holdout.csv}.
"""

from __future__ import annotations

import multiprocessing as mp
import sys
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from preprocess import load_records, process_one   # noqa: E402

OUT = ROOT / "data/holdout_unused"


def main() -> int:
    cfg = yaml.safe_load((ROOT / "configs/preprocess.yaml").read_text())
    man = pd.read_csv(ROOT / "data/processed/manifest.csv", dtype=str)
    used_users = set(man.user_id)
    used_keys = set(Path(p).stem for p in man.image_path)

    records = [r for r in load_records(cfg) if r["user_id"] not in used_users]
    assert not any(r["key"] in used_keys for r in records), "holdout shares an image with the subset"
    assert not ({r["user_id"] for r in records} & used_users), "holdout shares a subject with the subset"

    crop = cfg["crop"]
    img_root = ROOT / cfg["paths"]["images"]
    for c in cfg["classes"]:
        (OUT / "images" / c).mkdir(parents=True, exist_ok=True)
    tasks = [(str(img_root / r["label"] / f"{r['key']}.jpg"), str(OUT / "images" / r["label"] / f"{r['key']}.jpg"),
              r["bbox"], float(crop["margin"]), crop["mode"], int(crop["size"]), int(crop["jpeg_quality"]), False)
             for r in records]
    with mp.Pool(24) as pool:
        errs = [e for e in pool.imap_unordered(process_one, tasks, chunksize=256) if e]
    assert not errs, errs[:3]

    rows = [{"image_path": str((OUT / "images" / r["label"] / f"{r['key']}.jpg").relative_to(ROOT)),
             "label": r["label"], "user_id": r["user_id"], "split": "test", "inner_split": "test"} for r in records]
    df = pd.DataFrame(rows).sort_values(["label", "image_path"])
    df.to_csv(OUT / "manifest_holdout.csv", index=False)
    print(f"holdout: {len(df):,} imagens, {df.user_id.nunique():,} sujeitos nunca usados, {df.label.nunique()} classes")
    print(df.label.value_counts().sort_index().to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
