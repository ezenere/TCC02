"""Impact of HaGRID multi-account subjects on the test accuracy.

Input: results/audit/near_duplicates.csv (dHash candidates from audit_leakage.py).
1. pixel check: mean absolute difference (MAD, 0-255) of 32x32 grayscale thumbnails;
   a pair is a near-identical photo when MAD < thr AND the labels agree;
2. a test USER is "suspect" if any of its images is near-identical to a training image
   (the same person/session under another user_id);
3. evaluate checkpoints on the test split without the suspect users and on the suspect
   users only (all their images, not just the duplicated ones).
Writes results/audit/clean_vs_suspect.csv.  --no-eval skips step 3 (CPU only).
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
AUD = ROOT / "results/audit"


def thumb(p: str) -> np.ndarray:
    with Image.open(ROOT / p) as im:
        return np.asarray(im.convert("L").resize((32, 32), Image.BILINEAR), dtype=np.float32)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--thresholds", type=float, nargs="+", default=[10, 20])
    ap.add_argument("--checkpoints", nargs="+", default=["runs/eixo1_resnet50_s0/checkpoints/best.pt",
                                                         "runs/eixo1_densenet121_s0/checkpoints/best.pt"])
    ap.add_argument("--no-eval", action="store_true")
    args = ap.parse_args()

    nd = pd.read_csv(AUD / "near_duplicates.csv")
    paths = sorted(set(nd.test_image) | set(nd.train_image))
    with mp.Pool(24) as pool:
        t = dict(zip(paths, pool.map(thumb, paths, chunksize=256)))
    nd["mad"] = [float(np.abs(t[a] - t[b]).mean()) for a, b in zip(nd.test_image, nd.train_image)]
    nd.to_csv(AUD / "near_duplicates.csv", index=False)

    man = pd.read_csv(ROOT / "data/processed/manifest_v2.csv", dtype=str, keep_default_na=False)
    user = dict(zip(man.image_path, man.user_id))
    n_test = int((man.split == "test").sum())
    summary, rows = [], []
    tmp = AUD / "_tmp"
    tmp.mkdir(exist_ok=True)
    for thr in args.thresholds:
        dup = nd[(nd.mad < thr) & nd.same_label]
        sus_users = {user[p] for p in dup.test_image}
        is_sus = (man.split == "test") & man.user_id.isin(sus_users)
        summary.append({"mad_thr": thr, "pairs": len(dup), "test_images_duplicated": int(dup.test_image.nunique()),
                        "suspect_test_users": len(sus_users), "suspect_test_images": int(is_sus.sum()),
                        "share_of_test": float(is_sus.sum() / n_test),
                        "train_users_involved": len({user[p] for p in dup.train_image})})
        if args.no_eval:
            continue
        for kind, frame in (("clean", man[~is_sus]), ("suspect", pd.concat([man[man.split != "test"], man[is_sus]]))):
            mpath = tmp / f"manifest_{kind}_mad{int(thr)}.csv"
            frame.to_csv(mpath, index=False)
            for ck in args.checkpoints:
                out = tmp / f"{kind}_{int(thr)}_{Path(ck).parents[1].name}.json"
                subprocess.run([sys.executable, "src/eval.py", "--checkpoint", ck, "--split", "test",
                                "--manifest", str(mpath), "--out", str(out)], cwd=ROOT, check=True,
                               env={**__import__("os").environ, "PYTHONPATH": "src"},
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                m = json.loads(out.read_text())
                rows.append({"modelo": Path(ck).parents[1].name, "conjunto": f"{kind} MAD<{int(thr)}", "n": m["n"],
                             "erros": m["n_errors"], "erro_%": round(100 * m["error_rate"], 4)})
    pd.DataFrame(summary).to_csv(AUD / "multiaccount_summary.csv", index=False)
    print(pd.DataFrame(summary).to_string(index=False))
    if rows:
        pd.DataFrame(rows).to_csv(AUD / "clean_vs_suspect_eval.csv", index=False)
        print(pd.DataFrame(rows).to_string(index=False))
    for f in tmp.glob("*"):
        f.unlink()
    tmp.rmdir()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
