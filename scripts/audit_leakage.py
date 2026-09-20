"""Leakage audit of the working subset (requested after the ~99.8% accuracy).

Part A — manifest vs raw annotations (independent re-derivation):
  A1 subjects disjoint across fit / val / test;
  A2 image keys unique; no key in two splits;
  A3 every manifest row re-checked against the raw HaGRID JSONs (user_id, class);
  A4 keys that appear in more than one class JSON, and whether they straddle splits;
  A5 cross-tab of our split vs HaGRID's official split (informational).
Part B — pixel content (catches "same photo under another user_id"):
  B1 exact duplicates across splits (md5 of the processed JPEG);
  B2 near duplicates across splits (64-bit dHash, Hamming <= 3, 4x16-bit banding),
     with a contact sheet of the closest pairs for visual inspection.

Writes results/audit/leakage_audit.json, near_duplicates.csv, near_duplicates.png.
"""

from __future__ import annotations

import hashlib
import json
import multiprocessing as mp
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/audit"
MANIFEST = ROOT / "data/processed/manifest_v3.csv"
RAW_ANN = ROOT / "data/raw/annotations"
POP = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)


def hashes(path: str) -> tuple[str, int]:
    data = Path(path).read_bytes()
    with Image.open(path) as im:
        g = np.asarray(im.convert("L").resize((9, 8), Image.BILINEAR), dtype=np.int16)
    bits = (g[:, 1:] > g[:, :-1]).flatten()
    return hashlib.md5(data).hexdigest(), int(np.packbits(bits).view(">u8")[0])


def hamming(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    x = np.bitwise_xor(a[:, None], b[None, :])
    return POP[x.view(np.uint8).reshape(*x.shape, 8)].sum(-1)


def cross_near_dups(h: np.ndarray, idx_a: np.ndarray, idx_b: np.ndarray, max_dist: int = 3):
    """Pairs (i in A, j in B, dist) with Hamming <= max_dist. Banding: 4 bands of 16 bits —
    two hashes within distance 3 must agree exactly on at least one band."""
    pairs = {}
    for band in range(4):
        key = (h >> np.uint64(16 * band)) & np.uint64(0xFFFF)
        ka, kb = key[idx_a], key[idx_b]
        order_b = np.argsort(kb, kind="stable")
        kb_sorted = kb[order_b]
        for k in np.intersect1d(np.unique(ka), np.unique(kb)):
            ia = idx_a[ka == k]
            lo, hi = np.searchsorted(kb_sorted, k, "left"), np.searchsorted(kb_sorted, k, "right")
            ib = idx_b[order_b[lo:hi]]
            d = hamming(h[ia], h[ib])
            for r, c in zip(*np.nonzero(d <= max_dist)):
                pairs[(int(ia[r]), int(ib[c]))] = int(d[r, c])
    return pairs


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(MANIFEST, dtype=str, keep_default_na=False)
    df["key"] = df.image_path.map(lambda p: Path(p).stem)
    rep: dict = {"manifest": str(MANIFEST.relative_to(ROOT)), "n_images": len(df)}

    # ---------------- A1 / A2 ----------------
    users = {s: set(df.loc[df.inner_split == s, "user_id"]) for s in ("fit", "val", "test")}
    rep["A1_subject_overlap"] = {"fit∩test": len(users["fit"] & users["test"]), "val∩test": len(users["val"] & users["test"]),
                                 "fit∩val": len(users["fit"] & users["val"]),
                                 "n_users": {k: len(v) for k, v in users.items()}}
    rep["A2_keys"] = {"duplicate_image_paths": int(df.image_path.duplicated().sum()),
                      "keys_in_more_than_one_split": int((df.groupby("key").inner_split.nunique() > 1).sum()),
                      "duplicate_keys_total": int(df.key.duplicated().sum())}

    # ---------------- A3 / A4 / A5: raw annotations ----------------
    raw_user, raw_classes, raw_official = {}, defaultdict(set), {}
    classes = sorted(df.label.unique())
    for split in ("train", "val", "test"):
        for cls in classes:
            data = json.loads((RAW_ANN / split / f"{cls}.json").read_text())
            for k, rec in data.items():
                raw_classes[k].add(cls)
                raw_official[k] = split
                prev = raw_user.setdefault(k, rec["user_id"])
                assert prev == rec["user_id"], f"raw key {k} has two user_ids"
    in_raw = df.key.isin(raw_user.keys())
    uid_ok = df.key.map(raw_user) == df.user_id
    cls_ok = [lab in raw_classes.get(k, ()) for k, lab in zip(df.key, df.label)]
    multi = {k for k, c in raw_classes.items() if len(c) > 1}
    rep["A3_raw_crosscheck"] = {"rows_found_in_raw": int(in_raw.sum()), "user_id_matches": int(uid_ok.sum()),
                                "label_matches": int(sum(cls_ok)), "n_rows": len(df)}
    rep["A4_multi_class_keys"] = {"raw_keys_in_2+_class_jsons": len(multi),
                                  "of_which_in_manifest": int(df.key.isin(multi).sum())}
    df["official"] = df.key.map(raw_official)
    rep["A5_ours_vs_official_split"] = pd.crosstab(df.inner_split, df.official).to_dict()

    # a user_id must never sit in two of HaGRID's own splits either (sanity of the id itself)
    raw_user_split = defaultdict(set)
    for k, u in raw_user.items():
        raw_user_split[u].add(raw_official[k])
    rep["A5_users_in_2+_official_splits"] = int(sum(len(v) > 1 for v in raw_user_split.values()))

    # ---------------- B: content hashes ----------------
    with mp.Pool(24) as pool:
        res = pool.map(hashes, [str(ROOT / p) for p in df.image_path], chunksize=512)
    df["md5"] = [r[0] for r in res]
    h = np.array([r[1] for r in res], dtype=np.uint64)

    md5_split = df.groupby("md5").inner_split.agg(lambda s: tuple(sorted(set(s))))
    md5_users = df.groupby("md5").user_id.nunique()
    rep["B1_exact_duplicates"] = {
        "md5_groups_with_2+_files": int((df.md5.value_counts() > 1).sum()),
        "groups_spanning_train_and_test": int(sum(("test" in t and len(t) > 1) for t in md5_split)),
        "groups_spanning_fit_and_val": int(sum(t == ("fit", "val") for t in md5_split)),
        "groups_with_2+_user_ids": int((md5_users > 1).sum())}

    idx = {s: np.flatnonzero((df.inner_split == s).to_numpy()) for s in ("fit", "val", "test")}
    train_idx = np.concatenate([idx["fit"], idx["val"]])
    pairs_tt = cross_near_dups(h, idx["test"], train_idx)
    pairs_fv = cross_near_dups(h, idx["val"], idx["fit"])
    rows = [{"test_image": df.image_path[i], "train_image": df.image_path[j], "hamming": d,
             "test_label": df.label[i], "train_label": df.label[j], "same_label": df.label[i] == df.label[j],
             "test_user": df.user_id[i][:12], "train_user": df.user_id[j][:12]} for (i, j), d in pairs_tt.items()]
    nd = pd.DataFrame(rows).sort_values(["hamming", "test_image"]) if rows else pd.DataFrame(
        columns=["test_image", "train_image", "hamming", "same_label"])
    nd.to_csv(OUT / "near_duplicates.csv", index=False)
    n_test = len(idx["test"])
    rep["B2_near_duplicates_dhash<=3"] = {
        "test_vs_train_pairs": len(nd), "distinct_test_images": int(nd.test_image.nunique()) if len(nd) else 0,
        "share_of_test": (nd.test_image.nunique() / n_test) if len(nd) else 0.0,
        "pairs_by_hamming": {int(k): int(v) for k, v in Counter(nd.hamming).items()} if len(nd) else {},
        "pairs_with_same_label": int(nd.same_label.sum()) if len(nd) else 0,
        "val_vs_fit_pairs": len(pairs_fv)}

    if len(nd):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        show = nd.head(24)
        fig, axes = plt.subplots(4, 12, figsize=(24, 9))
        for ax in axes.flat:
            ax.axis("off")
        for n, r in enumerate(show.itertuples()):
            for col, (p, tag) in enumerate(((r.test_image, f"TESTE {r.test_label}"), (r.train_image, f"treino {r.train_label}"))):
                ax = axes[n // 6, (n % 6) * 2 + col]
                ax.imshow(Image.open(ROOT / p))
                ax.set_title(f"{tag}\nd={r.hamming}", fontsize=7)
        fig.suptitle("Pares teste × treino com dHash a distância ≤ 3 (os 24 mais próximos) — inspeção visual", fontsize=12)
        fig.tight_layout()
        fig.savefig(OUT / "near_duplicates.png", dpi=110)

    (OUT / "leakage_audit.json").write_text(json.dumps(rep, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(rep, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
