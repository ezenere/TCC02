"""Verify a manifest version against its invariants and MANIFESTS.md.

Exit code 1 on any violation — this is the completion criterion of the
manifest tasks, so it must fail loudly rather than print warnings.

    python scripts/verify_manifest.py --version 1
    python scripts/verify_manifest.py --version 2
"""

from __future__ import annotations

import argparse
import hashlib
import re
import sys
from pathlib import Path

import pandas as pd

PROCESSED = Path("data/processed")
REGISTRY = PROCESSED / "MANIFESTS.md"
EXPECTED = {"n_total": 278_715, "n_train": 195_102, "n_test": 83_613,
            "n_classes": 18, "users_train": 14_073, "users_test": 6_044}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def manifest_path(version: int) -> Path:
    return PROCESSED / ("manifest.csv" if version == 1 else f"manifest_v{version}.csv")


def registered_sha(version: int) -> str | None:
    if not REGISTRY.exists():
        return None
    m = re.search(rf"^\|\s*v{version}\s*\|[^|]*\|\s*`([0-9a-f]{{64}})`",
                  REGISTRY.read_text(), re.M)
    return m.group(1) if m else None


def check(cond: bool, msg: str, failures: list[str]) -> None:
    print(("  ok   " if cond else "  FAIL ") + msg)
    if not cond:
        failures.append(msg)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", type=int, required=True)
    args = ap.parse_args()
    v = args.version
    path = manifest_path(v)
    failures: list[str] = []

    print(f"manifesto v{v}: {path}")
    if not path.exists():
        print("  FAIL arquivo inexistente")
        return 1

    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    base = ["image_path", "label", "user_id", "split"]
    check(list(df.columns[:4]) == base, f"colunas base {base}", failures)

    # --- base invariants (all versions) ------------------------------------
    tr, te = df[df.split == "train"], df[df.split == "test"]
    check(len(df) == EXPECTED["n_total"], f"total = {len(df):,}", failures)
    check(len(tr) == EXPECTED["n_train"], f"treino = {len(tr):,}", failures)
    check(len(te) == EXPECTED["n_test"], f"teste = {len(te):,}", failures)
    check(df.label.nunique() == EXPECTED["n_classes"], f"classes = {df.label.nunique()}", failures)
    check(tr.user_id.nunique() == EXPECTED["users_train"], f"sujeitos treino = {tr.user_id.nunique():,}", failures)
    check(te.user_id.nunique() == EXPECTED["users_test"], f"sujeitos teste = {te.user_id.nunique():,}", failures)
    check(not (set(tr.user_id) & set(te.user_id)), "treino ∩ teste (sujeitos) = ∅", failures)
    check(not df.image_path.duplicated().any(), "image_path sem duplicatas", failures)
    missing = sum(not Path(p).exists() for p in df.image_path)
    check(missing == 0, f"imagens ausentes em disco = {missing}", failures)

    # --- version-specific -------------------------------------------------
    if v >= 2:
        v1 = pd.read_csv(manifest_path(1), dtype=str, keep_default_na=False)
        check("inner_split" in df.columns, "coluna inner_split presente", failures)
        check(df[base].equals(v1[base]), "colunas base idênticas à v1, linha a linha", failures)
        check(df.loc[df.split == "test", base].reset_index(drop=True)
              .equals(v1.loc[v1.split == "test", base].reset_index(drop=True)),
              "split de teste inalterado linha a linha vs v1", failures)

        fit = df[df.inner_split == "fit"]
        val = df[df.inner_split == "val"]
        check((df.loc[df.split == "test", "inner_split"] == "test").all(),
              "linhas de teste têm inner_split = test", failures)
        check(df.loc[df.split == "train", "inner_split"].isin(["fit", "val"]).all(),
              "linhas de treino têm inner_split ∈ {fit, val}", failures)
        check(not (set(fit.user_id) & set(val.user_id)), "fit ∩ val (sujeitos) = ∅", failures)
        check(not (set(val.user_id) & set(te.user_id)), "val ∩ teste (sujeitos) = ∅", failures)
        vfrac = len(val) / len(tr)
        check(0.08 <= vfrac <= 0.12, f"val = {len(val):,} imgs ({100 * vfrac:.2f}% do treino, "
              f"{val.user_id.nunique():,} sujeitos)", failures)
        check(len(fit) + len(val) == len(tr), f"fit = {len(fit):,} imgs ({fit.user_id.nunique():,} sujeitos)", failures)
        check(val.label.nunique() == EXPECTED["n_classes"], "18 classes presentes em val", failures)
        per = val.label.value_counts() / tr.label.value_counts()
        check(per.max() - per.min() < 0.03, f"val por classe {100 * per.min():.2f}%..{100 * per.max():.2f}% (amplitude < 3 p.p.)", failures)

    # --- registry ---------------------------------------------------------
    actual = sha256(path)
    reg = registered_sha(v)
    check(reg is not None, f"v{v} registrado em {REGISTRY}", failures)
    if reg is not None:
        check(reg == actual, f"sha256 registrado == sha256 em disco ({actual[:16]}...)", failures)

    print(f"\n{'OK' if not failures else 'FALHOU'}: {len(failures)} violação(ões)")
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
