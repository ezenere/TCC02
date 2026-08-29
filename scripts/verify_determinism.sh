#!/usr/bin/env bash
# Determinism criterion for the subset: two independent processes, with
# different PYTHONHASHSEED, must produce a byte-identical manifest. A differing
# hash means some iteration order leaked in from a set/dict/filesystem listing.
set -euo pipefail

PY="${PY:-python}"
CFG="${1:-configs/preprocess.yaml}"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

for seed in 0 12345; do
  PYTHONHASHSEED="$seed" "$PY" src/preprocess.py --config "$CFG" \
    --plan-only --manifest-out "$TMP/manifest_$seed.csv" >"$TMP/log_$seed.txt" 2>&1
done

A="$TMP/manifest_0.csv"; B="$TMP/manifest_12345.csv"
HA="$(sha256sum "$A" | cut -d' ' -f1)"
HB="$(sha256sum "$B" | cut -d' ' -f1)"

echo "PYTHONHASHSEED=0     : $HA"
echo "PYTHONHASHSEED=12345 : $HB"

if ! cmp -s "$A" "$B"; then
  echo "FAIL: manifests differ ($(diff <(sort "$A") <(sort "$B") | grep -c '^[<>]') linhas)" >&2
  exit 1
fi

if [ -f data/processed/manifest.csv ]; then
  HC="$(sha256sum data/processed/manifest.csv | cut -d' ' -f1)"
  echo "manifesto em disco   : $HC"
  [ "$HC" = "$HA" ] || { echo "FAIL: on-disk manifest diverges from the plan" >&2; exit 1; }
fi

echo "OK: manifesto byte-identico entre execucoes independentes"
