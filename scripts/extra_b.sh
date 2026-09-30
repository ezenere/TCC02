#!/usr/bin/env bash
# Extra B — does compression hurt more when there is less data?
# Prune 90% + fine-tune (5 ep.) the axis-4 model trained on 25% of the data (seed 0),
# on the SAME 25% fraction, then int8 on CPU and TensorRT (calibration restricted to
# that fraction). Compare against the 100%-data cells of axis 3.
set -uo pipefail; cd "$(dirname "$0")/.."
PY="${PY:-/home/ezenere/miniconda3/envs/tcc/bin/python}"; export PYTHONPATH=src TQDM_MININTERVAL=60
SRC=runs/eixo4_resnet50_f025_s0; NAME=extraB_resnet50_f025_p90_s0; LOG=runs/queue_extra_b.log
log() { printf '%s  %s\n' "$(date '+%F %T')" "$*" | tee -a "$LOG"; }
log "=== extra B ==="
if [ ! -f "runs/$NAME/metrics.json" ]; then
  scripts/gpu_wait.sh 8000 3 | tee -a "$LOG"; mkdir -p "runs/$NAME"; log "start $NAME"
  "$PY" -u src/train.py --config configs/prune_resnet50.yaml --seed 0 --sparsity 0.9 --init-from "$SRC/checkpoints/best.pt" \
        --manifest data/processed/manifest_v3.csv --frac-column frac_25_s0 --run-name "$NAME" --resume --eval-test >> "runs/$NAME/train.log" 2>&1
  log "end   $NAME rc=$?"
fi
"$PY" src/measure/cost.py --checkpoint "runs/$NAME/checkpoints/best.pt" >> "runs/$NAME/train.log" 2>&1
[ -f "runs/$NAME/metrics_int8_fbgemm.json" ] || systemd-run --user --scope -q -p MemoryMax=16G "$PY" src/compress/quantize_cpu.py --checkpoint "runs/$NAME/checkpoints/best.pt" --threads 8 >> "runs/$NAME/train.log" 2>&1
[ -f "$SRC/metrics_int8_fbgemm.json" ] || systemd-run --user --scope -q -p MemoryMax=16G "$PY" src/compress/quantize_cpu.py --checkpoint "$SRC/checkpoints/best.pt" --threads 8 >> "$SRC/train.log" 2>&1
scripts/gpu_wait.sh 6000 2 | tee -a "$LOG"
scripts/queue_trt.sh "runs/$NAME" "int8" >> "$LOG" 2>&1
log "=== extra B concluido ==="
