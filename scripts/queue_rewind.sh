#!/usr/bin/env bash
# Eixo 2c — control for eixo 2b: prune the TRAINED model, then retrain with the full
# 30-epoch schedule (LR rewinding). Waits for a free GPU before every run; resumable.
#   scripts/queue_rewind.sh "0" "0.98 0.95"
set -uo pipefail
cd "$(dirname "$0")/.."
PY="${PY:-/home/ezenere/miniconda3/envs/tcc/bin/python}"
SEEDS="${1:-0}"; LEVELS="${2:-0.98 0.95}"
export PYTHONPATH=src TQDM_MININTERVAL=60
QLOG="runs/queue_eixo2c_rewind.log"; mkdir -p runs
log() { printf '%s  %s\n' "$(date '+%F %T')" "$*" | tee -a "$QLOG"; }
run_one() {
  local arch="$1" seed="$2" sp="$3" pct name init
  pct="$(python3 -c "print(f'{round(100*$sp):02d}')")"; name="eixo2c_rewind_${arch}_p${pct}_s${seed}"
  init="runs/eixo1_${arch}_s${seed}/checkpoints/best.pt"
  if [ -f "runs/$name/metrics.json" ]; then log "skip  $name"; return 0; fi
  scripts/gpu_wait.sh 8000 3 | tee -a "$QLOG"
  mkdir -p "runs/$name"; log "start $name"; local t0=$SECONDS
  "$PY" -u src/train.py --config "configs/prune_rewind_${arch}.yaml" --seed "$seed" --sparsity "$sp" --init-from "$init" \
        --run-name "$name" --resume --eval-test >> "runs/$name/train.log" 2>&1
  local rc=$?; log "end   $name rc=$rc  $(( (SECONDS - t0) / 60 )) min"
  [ $rc -eq 0 ] && "$PY" src/measure/cost.py --checkpoint "runs/$name/checkpoints/best.pt" >> "runs/$name/train.log" 2>&1
  return $rc
}
log "=== rewind: seeds [$SEEDS] niveis [$LEVELS] ==="
for seed in $SEEDS; do for sp in $LEVELS; do for arch in resnet50 densenet121; do
  for attempt in 1 2 3; do run_one "$arch" "$seed" "$sp" && break || log "retry($attempt) $arch s$seed p$sp"; done
done; done; done
log "=== rewind concluido ==="
