#!/usr/bin/env bash
# Serial, resumable training queue (the GPU is serial: never two runs at once).
#
#   scripts/queue.sh eixo1            # 2 archs x seeds 0,1,2 (seed 0 first for both)
#   scripts/queue.sh eixo1 0          # only seed 0 of both archs
#
# A run is skipped when its metrics.json exists (test already evaluated); an
# interrupted run is resumed from last.pt. Each run logs to runs/<name>/train.log;
# the queue itself logs to runs/queue_<axis>.log.
set -uo pipefail
cd "$(dirname "$0")/.."

PY="${PY:-/home/ezenere/miniconda3/envs/tcc/bin/python}"
AXIS="${1:-eixo1}"
SEEDS="${2:-0 1 2}"
export PYTHONPATH=src
export TQDM_MININTERVAL=60   # keep train.log small
QLOG="runs/queue_${AXIS}.log"
mkdir -p runs

log() { printf '%s  %s\n' "$(date '+%F %T')" "$*" | tee -a "$QLOG"; }

run_one() {                # run_one <config> <seed>
  local cfg="$1" seed="$2" arch name
  arch="$(grep -E '^\s*arch:' "$cfg" | awk '{print $2}')"
  name="${AXIS}_${arch}_s${seed}"
  if [ -f "runs/$name/metrics.json" ]; then log "skip  $name (metrics.json existe)"; return 0; fi
  mkdir -p "runs/$name"
  log "start $name"
  local t0=$SECONDS
  "$PY" -u src/train.py --config "$cfg" --seed "$seed" --run-name "$name" --resume --eval-test \
      >> "runs/$name/train.log" 2>&1
  local rc=$?
  log "end   $name rc=$rc  $(( (SECONDS - t0) / 60 )) min"
  return $rc
}

log "=== fila $AXIS seeds [$SEEDS] ==="
for seed in $SEEDS; do
  for cfg in configs/train_resnet50.yaml configs/train_densenet121.yaml; do
    run_one "$cfg" "$seed" || { log "retry $cfg s$seed"; run_one "$cfg" "$seed" || log "FALHOU $cfg s$seed"; }
  done
done
log "=== fila $AXIS concluida ==="
