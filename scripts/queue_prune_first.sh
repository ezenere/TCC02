#!/usr/bin/env bash
# Eixo 2b — prune the ImageNet weights first, then the full 30-epoch training
# with fixed masks (configs/prune_first_<arch>.yaml). Serial and resumable.
#   scripts/queue_prune_first.sh "0" "0.9 0.95 0.98 0.7 0.5"
set -uo pipefail
cd "$(dirname "$0")/.."
PY="${PY:-/home/ezenere/miniconda3/envs/tcc/bin/python}"
SEEDS="${1:-0}"; LEVELS="${2:-0.9 0.95 0.98 0.7 0.5}"
export PYTHONPATH=src TQDM_MININTERVAL=60
QLOG="runs/queue_eixo2b_prune_first.log"; mkdir -p runs
log() { printf '%s  %s\n' "$(date '+%F %T')" "$*" | tee -a "$QLOG"; }

run_one() {   # run_one <arch> <seed> <sparsity>
  local arch="$1" seed="$2" sp="$3"
  local pct; pct="$(python3 -c "print(f'{round(100*$sp):02d}')")"
  local name="eixo2b_prunefirst_${arch}_p${pct}_s${seed}"
  if [ -f "runs/$name/metrics.json" ]; then log "skip  $name"; return 0; fi
  scripts/gpu_wait.sh 8000 3 | tee -a "$QLOG"          # the GPU is shared: wait until it is free
  mkdir -p "runs/$name"; log "start $name"; local t0=$SECONDS
  "$PY" -u src/train.py --config "configs/prune_first_${arch}.yaml" --seed "$seed" --sparsity "$sp" \
        --run-name "$name" --resume --eval-test >> "runs/$name/train.log" 2>&1
  local rc=$?; log "end   $name rc=$rc  $(( (SECONDS - t0) / 60 )) min"
  [ $rc -eq 0 ] && "$PY" src/measure/cost.py --checkpoint "runs/$name/checkpoints/best.pt" >> "runs/$name/train.log" 2>&1
  return $rc
}

log "=== poda-antes: seeds [$SEEDS] niveis [$LEVELS] ==="
for seed in $SEEDS; do for sp in $LEVELS; do for arch in resnet50 densenet121; do
  run_one "$arch" "$seed" "$sp" || { log "retry $arch s$seed p$sp"; run_one "$arch" "$seed" "$sp" || log "FALHOU $arch s$seed p$sp"; }
done; done; done
log "=== poda-antes concluida: seeds [$SEEDS] niveis [$LEVELS] ==="
