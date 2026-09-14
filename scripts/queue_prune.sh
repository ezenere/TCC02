#!/usr/bin/env bash
# Serial, resumable pruning sweep: for each seed x arch x sparsity, prune the
# axis-1 best.pt of the SAME seed and fine-tune (configs/prune_<arch>.yaml).
#
#   scripts/queue_prune.sh "0"     "0.5 0.7 0.9"        # seed 0, three levels
#   scripts/queue_prune.sh "1 2"   "0.5 0.7 0.9"        # remaining seeds
#   scripts/queue_prune.sh "0 1 2" "0.95 0.98"          # extension, if triggered
set -uo pipefail
cd "$(dirname "$0")/.."
PY="${PY:-/home/ezenere/miniconda3/envs/tcc/bin/python}"
SEEDS="${1:-0}"; LEVELS="${2:-0.5 0.7 0.9}"
export PYTHONPATH=src TQDM_MININTERVAL=60
QLOG="runs/queue_eixo2_prune.log"; mkdir -p runs
log() { printf '%s  %s\n' "$(date '+%F %T')" "$*" | tee -a "$QLOG"; }

run_one() {   # run_one <arch> <seed> <sparsity>
  local arch="$1" seed="$2" sp="$3"
  local pct; pct="$(python3 -c "print(f'{round(100*$sp):02d}')")"
  local name="eixo2_prune_${arch}_p${pct}_s${seed}"
  local init="runs/eixo1_${arch}_s${seed}/checkpoints/best.pt"
  [ -f "$init" ] || { log "FALHOU $name: $init ausente"; return 1; }
  if [ -f "runs/$name/metrics.json" ]; then log "skip  $name"; return 0; fi
  mkdir -p "runs/$name"; log "start $name"; local t0=$SECONDS
  "$PY" -u src/train.py --config "configs/prune_${arch}.yaml" --seed "$seed" --sparsity "$sp" \
        --init-from "$init" --run-name "$name" --resume --eval-test >> "runs/$name/train.log" 2>&1
  local rc=$?; log "end   $name rc=$rc  $(( (SECONDS - t0) / 60 )) min"
  [ $rc -eq 0 ] && "$PY" src/measure/cost.py --checkpoint "runs/$name/checkpoints/best.pt" >> "runs/$name/train.log" 2>&1
  return $rc
}

log "=== poda: seeds [$SEEDS] niveis [$LEVELS] ==="
for seed in $SEEDS; do for sp in $LEVELS; do for arch in resnet50 densenet121; do
  run_one "$arch" "$seed" "$sp" || { log "retry $arch s$seed p$sp"; run_one "$arch" "$seed" "$sp" || log "FALHOU $arch s$seed p$sp"; }
done; done; done
log "=== poda concluida: seeds [$SEEDS] niveis [$LEVELS] ==="
