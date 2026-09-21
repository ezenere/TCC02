#!/usr/bin/env bash
# TensorRT cells for a list of run dirs: ONNX export -> Q/DQ (percentile 99.99, memory cap)
# -> engines -> test evaluation. GPU-serial; run with the GPU otherwise idle.
#   scripts/queue_trt.sh "runs/eixo1_resnet50_s1 runs/eixo1_densenet121_s1" "fp32 fp16 int8"
set -uo pipefail; cd "$(dirname "$0")/.."
PY="${PY:-/home/ezenere/miniconda3/envs/tcc/bin/python}"; export PYTHONPATH=src
RUNS="$1"; PRECS="${2:-fp16 int8}"; QLOG=runs/queue_trt.log
log() { printf '%s  %s\n' "$(date '+%F %T')" "$*" | tee -a "$QLOG"; }
for run in $RUNS; do
  log "start $run [$PRECS]"
  [ -f "$run/model.onnx" ] || "$PY" src/compress/export_onnx.py --checkpoint "$run/checkpoints/best.pt" >> "$run/trt.log" 2>&1 || { log "FALHOU export $run"; continue; }
  if [[ " $PRECS " == *" int8 "* ]] && [ ! -f "$run/model_qdq_int8.onnx" ]; then
    systemd-run --user --scope -q -p MemoryMax=16G "$PY" src/compress/quantize_onnx_qdq.py --run "$run" >> "$run/trt.log" 2>&1 || { log "FALHOU qdq $run"; continue; }
  fi
  "$PY" src/compress/trt_build.py --run "$run" --precision $PRECS >> "$run/trt.log" 2>&1 || { log "FALHOU build $run"; continue; }
  for p in $PRECS; do
    [ -f "$run/metrics_trt_$p.json" ] && continue
    "$PY" src/compress/trt_eval.py --run "$run" --precision "$p" >> "$run/trt.log" 2>&1 || log "FALHOU eval $run $p"
  done
  log "end   $run"
done
log "=== fila trt concluida ==="
