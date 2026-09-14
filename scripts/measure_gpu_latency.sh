#!/usr/bin/env bash
# GPU latency of every cell. RUN FROM A TTY WITH THE GRAPHICAL SESSION CLOSED,
# so the desktop compositor does not contaminate the distribution.
#   NOTE="sessão gráfica fechada (TTY)" scripts/measure_gpu_latency.sh [seed]
set -uo pipefail; cd "$(dirname "$0")/.."
PY="${PY:-/home/ezenere/miniconda3/envs/tcc/bin/python}"; export PYTHONPATH=src
SEED="${1:-0}"; NOTE="${NOTE:-sessão gráfica: NÃO DECLARADA}"
nvidia-smi --query-gpu=name,driver_version,clocks.sm,clocks.mem,temperature.gpu --format=csv
for arch in resnet50 densenet121; do
  run="runs/eixo1_${arch}_s${SEED}"
  "$PY" src/measure/latency.py --artifact "$run/checkpoints/best.pt" --kind eager --device cuda --precision fp32 --note "$NOTE"
  "$PY" src/measure/latency.py --artifact "$run/checkpoints/best.pt" --kind eager --device cuda --precision fp16 --note "$NOTE"
  for prec in fp32 fp16 int8; do
    eng="$run/trt/model_${prec}.engine"
    [ -f "$eng" ] && "$PY" src/measure/latency.py --artifact "$eng" --kind trt --device cuda --precision $prec --note "$NOTE"
  done
done
