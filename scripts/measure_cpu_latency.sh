#!/usr/bin/env bash
# CPU latency of every cell (FP32 eager, int8 TorchScript, ONNX FP32), 1 and 16
# threads, batch 1 and 32. Run with the machine otherwise idle (no training).
#   scripts/measure_cpu_latency.sh [seed]        # default seed 0
set -uo pipefail; cd "$(dirname "$0")/.."
PY="${PY:-/home/ezenere/miniconda3/envs/tcc/bin/python}"; export PYTHONPATH=src
SEED="${1:-0}"; NOTE="${NOTE:-sistema em repouso}"
for arch in resnet50 densenet121; do
  run="runs/eixo1_${arch}_s${SEED}"
  for t in 1 16; do
    "$PY" src/measure/latency.py --artifact "$run/checkpoints/best.pt" --kind eager --device cpu --threads $t --note "$NOTE"
    [ -f "$run/model_int8_fbgemm.pt" ] && "$PY" src/measure/latency.py --artifact "$run/model_int8_fbgemm.pt" --kind torchscript --device cpu --precision int8 --threads $t --note "$NOTE"
    [ -f "$run/model.onnx" ] && "$PY" src/measure/latency.py --artifact "$run/model.onnx" --kind onnx --device cpu --threads $t --note "$NOTE"
  done
  # pruned cells: the dense kernels do not exploit sparsity; measured once to document it
  for p in runs/eixo2_prune_${arch}_p*_s${SEED}; do
    [ -f "$p/checkpoints/best.pt" ] && "$PY" src/measure/latency.py --artifact "$p/checkpoints/best.pt" --kind eager --device cpu --threads 16 --batch 1 --iters 100 --note "$NOTE"
  done
done
