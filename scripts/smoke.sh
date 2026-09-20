#!/usr/bin/env bash
# End-to-end smoke of the whole pipeline on a few hundred images (~3 min, needs the GPU
# and data/processed/): train 2 epochs -> prune 50% + 1 epoch -> int8 CPU -> cost ->
# latency -> ONNX export. Everything lands in runs/_smoke/ and is removed at the end.
set -euo pipefail
cd "$(dirname "$0")/.."
PY="${PY:-/home/ezenere/miniconda3/envs/tcc/bin/python}"; export PYTHONPATH=src TQDM_MININTERVAL=60
R=runs/_smoke; rm -rf "$R"
step() { printf '\n== %s ==\n' "$*"; }
# show only the lines that match, never fail on "no match"; python failures still propagate (pipefail)
show() { tr '\r' '\n' | { grep -a -E "$1" || true; } | sed -E 's/^.*(epoca [0-9]+\/)/\1/' | tail -n "${2:-5}"; }

step "1/7 manifestos"; for v in 1 2 3; do "$PY" scripts/verify_manifest.py --version $v | tail -1; done
step "2/7 treino (2 épocas, 500 imagens)"
"$PY" src/train.py --config configs/test_smoke.yaml --run-name smoke_dense --epochs 2 --limit-fit 500 --limit-val 200 2>&1 | show "epoca [0-9]+/"
step "3/7 poda 50% + fine-tuning (1 época)"
"$PY" src/train.py --config configs/test_smoke.yaml --run-name smoke_pruned --epochs 1 --limit-fit 500 --limit-val 200 \
      --init-from "$R/smoke_dense/checkpoints/best.pt" --sparsity 0.5 2>&1 | show "poda global|epoca [0-9]+/"
step "4/7 custo estático"
"$PY" src/measure/cost.py --checkpoint "$R/smoke_pruned/checkpoints/best.pt" | head -1
step "5/7 int8 CPU (fbgemm), 256 imagens de teste"
"$PY" src/compress/quantize_cpu.py --checkpoint "$R/smoke_dense/checkpoints/best.pt" --calib-n 128 --limit-test 256 --threads 8 2>&1 | show "int8/x86" 1
step "6/7 latência CPU (int8, 1 thread, 20 medidas)"
"$PY" src/measure/latency.py --artifact "$R/smoke_dense/model_int8_fbgemm.pt" --kind torchscript --device cpu --precision int8 --threads 1 --batch 1 --warmup 5 --iters 20 | head -1
step "7/7 export ONNX + paridade"
"$PY" src/compress/export_onnx.py --checkpoint "$R/smoke_dense/checkpoints/best.pt" --parity-n 32 2>&1 | show "paridade" 1
rm -rf "$R"
printf '\nSMOKE OK\n'
