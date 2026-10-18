#!/usr/bin/env bash
# Medição FINAL de latência em GPU. Rodar em um TTY com a sessão gráfica PARADA.
#
#   Ctrl+Alt+F3  →  login  →
#   cd ~/Documents/Faculdade/TCC && sudo systemctl stop sddm && scripts/latency_tty.sh
#   ... ao terminar:  sudo systemctl start sddm   (ou Ctrl+Alt+F1/F2 se preferir só voltar)
#
# O que faz: (1) recusa rodar se houver sessão gráfica ou outro processo na GPU;
# (2) reconstrói as engines TensorRT das células medidas (seed 0: baseline fp32/fp16/int8
# das duas redes + poda 90/95% + int8) com a GPU ociosa — a seleção de kernels do TensorRT é
# por cronometragem, então engines construídas com o desktop aberto podem ser piores;
# (3) mede tudo (PyTorch eager fp32/fp16, TensorRT fp32/fp16/int8, lotes 1 e 32) com a
# nota "sessão gráfica fechada (TTY)" gravada em cada JSON. Cada célula é medida em 5 rodadas
# (mediana), porque a latência de lote 1 alterna entre dois patamares ~15% apart entre medições
# (results/eixo3/build_variance.md). ~45 min. Passo OPCIONAL: a medição de 18/10 com 5 rodadas e sessão
# aberta não mostrou diferença sistemática em relação ao TTY.
set -uo pipefail; cd "$(dirname "$0")/.."
PY="${PY:-/home/ezenere/miniconda3/envs/tcc/bin/python}"; export PYTHONPATH=src
LOG=runs/latency_gpu_tty.log; log() { printf '%s  %s\n' "$(date '+%F %T')" "$*" | tee -a "$LOG"; }

# ---- 1. pré-condições -------------------------------------------------------
if [ -n "${DISPLAY:-}" ] || [ -n "${WAYLAND_DISPLAY:-}" ]; then
  echo "ERRO: isto parece uma sessão gráfica (DISPLAY/WAYLAND_DISPLAY definidos). Rode em um TTY."; exit 1; fi
if systemctl is-active --quiet sddm 2>/dev/null; then
  echo "ERRO: o sddm (sessão gráfica) está ativo. Pare com: sudo systemctl stop sddm"; exit 1; fi
busy="$(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader | awk -F', ' '$2+0 > 100')"
if [ -n "$busy" ]; then echo "ERRO: outro processo usa a GPU:"; echo "$busy"; exit 1; fi
log "=== latência GPU em TTY — início ==="
nvidia-smi --query-gpu=name,driver_version,clocks.sm,clocks.mem,temperature.gpu,memory.used --format=csv | tee -a "$LOG"

# ---- 2. engines com a GPU ociosa -------------------------------------------
NOTE="sessão gráfica fechada (TTY); 5 rodadas (mediana)"
for run in runs/eixo1_resnet50_s0 runs/eixo1_densenet121_s0; do
  log "engines $run (fp32 fp16 int8, --force)"
  "$PY" src/compress/trt_build.py --run "$run" --precision fp32 fp16 int8 --force >> "$LOG" 2>&1 || { log "FALHOU build $run"; exit 1; }
done
for run in runs/eixo2_prune_resnet50_p90_s0 runs/eixo2_prune_resnet50_p95_s0 runs/eixo2_prune_densenet121_p90_s0 runs/eixo2_prune_densenet121_p95_s0; do
  log "engine int8 $run (--force)"
  "$PY" src/compress/trt_build.py --run "$run" --precision int8 --force >> "$LOG" 2>&1 || { log "FALHOU build $run"; exit 1; }
done
# sanidade: as engines novas devem reproduzir a acurácia das antigas (teste completo, ~1 min cada)
for run in runs/eixo1_resnet50_s0 runs/eixo1_densenet121_s0; do for p in fp32 fp16 int8; do
  "$PY" src/compress/trt_eval.py --run "$run" --precision $p >> "$LOG" 2>&1 || log "AVISO: eval $run $p falhou"
done; done

# ---- 3. medições ------------------------------------------------------------
find runs -maxdepth 3 -name 'latency_*cuda*.json' -delete
NOTE="$NOTE" scripts/measure_gpu_latency.sh 0 2>&1 | grep -v "^\[" | tee -a "$LOG"
for run in runs/eixo2_prune_resnet50_p90_s0 runs/eixo2_prune_resnet50_p95_s0 runs/eixo2_prune_densenet121_p90_s0 runs/eixo2_prune_densenet121_p95_s0; do
  "$PY" src/measure/latency.py --artifact "$run/trt/model_int8.engine" --kind trt --device cuda --precision int8 --note "$NOTE" 2>&1 | grep -v "^\[" | tee -a "$LOG"
  "$PY" src/measure/latency.py --artifact "$run/checkpoints/best.pt" --kind eager --device cuda --precision fp32 --batch 1 --note "$NOTE" 2>&1 | grep -v "^\[" | tee -a "$LOG"
done
n="$(find runs -maxdepth 3 -name 'latency_*cuda*.json' | wc -l)"
log "=== concluído: $n medições gravadas. Agora: sudo systemctl start sddm ==="
