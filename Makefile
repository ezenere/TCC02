# Alvos idempotentes: cada fila pula o que já tem metrics.json. `make help` lista tudo.
PY ?= /home/ezenere/miniconda3/envs/tcc/bin/python
export PYTHONPATH := src

.PHONY: relatorio help test smoke verify audit eixo1 eixo2 eixo2b trt eixo3 eixo4 results figures latency-cpu latency-gpu freeze

help:            ## lista os alvos
	@grep -E '^[a-z0-9-]+:.*##' $(MAKEFILE_LIST) | awk -F':.*## ' '{printf "  %-14s %s\n", $$1, $$2}'
test:            ## testes unitários (CPU) + smoke de treino (GPU)
	$(PY) -m pytest tests -q
smoke:           ## pipeline inteiro em miniatura (~3 min, GPU)
	scripts/smoke.sh
verify:          ## invariantes dos manifestos v1/v2/v3 e determinismo
	for v in 1 2 3; do $(PY) scripts/verify_manifest.py --version $$v; done
	PY=$(PY) scripts/verify_determinism.sh
audit:           ## auditoria de vazamento (CPU) — results/audit/
	$(PY) scripts/audit_leakage.py && $(PY) scripts/audit_multiaccount.py --no-eval
eixo1:           ## 2 arquiteturas x 3 seeds, 30 épocas (~10 h GPU)
	scripts/queue.sh eixo1
eixo2:           ## poda depois do treino: 5 níveis x 3 seeds (~8,5 h GPU) + int8 CPU
	scripts/queue_prune.sh "0 1 2" "0.5 0.7 0.9 0.95 0.98"
	for a in resnet50 densenet121; do for s in 0 1 2; do $(PY) src/compress/quantize_cpu.py --checkpoint runs/eixo1_$${a}_s$${s}/checkpoints/best.pt --threads 8; done; done
eixo2b:          ## poda antes do treino, seed 0 (~16 h GPU)
	scripts/queue_prune_first.sh "0" "0.9 0.95 0.98 0.7 0.5"
trt:             ## ONNX -> Q/DQ -> engines TensorRT -> teste (GPU ociosa)
	scripts/queue_trt.sh "$$(ls -d runs/eixo1_*_s[012])" "fp32 fp16 int8"
latency-cpu:     ## latência CPU de todas as células (máquina em repouso)
	scripts/measure_cpu_latency.sh 0
latency-gpu:     ## latência GPU — rodar em TTY com a sessão gráfica fechada
	scripts/measure_gpu_latency.sh 0
eixo3:           ## tabela cruzada, Pareto e decisão
	$(PY) results/eixo3/make_eixo3.py --decide
eixo4:           ## frações x seeds na configuração vencedora (ver results/eixo3/decision.md)
	$(PY) src/reduce_data.py --config configs/train_resnet50.yaml --seeds 0 1 2
results:         ## regenera todos os CSVs, tabelas de README e figuras a partir de runs/
	$(PY) results/eixo1/make_eixo1.py
	$(PY) results/eixo2/make_pruning.py
	$(PY) results/eixo2/make_quant.py
	-$(PY) results/eixo2/make_prune_compare.py
	$(PY) results/eixo3/make_eixo3.py --decide
	-$(PY) results/eixo4/make_eixo4.py --powerlaw
	$(PY) results/analise/make_erros.py
figures: results ## idem (as figuras saem dos mesmos scripts)
	$(PY) scripts/plot_curves.py --prefix eixo1
relatorio:       ## relatório em linguagem simples: figuras + PDF (docs/relatorio/)
	$(PY) docs/relatorio/make_figuras.py
	$(PY) docs/relatorio/make_tabela.py
	$(PY) scripts/md2pdf.py docs/relatorio/RELATORIO.md
freeze:          ## regenera requirements.txt a partir do env
	scripts/freeze.sh
