# Documentação de referência para o artigo: índice

Reconhecimento de gestos de mão (HaGRIDv2, 18 classes): ResNet-50 vs DenseNet-121, poda, quantização int8, benchmark e redução de dados anotados. Esta pasta reúne, em cinco documentos, tudo o que o artigo precisa citar: cronologia, protocolo, resultados por eixo com erros por seed, validade e limitações. Todo número tem a fonte (CSV ou JSON) indicada logo abaixo da tabela; nada é digitado à mão fora dos CSVs gerados por `make results`. Público: autor e orientador.

| documento | conteúdo |
|---|---|
| [01_LINHA_DO_TEMPO.md](01_LINHA_DO_TEMPO.md) | cronologia dia a dia de 28/08 a 17/10/2026: o que foi implementado, quantos runs e horas de GPU, decisões do orientador (14/09: 8 decisões; 19/09: auditoria, poda antes do treino, app), incidentes e correções |
| [02_DADOS_E_PROTOCOLO.md](02_DADOS_E_PROTOCOLO.md) | dataset e licença, subamostragem de 50% por sujeito (algoritmo, seeds, determinismo), splits (70/30, val interno, máscaras aninhadas), crop, contagens por classe, augmentation, receita de treino, métricas, seeds, hardware e versões |
| [03_RESULTADOS_EIXO1_EIXO2.md](03_RESULTADOS_EIXO1_EIXO2.md) | eixo 1 (baseline, custo estático); eixo 2: poda em 5 níveis × 3 seeds, int8 CPU e TensorRT por seed, FP32/FP16, poda + int8; eixo 2b (poda antes) e 2c (LR rewinding); varredura de calibradores e a troca entropia → percentil 99,99; achados |
| [04_RESULTADOS_EIXO3_EIXO4.md](04_RESULTADOS_EIXO3_EIXO4.md) | tabela cruzada do eixo 3 com todas as latências (CPU 1/16 thr, GPU TensorRT lotes 1 e 32), critério de decisão e aplicação, vencedora; eixo 4 por fração × seed (FP32 e int8), lei de potência, early stopping descartado, extra B |
| [05_VALIDADE_E_LIMITACOES.md](05_VALIDADE_E_LIMITACOES.md) | auditoria de vazamento (todas as verificações), contas duplicadas e pior caso, holdout externo por seed, controle de rótulos embaralhados, análise de erros, limitações, lições de método |

## Achados principais (resumo)

1. **As duas arquiteturas empatam em erro.** ResNet-50 0,207 ± 0,009% vs DenseNet-121 0,197 ± 0,005% no teste (83.613 imagens, 6.044 sujeitos nunca vistos); a diferença de 0,010 p.p. fica abaixo de 2× o desvio entre seeds. A DenseNet tem 3,4× menos parâmetros e 30% menos MACs, mas é mais lenta em todos os runtimes medidos.
2. **O piso de ~0,2% é em boa parte ruído de rótulo:** 117 imagens são erradas pelas duas redes, 109 delas com a mesma predição; `stop` ↔ `palm` responde por 36% dos erros.
3. **90% dos pesos são dispensáveis** com fine-tuning de 5 épocas (razão de erro 1,10× / 1,03×); o joelho (> 1,5×) está em 98% (1,90× / 1,65×). A poda não-estruturada reduz o artefato comprimido (21% a 90%) mas **não a latência**.
4. **Podar antes ou depois do treino: o orçamento de retreino decide, não a ordem.** A 98%, podar o modelo treinado e retreinar 30 épocas dá 1,12 ± 0,02× (ResNet) e 1,23 ± 0,09× (DenseNet); podar os pesos ImageNet antes do treino fica em 1,40× / 1,54×.
5. **A int8 é gratuita para a ResNet-50** (1,04× CPU, 1,01× TensorRT) e custa à DenseNet-121 (1,24× CPU, 1,07× TensorRT); a engine int8 da DenseNet é mais lenta que a FP16 (399 de 1.497 camadas ficam em float).
6. **A calibração da int8 é um hiperparâmetro:** com entropia, o mesmo modelo foi de 194 a 2.596 erros conforme a amostra; percentil 99,99, escolhido em `val`, varia de 62 a 64 erros onde o denso faz 60. 28 células foram refeitas.
7. **Configuração vencedora do benchmark: ResNet-50 int8 TensorRT**, razão de erro 1,01×, 0,80 ms por imagem (lote 1), 23,9 MiB (latência GPU preliminar, medida com a sessão gráfica aberta; critério fixado antes dos resultados).
8. **Redução de dados:** 75% → 1,05×, 50% → 1,28×, 25% → 1,41×, 10% → 2,33×, 5% → 3,13× (ResNet-50, 3 seeds, 30 épocas fixas); erro ∝ N^−0,38 (R² 0,98); a int8 acompanha ponto a ponto. Mesmo com 8,8 mil imagens a acurácia é 99,36%.
9. **Compressão custa mais com menos dados** (extra B, uma seed): com 25% dos dados a poda de 90% custa 1,69× (vs 1,07×) e poda + int8 2,03× (vs 1,14×).
10. **O número de 99,8% se sustenta:** pipeline sem vazamento (sujeitos disjuntos, manifesto = fonte em 100% das linhas, rótulos embaralhados → 4,9%); o HaGRID tem contas duplicadas que explicam no máximo 0,07 p.p. (pior caso 99,72%); num holdout externo de 278.702 imagens de 19.042 sujeitos nunca usados, o erro é 0,235% / 0,216%.

## Artefatos públicos

Repositório: **github.com/ezenere/TCC02** (código, configs, logs por run, métricas, figuras, documentação). Checkpoints fora do git, publicados como releases:

| release | tag em | conteúdo |
|---|---|---|
| `ResNet` | `40f79c3` (29/08) | baseline de evidência (ResNet-50, 5 épocas, manifesto v1): `best.pt`, `last.pt` (180 MB cada) |
| `v0.2-eixo1` | `bcafa82` (20/09) | eixo 1: 6 × `best.pt` + 6 × `last.pt`, nomeados `<run>__best.pt` / `<run>__last.pt` (1,4 GiB) |
| `v0.3-eixo2` | `edf3eed` (20/09) | eixo 2: 30 modelos podados (`eixo2_prune_<arch>_p<NN>_s<k>__best.pt`, máscaras incluídas) + 10 artefatos int8 de CPU (`<run>__model_int8_fbgemm.pt`), 4,0 GiB. Engines TensorRT não publicadas (específicas da GPU e da versão; `make trt` reconstrói) |
| `v0.5-eixo4` | (30/09) | eixo 4: 18 × `best.pt` da ResNet-50 (`eixo4_resnet50_f<FFF>_s<k>__best.pt`, frações 100/75/50/25/10/5 × seeds 0–2), 3,2 GiB |

Não publicados como release (ficam em `runs/` local): eixo 2b (16 runs), eixo 2c (12 runs), extra B, controle de rótulos embaralhados. Seus `metrics.json`, `run_meta.json` e `metrics.csv` estão versionados no repositório.

Demo: `app/webcam_demo.py` (PyTorch, int8 TorchScript, ONNX Runtime ou TensorRT; `app/README.md`).

## Scripts que regeneram cada tabela

| tabela / resultado | script | saída |
|---|---|---|
| eixo 1 (baseline, por run, custo) | `results/eixo1/make_eixo1.py`; `src/measure/cost.py` | `results/eixo1/eixo1_runs.csv`, `eixo1_summary.csv`, `runs/*/cost.json` |
| curvas de treino | `scripts/plot_curves.py --prefix eixo1` | `results/eixo1/figures/` |
| poda (5 níveis × 3 seeds) | `results/eixo2/make_pruning.py` | `results/eixo2/pruning_runs.csv`, `pruning_summary.csv`, `figures/pruning_error_ratio` |
| poda antes × depois × LR rewinding | `results/eixo2/make_prune_compare.py` | `results/eixo2/prune_compare_runs.csv`, `prune_compare_summary.csv`, `figures/prune_compare` |
| quantização por backend | `results/eixo2/make_quant.py` | `results/eixo2/quant_runs.csv`, `quant_summary.csv` |
| varredura de calibradores | `scripts/calib_sweep.sh` | `results/eixo2/calib_sweep.csv` |
| tabela cruzada, Pareto e decisão | `results/eixo3/make_eixo3.py --decide` | `results/eixo3/eixo3_table.csv`, `eixo3_summary.csv`, `decision.md`, `figures/pareto_*` |
| latência | `scripts/measure_cpu_latency.sh`, `scripts/measure_gpu_latency.sh` (`src/measure/latency.py`) | `runs/*/latency_*.json` |
| eixo 4 (frações × seeds, lei de potência) | `results/eixo4/make_eixo4.py --powerlaw` | `results/eixo4/eixo4_runs.csv`, `eixo4_summary.csv`, `eixo4_powerlaw.json`, `figures/` |
| extra B | `results/analise/make_interacao.py` | `results/analise/interacao.csv`, `interacao.{pdf,png}` |
| análise de erros | `results/analise/make_erros.py` (`src/eval.py --save-preds`) | `results/analise/erros_overlap.json`, `confusion_pairs.csv`, `erros_*.png` |
| auditoria | `scripts/audit_leakage.py`, `scripts/audit_multiaccount.py`, `scripts/make_holdout.py` | `results/audit/leakage_audit.json`, `near_duplicates.csv`, `multiaccount_summary.csv`, `clean_vs_suspect.csv`, `holdout_summary.csv` |
| estatísticas do subconjunto | `src/validate_raw.py`, `src/preprocess.py` | `results/raw_validation.csv`, `results/subset_stats.csv` |
| manifestos | `src/preprocess.py`, `src/make_inner_split.py`, `src/make_masks.py`; `scripts/verify_manifest.py`, `scripts/verify_determinism.sh` | `data/processed/manifest{,_v2,_v3}.csv`, `MANIFESTS.md` |
| tudo de uma vez | `make results` (e `make figures`) | todos os CSVs, tabelas de README e figuras |

Documentos complementares no repositório: `RESULTS.md` (sumário executivo), `docs/METODOLOGIA.md` (execução por eixo), `STATUS.md` (diário), `docs/RUNBOOK.md` (como rodar e retomar), `docs/MAPA_CODIGO.md` (que arquivo faz o quê), `docs/relatorio/RELATORIO.md` (relatório em linguagem simples com o anexo de todos os acertos medidos).
