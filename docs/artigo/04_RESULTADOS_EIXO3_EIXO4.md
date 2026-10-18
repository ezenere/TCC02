# 04: Resultados: eixo 3 (benchmark) e eixo 4 (redução de dados anotados)

Convenções como no documento 03: teste de 83.613 imagens, razão de erro contra o baseline da mesma seed, média ± desvio de 3 seeds salvo indicação. Custo e latência medidos nos modelos da seed 0.

## 1. Eixo 3: Tabela cruzada

**Células.** {ResNet-50, DenseNet-121} × {baseline, poda 50/70/90/95/98%, int8 CPU (fbgemm), TensorRT FP32/FP16/INT8, poda@90/95 + int8 (CPU e TensorRT)}. A célula poda + int8 foi aprovada em 14/09 para responder se as técnicas compõem; aplicada em 90% e 95% (o último nível dentro do orçamento de 1,5×), seed 0 apenas.

**Métricas de custo** (`src/measure/cost.py`): parâmetros não-nulos (os zerados pela poda não contam), MACs a 224×224 (fvcore), tamanho em disco do artefato que seria implantado (`state_dict` FP32 para o baseline; o mesmo com gzip para os podados; TorchScript int8; engine TensorRT).

**Latência** (`src/measure/latency.py`): entrada sintética, 50 iterações de aquecimento e 300 medidas por lote; p50 (reportado), p95 e média; lotes 1 e 32. CPU: Ryzen 9 9950X, PyTorch eager FP32 e TorchScript int8 fbgemm no mesmo processo, 1 e 16 threads, máquina em repouso (medida em 20/09/2026). GPU: RTX 3080 Ti, PyTorch eager FP32/FP16 e engines TensorRT FP32/FP16/INT8; o baseline é medido também em TensorRT para que o ganho de int8 não se confunda com o ganho de runtime. Em GPU, cada célula é medida em **5 rodadas** independentes e o número reportado é a **mediana do p50** entre rodadas (`--rounds`; o JSON grava cada rodada e a dispersão). Modelos podados medidos uma vez em CPU e em GPU eager para documentar que kernels densos não exploram esparsidade.

**Histórico da medição em GPU.** A medição de 20/09 (sessão gráfica aberta, uma rodada) foi rotulada preliminar. Em 17/10 o autor mediu em TTY com a sessão gráfica parada e as engines reconstruídas com a GPU ociosa (`scripts/latency_tty.sh`, `8ade7b5`). Essa medição única inverteu a ordem de células que rodam o **mesmo grafo** (ResNet-50: int8 0,920 ms, poda 90% + int8 0,923, poda 95% + int8 0,797) e colocou a FP16 (0,827) à frente da int8. O experimento de 18/10 (`results/eixo3/build_variance.md`) mostrou que a latência de lote 1 alterna entre dois patamares ~15% apart (≈ 0,80 e ≈ 0,93 ms) para qualquer engine, com o clock SM estável: variância de medição, não da engine (a mesma engine reconstruída três vezes varia ~4%; as três engines de TTY, medidas em sequência, ficam em 0,923–0,930 ms). O harness ganhou as rodadas repetidas (`59f4132`) e **todas as células de GPU foram remedidas em 18/10** com 5 rodadas (`66621e1`; sessão gráfica aberta; entre TTY e sessão aberta não houve diferença sistemática: eager FP32 1,81 ms nas duas, TensorRT FP32 2,65 nas duas, DenseNet dentro de 1%). Os JSONs de TTY ficam arquivados como `*_tty1round.json`. Os números abaixo são os finais, de 18/10.

### 1.1 Qualidade e custo estático

| arquitetura | célula | seeds | razão de erro | erros (média) | não-nulos (M) | MACs (G) | artefato (MiB) |
|---|---|---|---|---|---|---|---|
| ResNet-50 | baseline | 3 | 1,00 | 173,3 | 23,54 | 4,11 | 90,1 |
| ResNet-50 | poda 50% | 3 | 0,97 ± 0,03 | 168,0 | 11,80 | 4,11 | 51,0 |
| ResNet-50 | poda 70% | 3 | 0,98 ± 0,02 | 169,7 | 7,10 | 4,11 | 34,8 |
| ResNet-50 | poda 90% | 3 | 1,10 ± 0,03 | 189,7 | 2,40 | 4,11 | 17,2 |
| ResNet-50 | poda 95% | 3 | 1,28 ± 0,07 | 221,7 | 1,23 | 4,11 | 12,1 |
| ResNet-50 | poda 98% | 3 | 1,90 ± 0,17 | 328,0 | 0,52 | 4,11 | 8,8 |
| ResNet-50 | int8 CPU (fbgemm) | 3 | 1,04 ± 0,07 | 180,0 | 23,54 | 4,11 | 23,0 |
| ResNet-50 | TensorRT FP32 | 3 | 1,00 ± 0,01 | 172,7 | 23,54 | 4,11 | 100,8 |
| ResNet-50 | TensorRT FP16 | 3 | 1,00 ± 0,01 | 173,7 | 23,54 | 4,11 | 45,5 |
| ResNet-50 | TensorRT INT8 | 3 | 1,01 ± 0,03 | 174,3 | 23,54 | 4,11 | 23,9 |
| ResNet-50 | poda 90% + int8 CPU | 1 | 1,14 | 202 | 2,40 | 4,11 | 23,0 |
| ResNet-50 | poda 95% + int8 CPU | 1 | 1,34 | 238 | 1,23 | 4,11 | 23,0 |
| ResNet-50 | poda 90% + TensorRT INT8 | 1 | 1,10 | 195 | 2,40 | 4,11 | 23,9 |
| ResNet-50 | poda 95% + TensorRT INT8 | 1 | 1,33 | 235 | 1,23 | 4,11 | 23,9 |
| DenseNet-121 | baseline | 3 | 1,00 | 164,7 | 6,97 | 2,86 | 27,2 |
| DenseNet-121 | poda 50% | 3 | 0,97 ± 0,06 | 159,3 | 3,53 | 2,86 | 15,5 |
| DenseNet-121 | poda 70% | 3 | 0,98 ± 0,05 | 161,7 | 2,15 | 2,86 | 10,8 |
| DenseNet-121 | poda 90% | 3 | 1,03 ± 0,04 | 169,0 | 0,77 | 2,86 | 5,7 |
| DenseNet-121 | poda 95% | 3 | 1,19 ± 0,05 | 196,3 | 0,43 | 2,86 | 4,2 |
| DenseNet-121 | poda 98% | 3 | 1,65 ± 0,05 | 272,0 | 0,22 | 2,86 | 3,2 |
| DenseNet-121 | int8 CPU (fbgemm) | 3 | 1,24 ± 0,07 | 203,7 | 6,97 | 2,86 | 7,7 |
| DenseNet-121 | TensorRT FP32 | 3 | 1,00 ± 0,00 | 165,0 | 6,97 | 2,86 | 34,3 |
| DenseNet-121 | TensorRT FP16 | 3 | 1,00 ± 0,00 | 164,3 | 6,97 | 2,86 | 15,0 |
| DenseNet-121 | TensorRT INT8 | 3 | 1,07 ± 0,03 | 175,3 | 6,97 | 2,86 | 14,7 |
| DenseNet-121 | poda 90% + int8 CPU | 1 | 1,37 | 220 | 0,77 | 2,86 | 7,7 |
| DenseNet-121 | poda 95% + int8 CPU | 1 | **1,58** | 253 | 0,43 | 2,86 | 7,7 |
| DenseNet-121 | poda 90% + TensorRT INT8 | 1 | 1,16 | 185 | 0,77 | 2,86 | 14,7 |
| DenseNet-121 | poda 95% + TensorRT INT8 | 1 | 1,29 | 207 | 0,43 | 2,86 | 14,7 |

Fonte: `results/eixo3/eixo3_summary.csv` (por célula) e `results/eixo3/eixo3_table.csv` (por seed). Células com `seeds = 1` são seed 0.

### 1.2 Latência, p50 em ms (seed 0)

CPU: Ryzen 9 9950X, máquina em repouso (20/09). GPU: RTX 3080 Ti, **medição final de 18/10, mediana de 5 rodadas** por célula. "—" = não se aplica ou não medido.

| arquitetura | célula | CPU 1 thr, lote 1 | CPU 16 thr, lote 1 | CPU 16 thr, lote 32 | GPU eager FP32, lote 1 | GPU eager FP16, lote 1 | GPU TRT, lote 1 | GPU TRT, lote 32 |
|---|---|---|---|---|---|---|---|---|
| ResNet-50 | baseline FP32 (eager) / TRT FP32 | 51,74 | 10,75 | 374,5 | 1,803 (lote 32: 23,2) | 1,480 (lote 32: 10,6) | 2,652 | 14,74 |
| ResNet-50 | poda 50% | — | 11,03 | — | — | — | — | — |
| ResNet-50 | poda 70% | — | 11,28 | — | — | — | — | — |
| ResNet-50 | poda 90% | — | 10,86 | — | 1,98 | — | — | — |
| ResNet-50 | poda 95% | — | 11,39 | — | 1,93 | — | — | — |
| ResNet-50 | poda 98% | — | 11,34 | — | — | — | — | — |
| ResNet-50 | int8 CPU (fbgemm) | 8,00 | **1,385** | 33,3 | — | — | — | — |
| ResNet-50 | TensorRT FP16 | — | — | — | — | — | 0,823 | 4,09 |
| ResNet-50 | TensorRT INT8 | — | — | — | — | — | **0,798** | 3,10 |
| ResNet-50 | poda 90% + TensorRT INT8 | — | — | — | — | — | 0,814 | — |
| ResNet-50 | poda 95% + TensorRT INT8 | — | — | — | — | — | 0,805 | — |
| DenseNet-121 | baseline FP32 (eager) / TRT FP32 | 41,25 | 16,64 | 377,1 | 3,497 (lote 32: 25,6) | 3,327 (lote 32: 13,3) | 4,206 | 18,13 |
| DenseNet-121 | poda 50% | — | 15,91 | — | — | — | — | — |
| DenseNet-121 | poda 70% | — | 15,79 | — | — | — | — | — |
| DenseNet-121 | poda 90% | — | 16,51 | — | 3,70 | — | — | — |
| DenseNet-121 | poda 95% | — | 16,24 | — | 3,72 | — | — | — |
| DenseNet-121 | poda 98% | — | 16,54 | — | — | — | — | — |
| DenseNet-121 | int8 CPU (fbgemm) | 7,23 | **2,86** | 31,5 | — | — | — | — |
| DenseNet-121 | TensorRT FP16 | — | — | — | — | — | 2,799 | 8,57 |
| DenseNet-121 | TensorRT INT8 | — | — | — | — | — | 3,192 | 17,21 |
| DenseNet-121 | poda 90% + TensorRT INT8 | — | — | — | — | — | 3,397 | — |
| DenseNet-121 | poda 95% + TensorRT INT8 | — | — | — | — | — | 3,339 | — |

Fonte: `results/eixo3/eixo3_summary.csv` (colunas `lat_*`), JSONs `runs/<run>/latency_*cuda*.json` (campo `note`: "5 rodadas (mediana)"; `p50_ms_per_round` por rodada), `runs/latency_cpu.log`; medição de TTY de 17/10 arquivada em `*_tty1round.json` e `runs/latency_gpu_tty.log`; a de 20/09 em `runs/latency_gpu_prelim.log`. Na linha do baseline, a coluna TRT é a engine FP32 (TF32 desligado). As três células int8 da ResNet-50 rodam o mesmo grafo: a mediana do grupo é 0,805 ms e a dispersão entre elas (0,798–0,814) é variância de build e de medição. Medição complementar (15–20/09): ONNX Runtime CPU FP32, lote 1: ResNet-50 5,9 ms, DenseNet-121 9,8 ms, sem ganho de 1 para 16 threads; em CPU o ORT é mais rápido que o eager na ResNet e mais lento na DenseNet, mais um sinal de que o backend interage com a arquitetura.

Figuras de Pareto: `results/eixo3/figures/pareto_{cpu,gpu,disk,macs}`.

### 1.3 Leitura

1. **Poda não-estruturada não compra latência.** Em CPU os podados ficam em 10,9–11,4 ms (ResNet-50, baseline 10,7) e 15,8–16,5 ms (DenseNet-121, baseline 16,6); em GPU rodam o mesmo grafo. O que cai é o artefato comprimido (10% do original a 98%) e os parâmetros efetivos.
2. **Quantização é o que acelera.** CPU int8: 7,8× na ResNet-50 (10,7 → 1,38 ms, 16 threads) e 5,8× na DenseNet-121 (16,6 → 2,86 ms). GPU, contra o baseline FP32 no mesmo runtime: ResNet-50 3,3× (2,65 → 0,80 ms); FP16 já dá 3,2× (0,82 ms) e a int8 só a supera com folga em lote 32 (3,10 vs 4,09 ms, 1,3×).
3. **A int8 não favorece a DenseNet-121:** em CPU custa 1,24× de erro; no TensorRT o erro quase não muda (1,07×), mas a latência fica pior que a FP16 (3,19 vs 2,80 ms; lote 32: 17,2 vs 8,6 ms). Causa: 399 das 1.497 camadas da engine ficam em float, com conversões Q/DQ em torno das concatenações.
4. **Custo teórico e real divergem.** A DenseNet-121 tem 30% menos MACs e 3,4× menos parâmetros, mas é mais lenta em todos os runtimes (TensorRT FP32: 4,21 vs 2,65 ms; CPU 16 thr: 16,6 vs 10,7 ms).
5. **As técnicas compõem de forma aproximadamente aditiva** na ResNet-50: poda 90% (1,07× na seed 0) + int8 dá 1,14× em CPU e 1,10× no TensorRT; 95% + int8, 1,34× / 1,33×. Na DenseNet-121 a int8 de CPU domina: 95% + int8 CPU = 1,58×, fora do orçamento.
6. **Lote 32:** o TensorRT INT8 da ResNet-50 processa 32 imagens em 3,10 ms (~10.300 img/s); a DenseNet INT8, em 17,2 ms.
7. **A latência de lote 1 em GPU é bimodal entre medições.** Uma medição isolada alterna entre dois patamares ~15% apart para qualquer engine (clock estável); a diferença FP16 × int8 na ResNet-50 (3%) é menor que esse ruído. Só a mediana de rodadas repetidas ordena as células com segurança; as células que rodam o mesmo grafo ficam dentro de 2% uma da outra depois disso (`results/eixo3/build_variance.md`).

## 2. Eixo 3: Decisão

**Critério (fixado em 14/09/2026, antes dos resultados):** entre as células com razão de erro média ≤ 1,5× do baseline da mesma arquitetura, vence a de **menor latência p50 em GPU (TensorRT, lote 1)**; empate → menor artefato em disco. Aplicado por `results/eixo3/make_eixo3.py --decide`; resultado em `results/eixo3/decision.md`.

**Excluídas pela razão de erro (> 1,5×):** DenseNet-121 poda 98% (1,65×), DenseNet-121 poda 95% + int8 CPU (1,58×), ResNet-50 poda 98% (1,90×).

**Finalistas (menor latência GPU):**

| arquitetura | célula | seeds | razão de erro | TRT lote 1, esta engine (ms) | TRT lote 1, grafo (ms) | artefato (bytes) |
|---|---|---|---|---|---|---|
| ResNet-50 | TensorRT INT8 | 3 | 1,01 | 0,798 | 0,805 | 25.078.800 |
| ResNet-50 | poda 95% + TensorRT INT8 | 1 | 1,33 | 0,805 | 0,805 | 25.112.300 |
| ResNet-50 | poda 90% + TensorRT INT8 | 1 | 1,10 | 0,814 | 0,805 | 25.076.000 |
| ResNet-50 | TensorRT FP16 | 3 | 1,00 | 0,823 | 0,823 | 47.731.200 |
| ResNet-50 | baseline (TRT FP32) | 3 | 1,00 | 2,652 | 2,652 | 94.492.700 |

Fonte: `results/eixo3/decision.md` (medição final de 18/10, mediana de 5 rodadas). "Grafo" = mediana das células da mesma arquitetura que rodam o mesmo grafo TensorRT.

**Leitura literal do critério (engine mais rápida, sem agrupar):** vence ResNet-50 **TensorRT INT8** (0,798 ms, razão 1,01×, 3 seeds), por 0,9% sobre poda 95% + int8 (0,805 ms) e 2% sobre poda 90% + int8 (0,814 ms). As três células rodam o **mesmo grafo int8**: a poda não-estruturada não altera os kernels; a dispersão entre elas é variância de build (~4% entre builds da mesma engine) e de medição, e está abaixo da estabilidade do harness (5%).

**Refinamentos (sinalizados ao orientador, 18/10):** (1) células com o mesmo grafo **compartilham a latência**, a mediana do grupo (0,805 ms para o grafo int8 da ResNet-50); (2) latências dentro de 5% contam como empate; desempate por artefato (o do critério); artefatos dentro de 1% também empatam e vence a menor razão de erro. Com eles a **vencedora é a mesma: ResNet-50 TensorRT INT8**, razão de erro 1,01×, 0,805 ms por imagem (esta engine: 0,798 ms), artefato de 23,9 MiB. A FP16 (0,823 ms) entra no empate de 5% e perde pelo artefato (45,5 MiB).

Em todas as leituras a arquitetura é a ResNet-50 e a precisão é int8 sem poda, então o eixo 4 (ResNet-50 densa + int8 TensorRT por ponto) foi feito na configuração certa. Caso a leitura literal tivesse apontado uma célula podada, cada ponto do eixo 4 exigiria ainda poda + fine-tuning na fração (previsto em `reduce_data.py --sparsity`); o extra B (seção 4) mede esse encadeamento em 25% dos dados.

**Status:** decisão **confirmada** com a medição final de 18/10. A decisão preliminar de 20/09 (uma rodada, sessão gráfica aberta) apontava a mesma célula; a medição única em TTY de 17/10 teria invertido a ordem entre as três células int8 do mesmo grafo (0,797 / 0,920 / 0,923 ms) e colocado a FP16 à frente — o que motivou o experimento de variância e as rodadas repetidas (`results/eixo3/build_variance.md`).

## 3. Eixo 4: Redução de dados anotados (ResNet-50)

**Pergunta.** Quanto do desempenho da configuração vencedora depende da quantidade de dados anotados?

**Receita ("reduz dados, mantém receita").** Mesmos hiperparâmetros do eixo 1 (SGD Nesterov, lr 0,0375, warmup 2 + cosseno, batch 96, AMP), **30 épocas fixas** com a agenda cosseno levada até o fim, `best.pt` por F1 macro em `val`, teste avaliado uma vez. A fração 100% é um run próprio com a mesma receita, para que todos os pontos da curva sejam comparáveis. Seed K governa a máscara (`frac_F_sK`), a inicialização da cabeça, a ordem dos dados e a augmentation. `val` e `test` são os mesmos em todas as frações. Em seguida, cada ponto é quantizado para TensorRT INT8 com calibração **restrita às imagens da fração daquele run** (1.024 imagens, percentil 99,99).

### 3.1 Resultados por fração (3 seeds)

| fração | imagens em `fit` | acurácia FP32 (%) | erro FP32 (%) | razão de erro FP32 | acurácia INT8 TRT (%) | razão de erro INT8 TRT | época do `best.pt` (média) |
|---|---|---|---|---|---|---|---|
| 100% | 175.591 | 99,795 ± 0,007 | 0,205 ± 0,007 | 1,00 | 99,799 ± 0,008 | 1,00 | 27,7 |
| 75% | 131.693 | 99,785 ± 0,014 | 0,215 ± 0,014 | 1,05 ± 0,05 | 99,785 ± 0,018 | 1,07 ± 0,05 | 28,3 |
| 50% | 87.798 | 99,738 ± 0,032 | 0,262 ± 0,032 | 1,28 ± 0,16 | 99,738 ± 0,036 | 1,31 ± 0,18 | 23,7 |
| 25% | 43.898 | 99,711 ± 0,011 | 0,289 ± 0,011 | 1,41 ± 0,09 | 99,703 ± 0,011 | 1,49 ± 0,09 | 26,7 |
| 10% | 17.561 | 99,522 ± 0,015 | 0,478 ± 0,015 | 2,33 ± 0,13 | 99,510 ± 0,012 | 2,45 ± 0,13 | 21,7 |
| 5% | 8.782 | 99,359 ± 0,029 | 0,641 ± 0,029 | 3,13 ± 0,21 | 99,340 ± 0,026 | 3,30 ± 0,21 | 19,7 |

Fonte: `results/eixo4/eixo4_summary.csv`. A razão é calculada contra o ponto de 100% **da mesma seed**.

### 3.2 Por seed (erros no teste e época selecionada)

| fração | s0 FP32 (ép.) | s1 FP32 (ép.) | s2 FP32 (ép.) | s0 INT8 | s1 INT8 | s2 INT8 |
|---|---|---|---|---|---|---|
| 100% | 177 (28) | 172 (28) | 165 (27) | 173 | 170 | 160 |
| 75% | 183 (27) | 189 (30) | 167 (28) | 193 | 182 | 164 |
| 50% | 201 (22) | 250 (19) | 206 (30) | 197 | 254 | 207 |
| 25% | 232 (24) | 251 (28) | 242 (28) | 239 | 258 | 249 |
| 10% | 403 (20) | 385 (28) | 410 (17) | 417 | 398 | 414 |
| 5% | 511 (24) | 559 (19) | 538 (16) | 529 | 573 | 554 |

Fonte: `results/eixo4/eixo4_runs.csv`. Os pontos de 100% (177, 172, 165) são runs novos e ficam dentro da faixa do eixo 1 (177, 165, 178), o que confirma a reprodutibilidade da receita.

### 3.3 Lei de potência

Ajuste `erro = a × N^−α` por mínimos quadrados em log-log sobre a curva média (Hestness et al., 2017), 6 pontos:

| estágio | α | a | R² |
|---|---|---|---|
| FP32 | 0,384 | 0,200 | 0,977 |
| TensorRT INT8 | 0,399 | 0,239 | 0,984 |

Fonte: `results/eixo4/eixo4_powerlaw.json` (`make_eixo4.py --powerlaw`). Figuras: `results/eixo4/figures/eixo4_curva`, `eixo4_por_classe_resnet50_{dense,dense+trt-int8}`.

**Leitura.** Com 75% dos dados o erro é indistinguível do de 100% (1,05×, dentro do ruído). Com metade, 1,28×; com um quarto, 1,41× (o erro sobe 41%). O joelho fica entre 25% e 10%: a 10% o erro mais que dobra (2,33×) e a 5% triplica (3,13×). Mesmo com 5% (8,8 mil imagens) a acurácia é 99,36%. A versão int8 da configuração vencedora acompanha a curva densa ponto a ponto (razões 1,07 / 1,31 / 1,49 / 2,45 / 3,30), com α compatível. O regime é fine-tuning a partir de ImageNet com 30 épocas fixas: a curva mede eficiência de dados nesse regime, não o orçamento de compute (frações pequenas treinam em minutos: 3–4 min a 5%). Custo da fila: 18 runs em 10,75 h (20/09 14:38 → 21/09 01:23) mais 1,7 h de continuação.

### 3.4 Parada antecipada: a primeira rodada, descartada

A decisão de 14/09 previa "30 épocas + early stopping por validação interna". A primeira rodada do eixo 4 usou paciência 5 e produziu uma **anomalia**: o ponto de 100% da seed 2 fez 277 erros (contra 177 e 182 nas outras seeds), pior que os pontos de 75% e 50% da mesma seed.

**Diagnóstico.** O run parou na época 15 com o lr em 56% do pico (0,0208 de 0,0375); o `best.pt` era da época 10. O erro em `val` oscila ±0,05 p.p. entre épocas vizinhas, então cinco épocas sem recorde ocorrem por acaso no meio do treino; com agenda cosseno, parar no meio corta justamente a fase em que o decaimento do lr traz o ganho final. **13 dos 18 runs** pararam antes da época 30 (10% s0 com lr a 33%; 50% s1 a 11%), e eram os de mais erros dentro de cada fração.

**Correção.** `--patience 0`: 30 épocas fixas, `best.pt` por `val`, como no eixo 1. A seleção do melhor checkpoint por `val` cumpre o papel da parada antecipada sem truncar a receita. Os 13 runs foram **continuados** de `last.pt` até a época 30 (o resume reproduz lote a lote o treino ininterrupto); os resultados truncados ficam em `metrics_es5.json` de cada run. Sinalizado ao orientador como ajuste de parâmetro de uma decisão fechada; confirmação pendente.

| run | parada (ép. do best / erros) | 30 épocas (ép. do best / erros) |
|---|---|---|
| 100% s2 | 10 / 277 | 27 / 165 |
| 100% s1 | 22 / 182 | 28 / 172 |
| 10% s0 | 14 / 475 | 20 / 403 |
| 10% s1 | 17 / 427 | 28 / 385 |
| 75% s1 | 23 / 188 | 30 / 189 |
| 50% s2 | 22 / 205 | 30 / 206 |
| demais 7 truncados (5% s0–s2, 10% s2, 25% s0, 50% s0–s1) | `best.pt` inalterado | mesmos erros |
| 5 runs não truncados (25% s1–s2, 75% s0 e s2, 100% s0) | — | — |

Fonte: `runs/eixo4_resnet50_f*_s*/metrics_es5.json` e `metrics.json`.

**Curva com paciência 5 (só para registro):** razão vs 100%: 75% 0,89 · 50% 1,08 · 25% 1,19 · 10% 2,17 · 5% 2,63, distorcida pelo ponto de 100% da seed 2. A curva válida é a da seção 3.1.

## 4. Extra B: compressão × quantidade de dados

**Pergunta.** A compressão custa o mesmo quando o modelo viu menos dados? Repete-se, sobre o modelo treinado com 25% dos dados (`eixo4_resnet50_f025_s0`), o que foi feito com 100%: poda global 90% + fine-tuning de 5 épocas **na mesma fração** (`extraB_resnet50_f025_p90_s0`, `scripts/extra_b.sh`), int8 CPU e int8 TensorRT (calibração restrita à fração). Razão de erro relativa ao **denso do mesmo regime**. Uma seed (0).

| célula | 100% dos dados: erros | 100%: razão vs denso | 25% dos dados: erros | 25%: razão vs denso |
|---|---|---|---|---|
| baseline FP32 | 177 | 1,00 | 232 | 1,00 |
| int8 CPU (fbgemm) | 170 | 0,96 | 266 | 1,15 |
| int8 TensorRT | 173 | 0,98 | 239 | 1,03 |
| poda 90% + fine-tuning 5 ép. | 190 | 1,07 | 393 | **1,69** |
| poda 90% + int8 CPU | 202 | 1,14 | 472 | **2,03** |
| poda 90% + int8 TensorRT | 195 | 1,10 | 420 | 1,81 |

Fonte: `results/analise/interacao.csv` (`make_interacao.py`); figura `results/analise/interacao.{pdf,png}`.

**Leitura.** O custo da compressão depende de quantos dados o modelo viu. Com 25% dos dados, a poda de 90% custa 1,69× (contra 1,07× com 100%) e poda + int8 CPU chega a 2,03× (contra 1,14×). Um modelo treinado com menos dados tem menos redundância para ceder. A direção é clara; a magnitude pede mais seeds (uma seed; 14 min de GPU). Consequência para a leitura literal da decisão do eixo 3: se a vencedora fosse "poda 90% + int8", a curva do eixo 4 cairia muito mais rápido que a da int8 sozinha.

## 5. Reprodução

```bash
make latency-cpu                                        # máquina em repouso
make latency-gpu                                        # 5 rodadas por célula (mediana); scripts/latency_tty.sh para TTY
python results/eixo3/make_eixo3.py --decide             # tabela, Pareto, decision.md
python src/reduce_data.py --config configs/train_resnet50.yaml --seeds 0 1 2   # 18 runs, retomável
scripts/queue_trt.sh "$(ls -d runs/eixo4_resnet50_f*_s[012])" "int8"
python results/eixo4/make_eixo4.py --powerlaw
scripts/extra_b.sh && python results/analise/make_interacao.py
```

Pesos do eixo 4: release `v0.5-eixo4` (18 × `best.pt`, frações 100/75/50/25/10/5 × seeds 0–2, 3,2 GiB).
