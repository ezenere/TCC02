# Eixo 3 — Benchmark: qualidade × custo × latência

> Gerado por `make_eixo3.py` a partir de `runs/` (`eixo3_table.csv` por seed, `eixo3_summary.csv` por célula). Razão de erro = taxa de erro
> da célula ÷ taxa de erro do baseline da **mesma seed**, no teste. Custo e latência medidos na seed 0. Latência GPU: p50 de
> 300 iterações após 50 de aquecimento, **mediana de 5 rodadas** por célula (medição final de 18/10; ver [`build_variance.md`](build_variance.md)).

## Tabela cruzada

Células: baseline, poda (5 níveis), int8 em CPU (fbgemm), TensorRT FP32/FP16/INT8, e poda + int8. "—" = não se aplica ou não medido.
Na linha `baseline`, a coluna de GPU é a engine TensorRT **FP32** (o baseline medido no mesmo runtime das células quantizadas).

<!-- eixo3:table:start -->
| arquitetura | célula | seeds | razão de erro | não-nulos (M) | MACs (G) | artefato (MiB) | CPU 16 thr, lote 1 (ms) | GPU TRT, lote 1 (ms) |
|---|---|---|---|---|---|---|---|---|
| DenseNet-121 | baseline | 3 | 1.00 | 6.97 | 2.86 | 27.2 | 16.64 | 4.206 |
| DenseNet-121 | prune-50 | 3 | 0.97 ± 0.06 | 3.53 | 2.86 | 15.5 | 15.91 | — |
| DenseNet-121 | prune-70 | 3 | 0.98 ± 0.05 | 2.15 | 2.86 | 10.8 | 15.79 | — |
| DenseNet-121 | prune-90 | 3 | 1.03 ± 0.04 | 0.77 | 2.86 | 5.7 | 16.51 | — |
| DenseNet-121 | prune-95 | 3 | 1.19 ± 0.05 | 0.43 | 2.86 | 4.2 | 16.24 | — |
| DenseNet-121 | prune-98 | 3 | 1.65 ± 0.05 | 0.22 | 2.86 | 3.2 | 16.54 | — |
| DenseNet-121 | int8-cpu | 3 | 1.24 ± 0.07 | 6.97 | 2.86 | 7.7 | 2.86 | — |
| DenseNet-121 | trt-fp32 | 3 | 1.00 ± 0.00 | 6.97 | 2.86 | 34.3 | — | 4.206 |
| DenseNet-121 | trt-fp16 | 3 | 1.00 ± 0.00 | 6.97 | 2.86 | 15.0 | — | 2.799 |
| DenseNet-121 | trt-int8 | 3 | 1.07 ± 0.03 | 6.97 | 2.86 | 14.7 | — | 3.192 |
| DenseNet-121 | prune-90+int8-cpu | 1 | 1.37 | 0.77 | 2.86 | 7.7 | — | — |
| DenseNet-121 | prune-95+int8-cpu | 1 | 1.58 | 0.43 | 2.86 | 7.7 | — | — |
| DenseNet-121 | prune-90+trt-int8 | 1 | 1.16 | 0.77 | 2.86 | 14.7 | — | 3.397 |
| DenseNet-121 | prune-95+trt-int8 | 1 | 1.29 | 0.43 | 2.86 | 14.7 | — | 3.339 |
| ResNet-50 | baseline | 3 | 1.00 | 23.54 | 4.11 | 90.1 | 10.75 | 2.652 |
| ResNet-50 | prune-50 | 3 | 0.97 ± 0.03 | 11.80 | 4.11 | 51.0 | 11.03 | — |
| ResNet-50 | prune-70 | 3 | 0.98 ± 0.02 | 7.10 | 4.11 | 34.8 | 11.28 | — |
| ResNet-50 | prune-90 | 3 | 1.10 ± 0.03 | 2.40 | 4.11 | 17.2 | 10.86 | — |
| ResNet-50 | prune-95 | 3 | 1.28 ± 0.07 | 1.23 | 4.11 | 12.1 | 11.39 | — |
| ResNet-50 | prune-98 | 3 | 1.90 ± 0.17 | 0.52 | 4.11 | 8.8 | 11.34 | — |
| ResNet-50 | int8-cpu | 3 | 1.04 ± 0.07 | 23.54 | 4.11 | 23.0 | 1.38 | — |
| ResNet-50 | trt-fp32 | 3 | 1.00 ± 0.01 | 23.54 | 4.11 | 100.8 | — | 2.652 |
| ResNet-50 | trt-fp16 | 3 | 1.00 ± 0.01 | 23.54 | 4.11 | 45.5 | — | 0.823 |
| ResNet-50 | trt-int8 | 3 | 1.01 ± 0.02 | 23.54 | 4.11 | 23.9 | — | 0.798 |
| ResNet-50 | prune-90+int8-cpu | 1 | 1.14 | 2.40 | 4.11 | 23.0 | — | — |
| ResNet-50 | prune-95+int8-cpu | 1 | 1.34 | 1.23 | 4.11 | 23.0 | — | — |
| ResNet-50 | prune-90+trt-int8 | 1 | 1.10 | 2.40 | 4.11 | 23.9 | — | 0.814 |
| ResNet-50 | prune-95+trt-int8 | 1 | 1.33 | 1.23 | 4.11 | 23.9 | — | 0.805 |
<!-- eixo3:table:end -->

Figuras de Pareto (razão de erro × custo): `figures/pareto_cpu`, `pareto_gpu`, `pareto_disk`, `pareto_macs` (`.pdf` e `.png`).

## Leitura

1. **Poda não-estruturada não compra latência.** Os modelos podados têm a mesma latência do baseline em CPU (10,9–11,4 ms contra 10,7 ms na
   ResNet-50) e o mesmo grafo em GPU; o que cai é o artefato comprimido (até 10% do original a 98%) e os parâmetros efetivos.
2. **Quantização é o que acelera.** int8 em CPU: 7,8× na ResNet-50 (10,7 → 1,38 ms, 16 threads) e 5,8× na DenseNet-121 (16,6 → 2,86 ms).
   Em GPU, contra o baseline FP32 no mesmo runtime: ResNet-50 3,3× (2,65 → 0,80 ms).
3. **A int8 não favorece a DenseNet-121:** em CPU custa 1,24× de erro (ResNet: 1,04×); no TensorRT o erro quase não muda (1,07×), mas a
   **latência fica pior que a FP16** (3,19 contra 2,80 ms em lote 1; 17,2 contra 8,57 ms em lote 32) — centenas de camadas em torno das
   concatenações ficam em float, com conversões Q/DQ. Na ResNet-50, int8 e FP16 ficam a 3% em lote 1 (0,80 contra 0,82 ms) e a int8 é
   1,3× mais rápida em lote 32 (3,10 contra 4,09 ms).
4. **Custo teórico e real divergem.** A DenseNet-121 tem 30% menos MACs e 3,4× menos parâmetros que a ResNet-50, mas é mais lenta em todos os
   runtimes medidos (FP32 TensorRT: 4,21 contra 2,65 ms).
5. **As técnicas compõem de forma aproximadamente aditiva:** na ResNet-50, poda 90% (1,07×) + int8 dá 1,14× em CPU e 1,10× no TensorRT.
6. **A calibração da int8 é um hiperparâmetro que precisa de validação.** Com entropia, o mesmo modelo variou de 194 a 2.596 erros conforme a
   amostra de calibração; com percentil 99,99 (escolhido em `val`), de 62 a 64 em validação. Ver `results/eixo2/calib_sweep.csv`.
7. **A latência de lote 1 em GPU é bimodal entre medições.** Uma medição isolada (50 + 300 iterações) alterna entre dois patamares
   ~15% apart (≈ 0,80 e ≈ 0,93 ms na ResNet-50 int8) para qualquer engine, com o clock SM estável; a diferença entre FP16 e int8 é
   menor que isso. A medição única em TTY de 17/10 pegou a FP16 no patamar rápido e a int8 no lento, e inverteu a ordem de três
   células que rodam o mesmo grafo. A mesma engine reconstruída três vezes varia ~4%. Por isso a medição final (18/10) usa 5 rodadas
   por célula (mediana do p50); entre TTY e sessão gráfica aberta não há diferença sistemática. Experimento completo em
   [`build_variance.md`](build_variance.md).

## Decisão

Critério fixado antes dos resultados: menor latência p50 em GPU (TensorRT, lote 1) entre as células com razão de erro ≤ 1,5×; empate → menor
artefato. Aplicação, finalistas, leitura literal × refinada e condições da medição em [`decision.md`](decision.md) (`make_eixo3.py --decide`).
A configuração vencedora alimenta o eixo 4.

**Refinamento (18/10, sinalizado ao orientador):** a poda não-estruturada não muda o grafo nem os kernels, então as células
`trt-int8`, `prune-90+trt-int8` e `prune-95+trt-int8` da mesma arquitetura rodam a mesma engine a menos da variância de build e de
medição; na regra de decisão elas **compartilham a latência** (mediana do grupo: 0,805 ms na ResNet-50, de 0,798 / 0,814 / 0,805) e
o desempate dentro do grupo segue o critério (artefato; artefatos a menos de 1% → menor razão de erro). Latências dentro de 5% contam
como empate (estabilidade do harness). Resultado: **ResNet-50 trt-int8**, razão de erro 1,01×, 0,805 ms (esta engine: 0,798 ms),
23,9 MiB. A leitura literal (engine mais rápida, sem agrupar) aponta a mesma célula, então a vencedora é a mesma da decisão
preliminar de 20/09 e o eixo 4 (ResNet-50 densa + int8 por ponto) já estava na configuração certa. Justificativa em [`build_variance.md`](build_variance.md).

## Reprodução

```bash
make latency-cpu                                   # máquina em repouso
make latency-gpu                                   # 5 rodadas por célula (mediana); sessão aberta ou TTY
scripts/latency_tty.sh                             # alternativa: TTY com a sessão gráfica parada, engines reconstruídas
make eixo3                                         # tabela, Pareto e decision.md
```
