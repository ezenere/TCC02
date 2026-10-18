# Variância da latência em GPU: builds e medições (ResNet-50 seed 0, TensorRT, lote 1)

Motivação: a medição em TTY de 17/10 inverteu a ordem entre células int8 da ResNet-50 que rodam **o mesmo grafo**
(baseline int8 0,920 ms; poda 90% + int8 0,923 ms; poda 95% + int8 **0,797 ms**), e colocou a FP16 (0,827 ms) à frente da int8.
A poda não-estruturada não altera o grafo nem os kernels (em CPU, podados = baseline), então essa dispersão não pode ser
propriedade dos pesos. Dois experimentos em 18/10 (sessão gráfica aberta; só a ordem relativa importa):

## 1. A mesma engine construída três vezes (variância entre builds)

| build | p50 lote 1 |
|---|---|
| 1 | 0,926 ms |
| 2 | 0,886 ms |
| 3 | 0,923 ms |

Dispersão de ~4%: a seleção de kernels por cronometragem do TensorRT varia pouco entre builds.

## 2. As três engines de TTY, medidas em sequência, mesmas condições

| engine (construída em TTY) | TTY, 17/10 | consecutivas, 18/10 |
|---|---|---|
| baseline int8 | 0,920 ms | 0,923 ms |
| poda 90% + int8 | 0,923 ms | 0,930 ms |
| poda 95% + int8 | **0,797 ms** | 0,928 ms |

O 0,797 ms não se reproduz: foi variância **de medição**, não da engine.

## 3. FP16 × INT8 alternadas, 4 rodadas (clock SM 1.980–2.010 MHz em todas)

| rodada | FP16 lote 1 | INT8 lote 1 | FP16 lote 32 | INT8 lote 32 |
|---|---|---|---|---|
| 1 | 0,947 | 0,806 | 4,07 | 3,30 |
| 2 | 0,840 | 0,800 | 4,30 | 3,33 |
| 3 | 0,930 | 0,844 | 4,32 | 3,11 |
| 4 | 0,947 | 0,923 | 4,32 | 3,11 |
| **mediana** | **0,94** | **0,82** | **4,31** | **3,20** |

A int8 é mais rápida que a FP16 nas 4 rodadas em lote 1 e por 25% em lote 32. Cada medição isolada de lote 1, porém, alterna
entre dois patamares (~0,80 e ~0,93 ms) para **qualquer** engine, com o clock estável: ruído de medição de ~15%, maior que a
diferença entre as duas engines. A medição única em TTY pegou a FP16 no patamar rápido e a int8 no lento.

## Consequências para o protocolo

1. **Latência em GPU, lote 1, exige rodadas repetidas:** o harness passou a aceitar `--rounds N` (padrão 5 para GPU) e reporta
   a **mediana das rodadas**, além de cada rodada; a estabilidade de 5% entre duas medições consecutivas (`tests/test_latency.py`)
   não captura essa alternância de patamar entre sessões.
2. **Células com o mesmo grafo compartilham a latência** (mediana do grupo) na regra de decisão do eixo 3: a poda não-estruturada
   não muda o grafo. Implementado em `make_eixo3.py --decide` (`graph_family`).
3. Entre TTY e sessão gráfica aberta não houve diferença sistemática nas demais células (eager FP32 1,81 ms nas duas; TensorRT FP32
   2,645 nas duas; DenseNet idêntica dentro de 1%): o compositor ocioso não contaminou a medição; a bimodalidade aparece nos dois.

Arquivos brutos: medições deste experimento não entram em `runs/` (só esta nota); as medições oficiais são as de `runs/*/latency_*cuda*.json`.
