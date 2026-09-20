# Decisão do eixo 3

Critério (fixado em 2026-09-14): entre as células com razão de erro média ≤ 1.5×, vence a de menor latência p50 em GPU (TensorRT, lote 1); empate → menor artefato em disco.

## Vencedora: **ResNet-50 — trt-int8**

razão de erro 1.08× · latência TRT lote 1 0.800 ms · artefato 23.9 MiB

**Leitura literal do critério:** ResNet-50 — prune-90+trt-int8 (0.793 ms, razão 1.42×). A diferença de latência para a vencedora acima é de 0.9%, abaixo da tolerância de estabilidade do harness (5%): as duas células rodam o mesmo grafo int8 (a poda não-estruturada não acelera kernels densos). **Refinamento proposto, pendente de confirmação:** latências dentro de 5% contam como empate; desempate por artefato (o do critério); artefatos dentro de 1% também empatam e vence a menor razão de erro.

Arquitetura em todas as leituras: **ResNet-50** — o treino denso do eixo 4 é o mesmo em qualquer desfecho.

## Finalistas

| arch     | cell              |   n_seeds |   error_ratio_mean |   lat_gpu_trt_b1_mean |   artifact_bytes_mean |
|:---------|:------------------|----------:|-------------------:|----------------------:|----------------------:|
| resnet50 | prune-90+trt-int8 |         1 |            1.42373 |              0.792821 |           2.50671e+07 |
| resnet50 | trt-int8          |         3 |            1.08188 |              0.799906 |           2.50731e+07 |
| resnet50 | prune-95+trt-int8 |         1 |            1.48023 |              0.930051 |           2.51117e+07 |
| resnet50 | trt-fp16          |         3 |            1.00201 |              0.941446 |           4.77395e+07 |
| resnet50 | baseline          |         3 |            1       |              2.64456  |           9.44927e+07 |

## Excluídas pela razão de erro

| arch        | cell              |   error_ratio_mean |
|:------------|:------------------|-------------------:|
| densenet121 | prune-90+trt-int8 |            1.5125  |
| densenet121 | prune-95+int8-cpu |            1.58125 |
| densenet121 | prune-95+trt-int8 |            1.91875 |
| densenet121 | prune-98          |            1.65162 |
| resnet50    | prune-98          |            1.89702 |

## Condições da medição de latência GPU

- PRELIMINAR: sessão gráfica ABERTA (KDE); refazer em TTY
