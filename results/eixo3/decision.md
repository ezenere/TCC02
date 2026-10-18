# Decisão do eixo 3

Critério (fixado em 2026-09-14): entre as células com razão de erro média ≤ 1.5×, vence a de menor latência p50 em GPU (TensorRT, lote 1); empate → menor artefato em disco.

## Vencedora: **ResNet-50 — trt-int8**

razão de erro 1.01× · latência TRT lote 1 0.805 ms (mediana do grafo ResNet-50 trt-int8; esta engine: 0.798 ms) · artefato 23.9 MiB

Células do mesmo grafo (latência compartilhada): prune-90+trt-int8 0.814 ms (razão 1.10×), trt-int8 0.798 ms (razão 1.01×), prune-95+trt-int8 0.805 ms (razão 1.33×).

Arquitetura em todas as leituras: **ResNet-50** — o treino denso do eixo 4 é o mesmo em qualquer desfecho.

## Finalistas

| arch     | cell              |   n_seeds |   error_ratio_mean |   lat_gpu_trt_b1_mean |   artifact_bytes_mean |
|:---------|:------------------|----------:|-------------------:|----------------------:|----------------------:|
| resnet50 | prune-90+trt-int8 |         1 |            1.10169 |              0.81426  |           2.5076e+07  |
| resnet50 | trt-int8          |         3 |            1.00949 |              0.797909 |           2.50788e+07 |
| resnet50 | prune-95+trt-int8 |         1 |            1.32768 |              0.804785 |           2.51123e+07 |
| resnet50 | trt-fp16          |         3 |            1.00201 |              0.822649 |           4.77312e+07 |
| resnet50 | baseline          |         3 |            1       |              2.65231  |           9.44927e+07 |

## Excluídas pela razão de erro

| arch        | cell              |   error_ratio_mean |
|:------------|:------------------|-------------------:|
| densenet121 | prune-95+int8-cpu |            1.58125 |
| densenet121 | prune-98          |            1.65162 |
| resnet50    | prune-98          |            1.89702 |

## Condições da medição de latência GPU

- 5 rodadas (mediana); sessão gráfica aberta; TTY de 17/10 sem diferença sistemática (results/eixo3/build_variance.md)
- sessão gráfica fechada (TTY)
