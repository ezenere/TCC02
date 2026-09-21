# Eixo 2 — Compressão: poda e quantização

> Tabelas geradas por `make_pruning.py` e `make_quant.py` a partir de `runs/`; nenhum número é digitado à mão.
> Razão de erro = taxa de erro da célula ÷ taxa de erro do baseline (eixo 1) **da mesma seed**, no teste.

## Poda global por magnitude + fine-tuning (5 épocas, lr 0,1×)

Média ± desvio entre seeds; "não-nulos" = parâmetros efetivos; gzip = tamanho do `state_dict` comprimido, único custo que a esparsidade não-estruturada reduz.

<!-- eixo2:prune:start -->
| esparsidade | ResNet-50 razão | DenseNet-121 razão | não-nulos R / D | gzip R / D (× baseline) |
|---|---|---|---|---|
| 50% | 0.97 ± 0.03 (n=3) | 0.97 ± 0.06 (n=3) | 11.80 M / 3.53 M | 0.61 / 0.61 |
| 70% | 0.98 ± 0.02 (n=3) | 0.98 ± 0.05 (n=3) | 7.10 M / 2.15 M | 0.42 / 0.43 |
| 90% | 1.10 ± 0.03 (n=3) | 1.03 ± 0.04 (n=3) | 2.40 M / 0.77 M | 0.21 / 0.22 |
| 95% | 1.28 ± 0.07 (n=3) | 1.19 ± 0.05 (n=3) | 1.23 M / 0.43 M | 0.14 / 0.17 |
| 98% | 1.90 ± 0.17 (n=3) | 1.65 ± 0.05 (n=3) | 0.52 M / 0.22 M | 0.10 / 0.13 |

**Joelho (razão > 1.5×):** DenseNet-121 98%, ResNet-50 98%
<!-- eixo2:prune:end -->

Figura: `figures/pruning_error_ratio.{pdf,png}`.

**Leitura.** Até 90% de esparsidade a degradação fica dentro do ruído entre seeds nas duas arquiteturas; 95% já é visível mas dentro do
orçamento de 1,5× do critério do eixo 3; o joelho está em 98%. A DenseNet-121 tolera cada nível um pouco melhor que a ResNet-50 apesar de
partir de 3,4× menos parâmetros. MACs e latência **não** mudam com poda não-estruturada (kernels densos calculam os zeros); o que cai é
o artefato comprimido, para ~21% em 90%. Poda estruturada, que reduziria MACs, é limitada na DenseNet pelas concatenações (ver METODOLOGIA § 2.1).

## Eixo 2b — podar antes ou depois do treino? (pedido do orientador)

**Depois** (protocolo original): treino completo de 30 épocas → poda global por magnitude → fine-tuning de 5 épocas.
**Antes**: poda global por magnitude sobre os pesos **ImageNet** (cabeça nova fica densa) → treino completo de 30 épocas com as
máscaras fixas. Mesma receita de otimização, mesmas seeds, mesma métrica. O custo de GPU por modelo podado é parecido quando se conta
o treino denso de origem (≈ 100 min), mas "antes" precisa de um treino completo **por nível** de esparsidade, enquanto "depois"
reaproveita um único treino denso para todos os níveis.

<!-- eixo2:compare:start -->
| arquitetura | esparsidade | poda depois: razão (erros) | poda antes: razão (erros) | min de GPU depois / antes |
|---|---|---|---|---|
| DenseNet-121 | 50% | 0.97 ± 0.06 (159, 168, 151) | 1.06 (170) | 17 / 99 |
| DenseNet-121 | 70% | 0.98 ± 0.05 (158, 171, 156) | 1.07 (171) | 17 / 99 |
| DenseNet-121 | 90% | 1.03 ± 0.04 (163, 178, 166) | 1.11 (178) | 17 / 101 |
| DenseNet-121 | 95% | 1.19 ± 0.05 (193, 206, 190) | 1.26 (202) | 17 / 101 |
| DenseNet-121 | 98% | 1.65 ± 0.05 (262, 285, 269) | 1.64 (263) | 17 / 99 |
| ResNet-50 | 50% | 0.97 ± 0.03 (166, 164, 174) | 0.99 (176) | 14 / 81 |
| ResNet-50 | 70% | 0.98 ± 0.02 (175, 163, 171) | 1.12 (199) | 14 / 81 |
| ResNet-50 | 90% | 1.10 ± 0.03 (190, 187, 192) | 1.08 (191) | 14 / 85 |
| ResNet-50 | 95% | 1.28 ± 0.07 (230, 221, 214) | 1.15 (203) | 14 / 82 |
| ResNet-50 | 98% | 1.90 ± 0.17 (323, 345, 316) | 1.36 (241) | 14 / 81 |
<!-- eixo2:compare:end -->

Figura: `figures/prune_compare.{pdf,png}` (`make_prune_compare.py`).

## Quantização int8 pós-treinamento

Calibração com as mesmas 1.024 imagens de `fit` nas duas rotas. CPU: FX graph mode, backend x86/fbgemm, artefato TorchScript. GPU: TensorRT 11,
rede fortemente tipada, ONNX Q/DQ com calibração por **percentil 99,99**, escolhida em `val` pela robustez à amostra de calibração
(`calib_sweep.csv`: 62–64 erros em val com percentil 99,99, contra 66–689 com entropia e 604–966 com MinMax; denso = 60). Acurácia reportada **por backend**.

<!-- eixo2:quant:start -->
| arquitetura | backend | precisão | seeds | razão de erro | erros / baseline (por seed) | artefato (× FP32) | img/s (aval.) |
|---|---|---|---|---|---|---|---|
| DenseNet-121 | cpu-fbgemm | int8 | 3 | 1.24 ± 0.07 | 210/160, 203/167, 198/167 | 7.7 MiB (0.28) | 476 |
| DenseNet-121 | gpu-tensorrt | fp16 | 3 | 1.00 ± 0.00 | 160/160, 166/167, 167/167 | 15.0 MiB (0.55) | 3198 |
| DenseNet-121 | gpu-tensorrt | fp32 | 3 | 1.00 ± 0.00 | 160/160, 167/167, 168/167 | 34.3 MiB (1.26) | 1670 |
| DenseNet-121 | gpu-tensorrt | int8 | 3 | 1.07 ± 0.03 | 175/160, 174/167, 177/167 | 14.7 MiB (0.54) | 1102 |
| ResNet-50 | cpu-fbgemm | int8 | 3 | 1.04 ± 0.07 | 170/177, 174/165, 196/178 | 23.0 MiB (0.26) | 452 |
| ResNet-50 | gpu-tensorrt | fp16 | 3 | 1.00 ± 0.01 | 176/177, 166/165, 179/178 | 45.5 MiB (0.51) | 5068 |
| ResNet-50 | gpu-tensorrt | fp32 | 3 | 1.00 ± 0.01 | 176/177, 163/165, 179/178 | 100.8 MiB (1.12) | 1965 |
| ResNet-50 | gpu-tensorrt | int8 | 3 | 1.01 ± 0.03 | 173/177, 166/165, 184/178 | 23.9 MiB (0.27) | 3845 |
<!-- eixo2:quant:end -->

**Leitura.** A int8 não custa nada à ResNet-50 em nenhum backend (CPU 1,04×, TensorRT 1,01×). Na DenseNet-121 o custo depende do backend:
**1,24× em CPU** (FX/fbgemm, escala por tensor nas ativações) e **1,07× no TensorRT** com calibração robusta. A hipótese para a CPU: a
concatenação densa junta num mesmo tensor ativações de escalas distintas, e a escala única por tensor perde resolução nos canais de menor
amplitude. **Lição de método:** com calibração por entropia e uma única amostra, o TensorRT parecia custar 1,19× à DenseNet e 1,08× à ResNet;
a varredura de robustez mostrou que boa parte disso era instabilidade do calibrador, não da arquitetura — o mesmo modelo ia de 194 a 2.596
erros no teste conforme as 1.024 imagens sorteadas. A engine int8 da DenseNet continua **mais lenta que a FP16** no TensorRT (3,17 contra
2,76 ms): centenas de camadas em torno das concatenações ficam em float, com conversões Q/DQ; na ResNet a int8 é 1,2× mais rápida que a FP16.

## Poda + int8 (célula extra do benchmark)

Aplicada nos níveis 90% e 95% (o último dentro do orçamento de 1,5×), seed 0. As duas técnicas **compõem de forma aproximadamente
aditiva**: a int8 acrescenta poucos erros ao modelo podado nos dois backends (TensorRT: ResNet 90% 190 → 195 erros, 95% 230 → 235;
DenseNet 90% 163 → 185, 95% 193 → 207; CPU: ResNet 90% 190 → 202). Tabela completa em `results/eixo3/eixo3_table.csv`
(células `prune-NN+int8-cpu` e `prune-NN+trt-int8`).

## Reprodução

```bash
scripts/queue_prune.sh "0 1 2" "0.5 0.7 0.9 0.95 0.98"        # poda (GPU)
python src/compress/quantize_cpu.py --checkpoint runs/eixo1_<arch>_s<k>/checkpoints/best.pt
scripts/queue_trt.sh "runs/eixo1_<arch>_s<k>" "fp32 fp16 int8"   # ONNX -> Q/DQ -> engines -> teste (GPU ociosa)
python results/eixo2/make_pruning.py; python results/eixo2/make_quant.py
```
