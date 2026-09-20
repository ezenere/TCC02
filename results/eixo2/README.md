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
<!-- eixo2:compare:end -->

Figura: `figures/prune_compare.{pdf,png}` (`make_prune_compare.py`).

## Quantização int8 pós-treinamento

Calibração com as mesmas 1.024 imagens de `fit` nas duas rotas. CPU: FX graph mode, backend x86/fbgemm, artefato TorchScript. GPU: TensorRT 11,
rede fortemente tipada, ONNX Q/DQ com calibração por entropia (escolhida em `val` contra MinMax). Acurácia reportada **por backend**.

<!-- eixo2:quant:start -->
| arquitetura | backend | precisão | seeds | razão de erro | erros / baseline (por seed) | artefato (× FP32) | img/s (aval.) |
|---|---|---|---|---|---|---|---|
| DenseNet-121 | cpu-fbgemm | int8 | 3 | 1.24 ± 0.07 | 210/160, 203/167, 198/167 | 7.7 MiB (0.28) | 476 |
| DenseNet-121 | gpu-tensorrt | fp16 | 3 | 1.00 ± 0.00 | 160/160, 166/167, 167/167 | 15.0 MiB (0.55) | 3198 |
| DenseNet-121 | gpu-tensorrt | fp32 | 3 | 1.00 ± 0.00 | 160/160, 167/167, 168/167 | 34.3 MiB (1.26) | 1670 |
| DenseNet-121 | gpu-tensorrt | int8 | 3 | 1.19 ± 0.08 | 190/160, 213/167, 185/167 | 14.5 MiB (0.53) | 1753 |
| ResNet-50 | cpu-fbgemm | int8 | 3 | 1.04 ± 0.07 | 170/177, 174/165, 196/178 | 23.0 MiB (0.26) | 452 |
| ResNet-50 | gpu-tensorrt | fp16 | 3 | 1.00 ± 0.01 | 176/177, 166/165, 179/178 | 45.5 MiB (0.51) | 5068 |
| ResNet-50 | gpu-tensorrt | fp32 | 3 | 1.00 ± 0.01 | 176/177, 163/165, 179/178 | 100.8 MiB (1.12) | 1965 |
| ResNet-50 | gpu-tensorrt | int8 | 3 | 1.08 ± 0.08 | 180/177, 174/165, 209/178 | 23.9 MiB (0.27) | 6189 |
<!-- eixo2:quant:end -->

**Leitura.** A int8 não custa nada à ResNet-50 em nenhum backend, mas degrada a DenseNet-121 nos dois (CPU 1,24×, TensorRT ≈1,2×): o efeito
é da arquitetura, não do runtime. Hipótese: a concatenação densa junta num mesmo tensor ativações de escalas distintas, e a escala única por
tensor do PTQ estático perde resolução nos canais de menor amplitude. A engine int8 da DenseNet é ainda **mais lenta que a FP16** no TensorRT,
porque centenas de camadas em torno das concatenações ficam em float com conversões Q/DQ; na ResNet a int8 é ~1,4× a FP16.

## Poda + int8 (célula extra do benchmark)

Aplicada nos níveis 90% e 95% (o último dentro do orçamento de 1,5×), seed 0. As degradações se somam de forma aproximadamente aditiva;
na DenseNet a int8 domina. Tabela completa em `results/eixo3/eixo3_table.csv` (células `prune-NN+int8-cpu` e `prune-NN+trt-int8`).

## Reprodução

```bash
scripts/queue_prune.sh "0 1 2" "0.5 0.7 0.9 0.95 0.98"        # poda (GPU)
python src/compress/quantize_cpu.py --checkpoint runs/eixo1_<arch>_s<k>/checkpoints/best.pt
scripts/queue_trt.sh "runs/eixo1_<arch>_s<k>" "fp32 fp16 int8"   # ONNX -> Q/DQ -> engines -> teste (GPU ociosa)
python results/eixo2/make_pruning.py; python results/eixo2/make_quant.py
```
