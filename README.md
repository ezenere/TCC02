# Reconhecimento de gestos de mão: arquiteturas, compressão e redução de dados anotados

Trabalho de conclusão de curso. Classificação de 18 gestos de mão do HaGRIDv2 com duas CNNs de conectividade oposta
(ResNet-50 e DenseNet-121), sob um protocolo em quatro eixos:

1. **Arquiteturas** — treino das duas redes com holdout 70/30 por sujeito, 3 seeds.
2. **Compressão** — poda global por magnitude (50–98%) e quantização int8 pós-treinamento, em CPU (fbgemm) e GPU (TensorRT).
3. **Benchmark** — qualidade × custo × latência de todas as células, e escolha da melhor configuração por um critério fixado antes dos resultados.
4. **Redução de dados anotados** — a melhor configuração treinada com 100, 75, 50, 25, 10 e 5% dos dados.

Documentos: [metodologia implementada](docs/METODOLOGIA.md) · [diário de execução](STATUS.md) · [como rodar](docs/RUNBOOK.md) ·
[auditoria de vazamento](results/audit/README.md) · [análise dos erros](results/analise/README.md).

## Resultados até aqui

Métrica primária: **taxa de erro no teste** (83.613 imagens de 6.044 sujeitos nunca vistos no treino) e **razão de erro** em relação
ao baseline da mesma seed — o baseline satura em ~99,8% de acurácia e deltas de acurácia ficam ilegíveis. Média ± desvio de 3 seeds.

| | ResNet-50 | DenseNet-121 | onde |
|---|---|---|---|
| erro do baseline | 0,207 ± 0,009% | 0,197 ± 0,005% | [eixo 1](results/eixo1/README.md) |
| parâmetros / MACs | 23,5 M / 4,11 G | 6,97 M / 2,86 G | |
| poda 90% (razão de erro) | 1,10 ± 0,03 | 1,03 ± 0,04 | [eixo 2](results/eixo2/README.md) |
| poda 95% | 1,28 ± 0,07 | 1,19 ± 0,05 | |
| poda 98% (joelho) | 1,90 ± 0,17 | 1,65 ± 0,05 | |
| int8 CPU (fbgemm) | 1,04 ± 0,07 | 1,24 ± 0,07 | |
| int8 GPU (TensorRT) | 1,08 ± 0,08 | 1,19 ± 0,08 | |
| erro num holdout externo de 19.042 sujeitos | 0,235 ± 0,008% | 0,216 ± 0,005% | [auditoria](results/audit/README.md) |

Leituras principais: as duas arquiteturas empatam em erro (o piso de ~0,2% é em boa parte ruído de rótulo do dataset: 117 imagens são
erradas pelas duas redes, quase sempre com a mesma predição); ambas toleram 90% de poda sem perda mensurável e quebram em 98%; a
quantização int8 é gratuita para a ResNet-50 e custa ~20% a mais de erro à DenseNet-121 nos dois backends; a engine int8 da DenseNet é
mais lenta que a FP16 no TensorRT. Eixos 3 e 4 em andamento — ver o [diário](STATUS.md).

## Estrutura

```
configs/      YAML de cada experimento          src/          pipeline (preprocess, datamodule, train, eval,
data/         brutos e processados (fora do git)               compress/, measure/, reduce_data)
runs/         um diretório por run: config, run_meta,          scripts/      filas, verificações, auditoria, medição
              métricas por época, TensorBoard, metrics.json    results/      CSVs, tabelas e figuras por eixo
tests/        pytest                                           app/          demo de webcam
docs/         metodologia e runbook
```

Cada run grava `config.yaml`, `run_meta.json` (seed, versão e sha256 do manifesto, commit git, versões, GPU), `metrics.csv` por época e
`metrics.json` no teste. Todo número em `results/` é gerado por um script `make_*.py` a partir de `runs/`. Os pesos (`.pt`), os ONNX e as
engines ficam fora do git e são publicados como releases.

## Reprodução

```bash
conda create -n tcc python=3.11 -y && conda activate tcc
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu128
# dados: HaGRIDv2 512px + annotations em data/raw/ (ver docs/RUNBOOK.md)
PYTHONPATH=src python src/validate_raw.py && PYTHONPATH=src python src/preprocess.py     # subconjunto, split, crops, manifesto
PYTHONPATH=src python src/make_inner_split.py && PYTHONPATH=src python src/make_masks.py  # manifestos v2 e v3
make verify      # invariantes e determinismo dos manifestos
make smoke       # pipeline inteiro em miniatura (~3 min)
make eixo1 eixo2 trt eixo3 eixo4 results
```

Hardware de referência: RTX 3080 Ti (12 GB), Ryzen 9 9950X, PyTorch 2.11 + CUDA 12.8, TensorRT 11.3.

## Pesos

| release | conteúdo |
|---|---|
| `ResNet` | baseline de evidência (ResNet-50, 5 épocas): `best.pt`, `last.pt` |
| [`v0.2-eixo1`](https://github.com/ezenere/TCC02/releases/tag/v0.2-eixo1) | eixo 1: 6 × `best.pt` + 6 × `last.pt`, nomeados `<run>__best.pt` / `<run>__last.pt` (1,4 GiB) |
| [`v0.3-eixo2`](https://github.com/ezenere/TCC02/releases/tag/v0.3-eixo2) | eixo 2: 30 modelos podados (`eixo2_prune_<arch>_p<NN>_s<k>__best.pt`, máscaras incluídas) e 10 artefatos int8 de CPU (`<run>__model_int8_fbgemm.pt`), 4,0 GiB. As engines TensorRT não são publicadas: são específicas da GPU e da versão do TensorRT e se reconstroem com `make trt`. |

## Demo

`python app/webcam_demo.py` — escolhe um modelo (PyTorch, int8, ONNX ou TensorRT) e classifica o gesto ao vivo. Ver [app/README.md](app/README.md).

## Dados e licença

HaGRIDv2 (Kapitanov et al., WACV 2024; Nuzhdin et al., arXiv:2412.01508), licença Creative Commons BY-SA. Este repositório não redistribui
as imagens; usa 18 classes e 50% dos sujeitos, amostrados por `user_id`.
