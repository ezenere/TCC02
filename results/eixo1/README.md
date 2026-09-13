# Eixo 1 — ResNet-50 vs DenseNet-121 (holdout 70/30 por sujeito)

> Tabelas e figuras desta pasta são geradas por `make_eixo1.py` e `scripts/plot_curves.py`
> a partir de `runs/eixo1_*/`. Nenhum número abaixo é digitado à mão; os valores em
> `eixo1_summary.csv` são a fonte.

## Protocolo

| item | valor |
|---|---|
| dados | HaGRIDv2 512px, 18 classes, subconjunto de 278.715 imagens (manifesto v2, sha `ed5c1eb7…`) |
| split | 70/30 **por sujeito**: teste 83.613 imgs / 6.044 sujeitos, nunca usado para seleção |
| validação interna | 10% dos sujeitos do treino: 19.511 imgs / 2.268 sujeitos (`val`); treino efetivo (`fit`): 175.591 imgs / 11.805 sujeitos |
| entrada | crop 256×256 → random crop 224 + hflip (treino); center crop 224 (avaliação) |
| inicialização | pesos ImageNet (torchvision `DEFAULT`), cabeça nova de 18 classes |
| otimização | SGD nesterov m=0,9, wd 1e-4, lr 0,0375, warmup linear 2 épocas + cosine, 30 épocas, batch 96, AMP fp16, channels_last |
| seleção | `best.pt` = maior F1 macro em `val`; teste avaliado uma vez nesse checkpoint |
| seeds | 0, 1, 2 por arquitetura (init da cabeça, ordem dos dados, augmentation) |
| hardware | RTX 3080 Ti 12 GB, Ryzen 9 9950X, torch 2.11.0+cu128 |

## Resultados no teste

_(preenchido por `make_eixo1.py` — ver `eixo1_summary.csv` e `eixo1_runs.csv`)_

| arquitetura | seeds | acurácia | F1 macro | taxa de erro | razão de erro vs ResNet-50 | tempo/época |
|---|---|---|---|---|---|---|
| ResNet-50 | | | | | 1,00 | |
| DenseNet-121 | | | | | | |

**A diferença entre arquiteturas é sinal?** Critério: |Δ taxa de erro| > 2 × maior desvio entre seeds. _(saída de `make_eixo1.py`)_

## Figuras

- `figures/eixo1_resnet50_curvas.{pdf,png}` — perda fit/val, erro em val, lr, seeds sobrepostas
- `figures/eixo1_densenet121_curvas.{pdf,png}`
- `figures/eixo1_comparativo.{pdf,png}` — erro e F1 em val, média ± std entre seeds

## Observações

- Confusões residuais esperadas (baseline de evidência): `stop` ↔ `palm` (diferem só pela abertura dos dedos), `peace`, `three`.
- O baseline saturou em ~99,8% de acurácia; por isso os eixos 2–4 reportam **taxa de erro e razão de erro**, não deltas de acurácia.

## Reprodução

```bash
scripts/queue.sh eixo1                       # 6 runs seriais, retomáveis
python results/eixo1/make_eixo1.py           # CSVs + sumário
python scripts/plot_curves.py --prefix eixo1 # figuras
python src/measure/cost.py --checkpoint runs/eixo1_resnet50_s0/checkpoints/best.pt
```
Pesos: Release `v0.2-eixo1` (6 × `best.pt` + 6 × `last.pt`).
