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

Gerado por `make_eixo1.py` (média ± desvio entre seeds; teste avaliado uma vez no `best.pt` de cada run).

<!-- eixo1:table:start -->
| arquitetura | seeds | acurácia (%) | F1 macro (%) | taxa de erro (%) | erros (de 83,613) | razão de erro vs ResNet-50 | s/época | img/s |
|---|---|---|---|---|---|---|---|---|
| DenseNet-121 | 3 | 99.803 ± 0.005 | 99.805 ± 0.005 | 0.1969 ± 0.0048 | 160, 167, 167 | 0.95 | 203 | 863 |
| ResNet-50 | 3 | 99.793 ± 0.009 | 99.795 ± 0.009 | 0.2073 ± 0.0087 | 177, 165, 178 | 1.00 | 164 | 1069 |

Por run (`eixo1_runs.csv`):

| run | best epoch (val) | acc | F1 macro | erro (%) | erros |
|---|---|---|---|---|---|
| `eixo1_densenet121_s0` | 30 | 0.99809 | 0.99810 | 0.1914 | 160 |
| `eixo1_densenet121_s1` | 26 | 0.99800 | 0.99802 | 0.1997 | 167 |
| `eixo1_densenet121_s2` | 22 | 0.99800 | 0.99802 | 0.1997 | 167 |
| `eixo1_resnet50_s0` | 28 | 0.99788 | 0.99791 | 0.2117 | 177 |
| `eixo1_resnet50_s1` | 28 | 0.99803 | 0.99805 | 0.1973 | 165 |
| `eixo1_resnet50_s2` | 21 | 0.99787 | 0.99789 | 0.2129 | 178 |

**sinal na dimensao arquitetura: NAO — |Δ erro| = 0.0104% vs 2×std max = 0.0173%**
<!-- eixo1:table:end -->

## Figuras

- `figures/eixo1_resnet50_curvas.{pdf,png}` — perda fit/val, erro em val, lr, seeds sobrepostas
- `figures/eixo1_densenet121_curvas.{pdf,png}`
- `figures/eixo1_comparativo.{pdf,png}` — erro e F1 em val, média ± std entre seeds

## Custo estático (`cost.json`, cabeça de 18 classes, entrada 224×224)

| arquitetura | parâmetros | MACs | state_dict FP32 |
|---|---|---|---|
| ResNet-50 | 23,54 M | 4,11 G | 90,1 MiB |
| DenseNet-121 | 6,97 M | 2,86 G | 27,2 MiB |

## Leitura dos resultados

1. **Erro equivalente.** DenseNet-121 0,1969 ± 0,0048% vs ResNet-50 0,2073 ± 0,0087% — diferença de 0,010 p.p.,
   abaixo de 2× o maior desvio entre seeds (0,017 p.p.). Os conjuntos de erros por seed se sobrepõem
   (160–167 vs 165–178 em 83.613). Pelo critério fixado antes dos experimentos, **não há sinal na dimensão
   arquitetura**; a razão de erro DenseNet/ResNet de 0,95 é compatível com ruído.
2. **Custo muito diferente para o mesmo erro.** A DenseNet-121 chega ao mesmo erro com 3,4× menos parâmetros,
   30% menos MACs e um artefato 3,3× menor — mas é 24% mais lenta por época nesta GPU (203 vs 164 s), porque a
   concatenação densa gera muitas operações pequenas e mal vetorizadas. Custo teórico (MACs) e custo real
   (tempo) divergem de sentido; o eixo 3 mede os dois.
3. **A época escolhida por val varia (21–30)** e o erro em val oscila ±0,05 p.p. entre épocas vizinhas na fase
   final: a seleção por validação interna importa, e o número de teste depende dela. Nenhum run foi selecionado
   pelo teste.
4. **Instabilidade inicial.** Ambas mostram um salto de erro na época 2 (pico do lr após o warmup de 2 épocas),
   mais forte e mais variável na DenseNet (uma seed chega a 2,2% em val); tudo se recupera até a época 4.
5. **Consequência para os eixos 2–4.** Com o teto saturado (~0,2% de erro, ~170 imagens), deltas de acurácia
   são ilegíveis; a comparação segue por **razão de erro**, e o sinal do benchmark virá da compressão, não da
   arquitetura. Isso reforça a decisão de reportar o erro como métrica primária.

## Proveniência

Os runs de seed 0 começaram antes do primeiro commit do código de treino: o `run_meta.json` deles registra
`40f79c3` com árvore suja e a nota de que o código é idêntico ao de `714b431`. Os demais registram o commit
limpo em que iniciaram. Todos usam o manifesto v2 (`ed5c1eb7…`).

## Reprodução

```bash
scripts/queue.sh eixo1                       # 6 runs seriais, retomáveis
python results/eixo1/make_eixo1.py           # CSVs + sumário
python scripts/plot_curves.py --prefix eixo1 # figuras
python src/measure/cost.py --checkpoint runs/eixo1_resnet50_s0/checkpoints/best.pt
```
Pesos: Release `v0.2-eixo1` (6 × `best.pt` + 6 × `last.pt`).
