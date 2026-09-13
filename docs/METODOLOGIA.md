# Metodologia implementada

Este documento descreve o que foi de fato implementado e executado, eixo por eixo,
com os scripts que produzem cada número reportado. As decisões de protocolo estão
consolidadas no arquivo de decisões do projeto; aqui está a execução.

## 0. Dados e pré-processamento (comum a todos os eixos)

**Fonte.** HaGRIDv2, versão 512px (`min_side = 512`), anotações com bounding boxes
normalizadas `[x, y, w, h]` e `user_id` por imagem (Kapitanov et al., 2024; Nuzhdin et al., 2024).
Licença CC BY-SA.

**Subconjunto.** As 18 classes originais da v1 (`call, dislike, fist, four, like, mute, ok, one,
palm, peace, peace_inverted, rock, stop, stop_inverted, three, three2, two_up, two_up_inverted`).
Do pool de 557.417 imagens dessas classes, 50% foi amostrado **por sujeito** (`user_id`): um
sujeito entra ou sai por inteiro, com balanceamento guloso por classe. Resultado: **278.715
imagens, 20.117 sujeitos**. Validação dos dados brutos (`src/validate_raw.py`): 0 imagens
ausentes, 0 bboxes malformadas, 1 imagem descartada por conter apenas mãos `no_gesture`.

**Split.** Holdout 70/30 **por sujeito** (`src/preprocess.py`): treino 195.102 imgs / 14.073
sujeitos; teste 83.613 imgs / 6.044 sujeitos; interseção de sujeitos = 0 (verificado por assert e
por script independente). Todas as classes ficam entre 29,35% e 30,74% no teste.
Do treino, 10% dos **sujeitos** formam a validação interna (`src/make_inner_split.py`):
`fit` 175.591 imgs / 11.805 sujeitos; `val` 19.511 imgs / 2.268 sujeitos (9,54%–10,10% de cada classe).

**Crop.** Bbox da mão do gesto (mãos `no_gesture` ignoradas), margem de 10% em cada lado,
expansão para quadrado com centro fixo, janela deslocada para dentro da imagem quando necessário
(3.265 casos, 1,17%) e lado reduzido a `min(W,H)` se ainda não couber (87 casos, 0,031%; a bbox
nunca é cortada). Resize bicúbico para 256×256, JPEG q90. Base: 4,2 GB.

**Manifesto.** `manifest.csv` (v1: `image_path, label, user_id, split`) e `manifest_v2.csv`
(+ `inner_split ∈ {fit, val, test}`). Versões imutáveis, sha256 registrado em
`data/processed/MANIFESTS.md`; cada run grava a versão e o sha usados. Todo experimento lê
exclusivamente do manifesto. Determinismo: o mesmo manifesto é gerado byte a byte em processos
independentes (`scripts/verify_determinism.sh`, `scripts/verify_manifest.py`).

## 1. Eixo 1 — Arquiteturas (ResNet-50 vs DenseNet-121)

**Pergunta.** Com o mesmo orçamento de treino e o mesmo protocolo, as duas estratégias de
conectividade (soma residual vs concatenação densa) diferem em erro no teste mais do que o
ruído entre seeds?

**Modelos.** `torchvision.models.resnet50` e `densenet121`, pesos ImageNet como inicialização,
cabeça linear nova de 18 classes (ResNet-50: 23,54 M parâmetros, 4,11 GMACs @224;
DenseNet-121: ≈7,0 M, ≈2,9 GMACs — `src/measure/cost.py`).

**Entrada e augmentation.** Random crop 224×224 a partir do crop armazenado de 256×256 e
flip horizontal (p=0,5). Nada mais: rotações são proibidas porque uma rotação de ~180°
converte classes nos seus pares `*_inverted`. Avaliação com center crop 224. Normalização ImageNet.

**Otimização.** SGD com momentum 0,9 (Nesterov), weight decay 1e-4, lr 0,0375 (escala linear
de 0,1 @ batch 256 para batch 96; Goyal et al., 2017), warmup linear de 2 épocas seguido de
cosine até zero, 30 épocas, batch 96 para ambas as arquiteturas, AMP fp16 com GradScaler,
`channels_last`, `cudnn.benchmark` (não-determinismo residual documentado no `run_meta.json`).

**Seleção e avaliação.** A cada época o modelo é avaliado em `val`; `best.pt` é o checkpoint de
maior F1 macro em `val`. O teste é avaliado **uma única vez**, ao final, nesse checkpoint
(`src/train.py --eval-test`; reavaliável por `src/eval.py`). Métricas: acurácia, F1 macro,
taxa de erro (1 − acurácia), matriz de confusão e métricas por classe (`src/metrics.py`).

**Repetições.** 3 seeds (0, 1, 2) por arquitetura; a seed governa a inicialização da cabeça, a
ordem dos dados e a augmentation. Reporta-se média ± desvio; a diferença entre arquiteturas é
considerada sinal se exceder 2× o maior desvio entre seeds.

**Rastreabilidade.** Cada run em `runs/eixo1_<arch>_s<seed>/` contém `config.yaml`,
`config_resolved.yaml`, `run_meta.json` (seed, versão e sha do manifesto, commit git, versões,
GPU), `metrics.csv` por época, TensorBoard, `metrics.json` (teste), `metrics_val_best.json`,
`cost.json`. Pesos (`best.pt`, `last.pt`) publicados como Release.

**Scripts que geram os números.** `scripts/queue.sh eixo1` (execução),
`results/eixo1/make_eixo1.py` (`eixo1_runs.csv`, `eixo1_summary.csv`),
`scripts/plot_curves.py` (figuras), `src/measure/cost.py` (custo estático).

## 2. Eixo 2 — Compressão
_(a preencher: poda global por magnitude com fine-tuning; PTQ int8 em CPU (fbgemm) e GPU (TensorRT))_

## 3. Eixo 3 — Benchmark
_(a preencher)_

## 4. Eixo 4 — Redução de dados anotados
_(a preencher)_
