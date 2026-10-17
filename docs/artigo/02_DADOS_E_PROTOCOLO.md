# 02: Dados e protocolo experimental

Documento de referência para a seção de Materiais e Métodos do artigo. Descreve o que foi de fato usado e executado. As decisões de protocolo estão consolidadas no arquivo de decisões do projeto (raiz do repositório) e em `docs/METODOLOGIA.md`; os números vêm de `results/subset_stats.csv`, `results/raw_validation.csv`, `data/processed/MANIFESTS.md`, `configs/*.yaml` e dos `run_meta.json` dos runs.

## 1. Dataset

| item | valor |
|---|---|
| fonte | HaGRIDv2, versão 512px (`hagridv2_512.zip`, ~119,4 GB, `min_side = 512`), repositório `hukenovs/hagrid` |
| anotações | `annotations.zip` (`annotations_with_landmarks/`): um JSON por gesto em `train/`, `val/`, `test/`, chaveado pelo nome da imagem sem extensão |
| bboxes | normalizadas (fração 0–1), formato `[x_topleft, y_topleft, w, h]`; válidas em qualquer resolução |
| campos usados | `bboxes`, `labels`, `united_bbox`, `united_label`, `user_id` |
| licença | Creative Commons BY-SA (variante do HaGRID). O repositório não redistribui imagens |
| citação | Kapitanov et al., WACV 2024; Nuzhdin et al., arXiv:2412.01508 |

**Classes.** As 18 classes originais da v1 (paper WACV): `call, dislike, fist, four, like, mute, ok, one, palm, peace, peace_inverted, rock, stop, stop_inverted, three, three2, two_up, two_up_inverted`. As 15 classes novas da v2 ficam fora do escopo. Nenhuma das 18 é bimanual; o tratamento de `united_bbox` previsto para gestos bimanuais não é acionado neste subconjunto. Mãos rotuladas `no_gesture` dentro de uma imagem de gesto são ignoradas.

**Validação dos dados brutos** (`src/validate_raw.py`, `results/raw_validation.csv`): 0 imagens ausentes, 0 bboxes malformadas, 1 imagem descartada por conter apenas mãos `no_gesture`. O pool das 18 classes nos três splits oficiais do HaGRID soma 557.417 imagens.

## 2. Subamostragem de 50% por sujeito

**Por que 50%.** O objeto do estudo é a comparação relativa entre arquiteturas e técnicas, não o teto absoluto de acurácia. Subconjuntos bem amostrados preservam o ordenamento entre modelos (Sorscher et al., NeurIPS 2022; Hestness et al., 2017). A fração de 50% deixa a base de trabalho em 4,2 GB e um treino de 30 épocas em 86–107 min.

**Unidade de amostragem: o sujeito** (`user_id`). Todas as imagens de um sujeito entram ou saem juntas. Isso é o que torna o split subject-disjoint possível depois.

**Algoritmo** (`select_users_balanced`, `src/preprocess.py`), guloso e balanceado por classe:

1. Conta-se, por sujeito, quantas imagens ele tem em cada classe.
2. A meta por classe é `fraction × total_da_classe` (fraction = 0,50).
3. A lista de sujeitos é ordenada explicitamente e depois embaralhada com `random.Random(seed)`. O resultado depende só do conjunto de sujeitos e da seed, nunca da ordem de dicionários ou do sistema de arquivos.
4. Percorre-se a lista; um sujeito é aceito se nenhuma das suas classes ultrapassa a meta inflada pela tolerância corrente. As tolerâncias crescem em passes sucessivos: `[0,01, 0,03, 0,10, 1,0]` (`configs/preprocess.yaml`). O laço para quando o total atinge a meta global.

**Seeds.** Subconjunto: 20250828. Split 70/30: 20250829 (seed + 1), com o mesmo amostrador e `fraction = 0,70` sobre os sujeitos do subconjunto. Validação interna (v2): 20250830 (seed + 2).

**Determinismo.** `scripts/verify_determinism.sh` gera o plano em dois processos com `PYTHONHASHSEED` distintos e exige manifesto byte-idêntico. O sha256 do manifesto v1 é `df8e43a346d095a5f8de9e8ea42cf13740e560a3a631350c3b98cc910c95c2d5`; qualquer mudança no hash invalida os runs que o registram.

**Resultado.** 278.715 imagens de 20.117 sujeitos (50,0% do pool).

## 3. Splits

### 3.1 Holdout 70/30 por sujeito (manifesto v1)

| split | imagens | sujeitos |
|---|---|---|
| treino | 195.102 | 14.073 |
| teste | 83.613 | 6.044 |
| interseção de sujeitos | 0 | |

Fonte: `data/processed/MANIFESTS.md`; verificação por `scripts/verify_manifest.py --version 1` e pela auditoria (`results/audit/leakage_audit.json`, A1). Nunca dividir por imagem: vazamento de sujeito invalidaria os eixos 1 e 4.

### 3.2 Validação interna por sujeitos (manifesto v2)

10% dos **sujeitos** do treino formam `val`; o restante é `fit`. O teste fica idêntico, linha a linha, marcado `inner_split = "test"`.

| inner_split | imagens | sujeitos |
|---|---|---|
| fit | 175.591 | 11.805 |
| val | 19.511 | 2.268 (9,54%–10,10% de cada classe) |
| test | 83.613 | 6.044 |

Gerador: `src/make_inner_split.py` (reutiliza o amostrador balanceado por classe). Sha256: `ed5c1eb749ec9e723055d1ce7284df37403df29dc16c0d81337b36a59dba374d`. É a versão de todos os treinos definitivos dos eixos 1 a 3.

Papel do `val`: escolher `best.pt` (maior F1 macro), parar cedo quando a parada antecipada está ligada, escolher o calibrador da int8. O teste é avaliado uma única vez por run, ao final, no `best.pt`. Isso não altera o holdout 70/30 do protocolo: a validação vem de dentro do treino.

### 3.3 Máscaras de fração aninhadas (manifesto v3)

25 colunas booleanas `frac_{75,50,25,10,5}_s{0..4}`, só em `fit`. Para cada seed K, cada classe de `fit` é permutada uma vez com RNG derivado de `sha256(sha_v2:K)[:8]`; a fração f mantém o prefixo `round(f × n_classe)`. Daí as máscaras são **estratificadas por classe** (±0,1 p.p.) e **aninhadas** (5 ⊂ 10 ⊂ 25 ⊂ 50 ⊂ 75 ⊂ fit) por construção, o que reduz a variância entre pontos da curva. `val` e `test` são idênticos aos da v2 em todas as frações. Sha256: `40f09516596091082e2192c81b6cccdbc0e6f9ff1f224f51b148c574fb6db70a`. Gerador `src/make_masks.py`; testes `tests/test_masks.py`.

| fração | imagens em `fit` |
|---|---|
| 100% | 175.591 |
| 75% | 131.693 |
| 50% | 87.798 |
| 25% | 43.898 |
| 10% | 17.561 |
| 5% | 8.782 |

Fonte: `results/eixo4/eixo4_summary.csv` (coluna `n_fit`).

### 3.4 Versionamento do manifesto

O manifesto é imutável por versão. Toda mudança gera arquivo novo, sha256 novo em `data/processed/MANIFESTS.md`, e cada run grava em `run_meta.json` a versão e o sha usados. Nunca sobrescrever. Todo experimento lê exclusivamente do manifesto; os arquivos de imagem não são tocados após o pré-processamento.

## 4. Pré-processamento (crop)

Config única e imutável: `configs/preprocess.yaml`, `crop.mode: square`.

1. Bbox da mão do gesto (mãos `no_gesture` ignoradas).
2. Margem de **10% em cada lado** da bbox (lado final = 1,2 × maior lado).
3. Expansão do lado menor até igualar o maior, com centro fixo (quadrado; preserva o aspecto).
4. Se o quadrado sai da imagem, a janela é deslocada para dentro: **3.265 casos (1,17%)**.
5. Se ainda assim não cabe, o lado é reduzido a `min(W, H)`: **87 casos (0,031%)**. Perde-se parte da margem, nunca a bbox. Sem pad por replicação (decisão registrada em 13/09/2026).
6. Resize bicúbico para 256×256, JPEG q90.

Base realizada: **4,2 GB** (treino 2,9 / teste 1,3). Folha de validação visual em `results/figures/crop_check.png`; figura de amostras em `results/figures/fig2_amostras_classes.{pdf,png}`.

## 5. Contagens por classe

Fonte: `results/subset_stats.csv` (gerado por `src/preprocess.py`). `users_*` conta sujeitos com ao menos uma imagem daquela classe; um sujeito aparece em várias classes, por isso a soma das colunas excede 14.073 / 6.044.

| classe | treino (imgs) | teste (imgs) | total | % teste | sujeitos treino | sujeitos teste |
|---|---|---|---|---|---|---|
| call | 9.582 | 4.179 | 13.761 | 30,37 | 2.318 | 939 |
| dislike | 11.182 | 4.774 | 15.956 | 29,92 | 3.162 | 1.345 |
| fist | 11.033 | 4.760 | 15.793 | 30,14 | 3.252 | 1.419 |
| four | 11.157 | 4.662 | 15.819 | 29,47 | 2.824 | 1.158 |
| like | 10.928 | 4.718 | 15.646 | 30,15 | 3.016 | 1.341 |
| mute | 11.276 | 5.004 | 16.280 | 30,74 | 3.482 | 1.547 |
| ok | 10.779 | 4.737 | 15.516 | 30,53 | 2.988 | 1.281 |
| one | 11.229 | 4.746 | 15.975 | 29,71 | 2.833 | 1.212 |
| palm | 11.187 | 4.826 | 16.013 | 30,14 | 3.159 | 1.384 |
| peace | 11.116 | 4.830 | 15.946 | 30,29 | 3.209 | 1.448 |
| peace_inverted | 10.341 | 4.333 | 14.674 | 29,53 | 2.306 | 956 |
| rock | 11.294 | 4.853 | 16.147 | 30,06 | 2.839 | 1.246 |
| stop | 10.847 | 4.740 | 15.587 | 30,41 | 3.094 | 1.378 |
| stop_inverted | 10.667 | 4.508 | 15.175 | 29,71 | 2.598 | 1.105 |
| three | 10.835 | 4.561 | 15.396 | 29,62 | 2.571 | 1.079 |
| three2 | 10.402 | 4.322 | 14.724 | 29,35 | 2.506 | 1.026 |
| two_up | 10.761 | 4.673 | 15.434 | 30,28 | 2.936 | 1.236 |
| two_up_inverted | 10.486 | 4.387 | 14.873 | 29,50 | 2.557 | 1.059 |
| **total** | **195.102** | **83.613** | **278.715** | 30,00 | | |

Fonte: `results/subset_stats.csv`. A fração de teste fica entre 29,35% e 30,74% em todas as classes. As classes são aproximadamente balanceadas (13,8 a 16,3 mil imagens).

## 6. Entrada e augmentation

| item | valor |
|---|---|
| treino | `RandomCrop(224)` a partir do crop armazenado de 256×256 + `RandomHorizontalFlip(p=0,5)` |
| avaliação | `CenterCrop(224)` |
| normalização | média e desvio do ImageNet |
| proibido | flip vertical, rotações, qualquer augmentation além das duas acima |

**Por que o hflip é seguro.** Os rótulos do HaGRID são agnósticos à mão (direita ou esquerda), e os pares `*_inverted` são rotações de ~180°, não espelhamentos.

**Por que rotação é proibida.** Uma rotação de ~180° converte `peace` em `peace_inverted`, `stop` em `stop_inverted`, `two_up` em `two_up_inverted`, e vice-versa: o rótulo seria corrompido silenciosamente. O teste `tests/test_datamodule.py::test_train_transform_has_no_rotation_or_vflip` garante que o pipeline não contém essas operações.

## 7. Receita de treino (eixo 1, herdada pelos demais)

| item | valor | fonte |
|---|---|---|
| arquiteturas | `torchvision.models.resnet50` e `densenet121` | `configs/train_<arch>.yaml` |
| inicialização | pesos ImageNet (`DEFAULT`), cabeça linear nova de 18 classes | |
| otimizador | SGD, momentum 0,9, Nesterov, weight decay 1e-4 | |
| taxa de aprendizado | 0,0375 = 0,1 × 96/256 (escala linear de Goyal et al., 2017, a partir de 0,1 @ batch 256) | |
| agenda | warmup linear de 2 épocas, depois cosseno até zero na época 30 | |
| épocas | 30 fixas (parada antecipada desligada, `early_stopping_patience: 0`) | |
| batch | 96 nas duas arquiteturas (AMP: ResNet-50 ~4,3 GiB, DenseNet-121 ~6,2 GiB de VRAM) | |
| precisão | AMP fp16 (`torch.autocast` + GradScaler), `channels_last`, `cudnn.benchmark = True` | |
| dataloader | 16 workers, `prefetch_factor 6`, `persistent_workers` desligado (necessário para o `--resume` replicar a augmentation) | |
| label smoothing | 0 | |
| seleção | `best.pt` = maior F1 macro em `val`, avaliado a cada época | |
| teste | avaliado **uma única vez**, ao final, no `best.pt` (`src/train.py --eval-test`; reavaliável por `src/eval.py`) | |
| retomada | `--resume` continua de `last.pt` com a mesma ordem de lotes e a mesma augmentation (teste: `test_resume_replays_the_same_run`, |Δloss| < 5e-3) | |

Variações por eixo:

| eixo | ponto de partida | lr | warmup | épocas | observação |
|---|---|---|---|---|---|
| 1 | ImageNet | 0,0375 | 2 | 30 | receita base |
| 2 (poda depois) | `best.pt` do eixo 1 da mesma seed | 0,00375 (0,1×) | 0 | 5 | máscaras reaplicadas a cada `optimizer.step()` |
| 2b (poda antes) | ImageNet podado | 0,0375 | 2 | 30 | cabeça densa (`exclude_head: true`) |
| 2c (LR rewinding) | `best.pt` do eixo 1 podado | 0,0375 | 2 | 30 | cabeça podável como no eixo 2 |
| 4 | ImageNet, `fit` restrito à máscara | 0,0375 | 2 | 30 | fração 100% é run próprio com a mesma receita |

Fonte: `configs/prune_<arch>.yaml`, `configs/prune_first_<arch>.yaml`, `configs/prune_rewind_<arch>.yaml`, `src/reduce_data.py`.

## 8. Métricas

| métrica | definição | uso |
|---|---|---|
| acurácia | acertos / 83.613 | reportada |
| F1 macro | média do F1 das 18 classes | critério de seleção em `val`; reportado no teste |
| taxa de erro | 1 − acurácia | métrica primária; também em número absoluto de erros |
| razão de erro | erro da célula ÷ erro do baseline **da mesma seed** | métrica primária de degradação nos eixos 2 a 4 |
| matriz de confusão | 18 × 18, `confusion_matrix.npy` por run | análise de erros |
| por classe | precisão, recall, F1 em `metrics.json` | eixo 4 (quais gestos degradam primeiro) |

**Por que razão de erro.** O baseline satura em ~99,8% de acurácia (~170 erros em 83.613). Deltas de acurácia da ordem de 0,01 p.p. são ilegíveis e dependem da terceira casa decimal. A razão entre taxas de erro torna a degradação legível ("1,5× os erros do baseline") e é comparável entre arquiteturas que partem de pisos diferentes. O eixo 3 é formulado como **localização do joelho**: o primeiro nível de compressão em que a razão excede 1,5×.

**Critério de sinal entre arquiteturas.** A diferença só é considerada sinal se exceder 2× o maior desvio-padrão entre seeds.

## 9. Seeds e repetições

- Seeds de dados: 20250828 (subconjunto), 20250829 (split), 20250830 (val), sha-derivadas (máscaras v3).
- Seeds de treino: 0, 1, 2. A seed governa a inicialização da cabeça, a ordem dos dados e a augmentation. Em cada eixo, a célula de seed k parte do modelo do eixo 1 da seed k; a razão de erro é calculada contra o baseline da mesma seed.
- Reporta-se média ± desvio-padrão amostral entre seeds, e os erros por seed entre parênteses.
- Células com uma seed (poda+int8, eixo 2b em 50/70/90%, extra B) são marcadas como tal em todas as tabelas.
- `cudnn.benchmark = True` deixa não-determinismo residual nos kernels de backward; isso está documentado no `run_meta.json` de cada run.

## 10. Rastreabilidade por run

Cada `runs/<run>/` contém: `config.yaml`, `config_resolved.yaml`, `run_meta.json` (seed, versão e sha256 do manifesto, commit git e se a árvore estava suja, versões de python/torch/torchvision/cuda/cudnn, GPU, hostname, horários, `best_epoch`, `stopped_early`), `metrics.csv` por época (perdas, acurácia e F1 em val, lr, tempo, throughput), `tb/` (TensorBoard), `metrics_val_best.json`, `metrics.json` (teste), `confusion_matrix.npy`, `cost.json`, `train.log`. Os checkpoints `best.pt`/`last.pt` ficam fora do git e são publicados como releases. Um run sem `run_meta.json` completo é inválido.

Proveniência conhecida: os runs de seed 0 do eixo 1 registram `git_commit = 40f79c3` com árvore suja e a nota de que o código é idêntico ao commit `714b431` (commitado durante o segundo run). Os demais registram commit limpo.

## 11. Hardware e versões

| item | valor |
|---|---|
| GPU | NVIDIA GeForce RTX 3080 Ti, 12 GB |
| CPU | AMD Ryzen 9 9950X, 16 núcleos |
| SO | Arch Linux |
| ambiente | conda `tcc`, Python 3.11.15 |
| torch / torchvision | 2.11.0+cu128 / 0.26.0+cu128 |
| CUDA / cuDNN | 12.8 / 9.19 |
| TensorRT | 11.3.0.99 |
| ONNX / ONNX Runtime | 1.22.0 / 1.30.0 |
| fvcore (MACs) | 0.1.5 |
| `requirements.txt` | gerado por `scripts/freeze.sh`, 152 pins |

Desempenho medido no treino (batch 96, AMP, `channels_last`): ResNet-50 164 s/época (1.069 img/s); DenseNet-121 203 s/época (863 img/s). O dataloader com 16 workers entrega ~6.500 img/s; o treino é limitado pela GPU.

## 12. Testes automatizados

`tests/` (26 testes, `pytest`): sujeitos disjuntos no manifesto v2 real e em frames sintéticos; máscara de fração restringe só `fit`; ausência de rotação e flip vertical; embaralhamento de rótulos afeta só `fit`; esparsidade obtida = alvo ± 0,1 p.p. em 50/70/90%; limiar global; máscaras sobrevivem a passos do otimizador; round-trip de checkpoint preserva máscaras; `--resume` replica o treino; harness de latência. `make smoke` roda o pipeline inteiro em miniatura (treino → poda → custo → int8 → latência → ONNX) em ~2–3 min.
