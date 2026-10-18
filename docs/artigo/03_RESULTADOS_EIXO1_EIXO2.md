# 03: Resultados: eixo 1 (arquiteturas) e eixo 2 (compressão)

Teste = 83.613 imagens de 6.044 sujeitos nunca vistos no treino, avaliado uma única vez por run, no `best.pt` escolhido por F1 macro em `val`. Média ± desvio-padrão de 3 seeds (0, 1, 2), salvo indicação. **Razão de erro** = taxa de erro da célula ÷ taxa de erro do baseline (eixo 1) da mesma seed. Todos os CSVs citados são gerados por scripts `make_*.py` a partir de `runs/` (`make results`).

## 1. Eixo 1: ResNet-50 × DenseNet-121

### 1.1 Baseline de evidência (agosto)

ResNet-50, 5 épocas, manifesto v1, última época reportada (`runs/resnet50_baseline_5ep/`, 29/08/2026): acurácia **99,79%**, F1 macro **99,79%**, 185,7 s/época. Confusão residual concentrada em `stop` ↔ `palm`. Serviu para mostrar que o teto estava saturado e motivou a adoção da razão de erro como métrica primária. Pesos na release `ResNet`.

### 1.2 Treinos definitivos (30 épocas, 3 seeds)

| arquitetura | acurácia (%) | F1 macro (%) | taxa de erro (%) | erros (s0, s1, s2) | razão vs ResNet-50 | s/época | img/s |
|---|---|---|---|---|---|---|---|
| ResNet-50 | 99,793 ± 0,009 | 99,795 ± 0,009 | 0,2073 ± 0,0087 | 177, 165, 178 | 1,00 | 164 | 1.069 |
| DenseNet-121 | 99,803 ± 0,005 | 99,805 ± 0,005 | 0,1969 ± 0,0048 | 160, 167, 167 | 0,95 | 203 | 863 |

Fonte: `results/eixo1/eixo1_summary.csv`.

Por run:

| run | época do `best.pt` (val) | acurácia | F1 macro | erro (%) | erros | duração |
|---|---|---|---|---|---|---|
| `eixo1_resnet50_s0` | 28 | 0,99788 | 0,99791 | 0,2117 | 177 | 86 min |
| `eixo1_resnet50_s1` | 28 | 0,99803 | 0,99805 | 0,1973 | 165 | 86 min |
| `eixo1_resnet50_s2` | 21 | 0,99787 | 0,99789 | 0,2129 | 178 | 87 min |
| `eixo1_densenet121_s0` | 30 | 0,99809 | 0,99810 | 0,1914 | 160 | 107 min |
| `eixo1_densenet121_s1` | 26 | 0,99800 | 0,99802 | 0,1997 | 167 | 106 min |
| `eixo1_densenet121_s2` | 22 | 0,99800 | 0,99802 | 0,1997 | 167 | 108 min |

Fonte: `results/eixo1/eixo1_runs.csv`; durações em `runs/queue_eixo1.log`.

**Leitura.** |Δ erro| = 0,0104 p.p. contra 2× o maior desvio entre seeds = 0,0173 p.p.: **não há sinal na dimensão arquitetura**. Os conjuntos de erros por seed se sobrepõem (160–167 vs 165–178). A época escolhida por `val` varia de 21 a 30, e o erro em `val` oscila ±0,05 p.p. entre épocas vizinhas na fase final: a seleção por validação interna importa. Ambas mostram um salto de erro na época 2 (pico do lr após o warmup), mais forte na DenseNet (uma seed chega a 2,2% em val), recuperado até a época 4. Figuras: `results/eixo1/figures/eixo1_{resnet50,densenet121}_curvas` e `eixo1_comparativo`.

### 1.3 Custo estático

| arquitetura | parâmetros (cabeça de 18 classes) | MACs @224 (fvcore, 1 MAC = 1) | `state_dict` FP32 | tempo por época |
|---|---|---|---|---|
| ResNet-50 | 23,54 M | 4,11 G | 90,1 MiB (gzip 83,7 MiB) | 164 s |
| DenseNet-121 | 6,97 M | 2,86 G | 27,2 MiB | 203 s |

Fonte: `runs/eixo1_<arch>_s0/cost.json` (`src/measure/cost.py`); `results/eixo1/README.md`.

A DenseNet-121 chega ao mesmo erro com 3,4× menos parâmetros, 30% menos MACs e um artefato 3,3× menor, mas é 24% mais lenta por época nesta GPU: a concatenação densa gera muitas operações pequenas e mal vetorizadas. Custo teórico (MACs) e custo real (tempo) divergem de sentido; o eixo 3 mede os dois.

Observação para o texto: "ResNet-50 ≈ 25,6 M parâmetros" vale para a cabeça ImageNet de 1.000 classes; com 18 classes o total é 23,5 M. DenseNet-121: 7,0 M (vs 8,0 M).

## 2. Eixo 2: Poda global por magnitude + fine-tuning (protocolo principal)

**Técnica** (`src/compress/pruning.py`): poda global, não-estruturada, por magnitude. Um único limiar de |w| sobre todos os pesos de `Conv2d` e `Linear` (biases e BatchNorm excluídos). O limiar global deixa a esparsidade se distribuir de forma desigual entre camadas. Máscaras booleanas explícitas, persistidas no checkpoint (`prune_masks`), reaplicadas após **cada** `optimizer.step()`; ao final, assert de esparsidade obtida = alvo ± 0,1 p.p. Não se usa `torch.nn.utils.prune` (os hooks `weight_orig`/`weight_mask` contaminam exportação ONNX e quantização).

**Protocolo.** Para cada arquitetura e seed: `best.pt` do eixo 1 da mesma seed → poda no nível s → fine-tuning de 5 épocas em `fit` (SGD, lr 0,00375 = 0,1× o inicial, cosseno sem warmup) → seleção por F1 em `val` → teste. Níveis iniciais 50/70/90%; regra automática: se a razão em 90% ficar < 2×, estender a 95 e 98%. A extensão foi disparada (90%: ResNet 1,07×, DenseNet 1,02× na seed 0).

### 2.1 Razão de erro por nível (3 seeds)

| esparsidade | ResNet-50 razão | erros R (s0, s1, s2) | DenseNet-121 razão | erros D (s0, s1, s2) |
|---|---|---|---|---|
| baseline | 1,00 | 177, 165, 178 | 1,00 | 160, 167, 167 |
| 50% | 0,97 ± 0,03 | 166, 164, 174 | 0,97 ± 0,06 | 159, 168, 151 |
| 70% | 0,98 ± 0,02 | 175, 163, 171 | 0,98 ± 0,05 | 158, 171, 156 |
| 90% | 1,10 ± 0,03 | 190, 187, 192 | 1,03 ± 0,04 | 163, 178, 166 |
| 95% | 1,28 ± 0,07 | 230, 221, 214 | 1,19 ± 0,05 | 193, 206, 190 |
| 98% | **1,90 ± 0,17** | 323, 345, 316 | **1,65 ± 0,05** | 262, 285, 269 |

Fonte: `results/eixo2/pruning_summary.csv`, `results/eixo2/pruning_runs.csv`, `results/eixo2/prune_compare_summary.csv` (coluna `errors`).

### 2.2 Acurácia, parâmetros efetivos e artefato

| esparsidade | acc R (%) | acc D (%) | não-nulos R | não-nulos D | gzip R (MiB, × base) | gzip D (MiB, × base) |
|---|---|---|---|---|---|---|
| 50% | 99,799 | 99,809 | 11,80 M | 3,53 M | 51,0 (0,61) | 15,5 (0,61) |
| 70% | 99,797 | 99,807 | 7,10 M | 2,15 M | 34,8 (0,42) | 10,8 (0,43) |
| 90% | 99,773 | 99,798 | 2,40 M | 0,77 M | 17,2 (0,21) | 5,7 (0,22) |
| 95% | 99,735 | 99,765 | 1,23 M | 0,43 M | 12,1 (0,14) | 4,2 (0,17) |
| 98% | 99,608 | 99,675 | 0,52 M | 0,22 M | 8,8 (0,10) | 3,2 (0,13) |

Fonte: `results/eixo2/pruning_summary.csv`. O gzip do `state_dict` é o único custo que a esparsidade não-estruturada reduz de fato; MACs e latência não mudam (seção 5 e documento 04).

**Leitura.** Até 90% a degradação fica dentro do ruído entre seeds nas duas arquiteturas. 95% já é visível, mas dentro do orçamento de 1,5× do critério do eixo 3. **Joelho (primeiro nível com razão > 1,5×): 98% nas duas.** A DenseNet tolera cada nível um pouco melhor que a ResNet apesar de partir de 3,4× menos parâmetros. Custo de GPU: 30 runs em ~8,5 h (14–15 min por run de ResNet, 18–19 min de DenseNet).

**Limitação registrada: poda estruturada em DenseNet.** A poda estruturada (remoção de filtros inteiros, a que reduz MACs e latência) é direta na ResNet, mas na DenseNet cada camada concatena as saídas de todas as anteriores do bloco; remover um canal altera a entrada de todas as seguintes e das transições. Por isso o protocolo usa poda não-estruturada nas duas, comparável entre elas. É uma constatação a reportar, não uma falha de implementação.

## 3. Eixo 2b, 2c e 2d: podar antes, depois ou sem retreino? Quatro protocolos

Pedido do orientador (19/09/2026): testar a poda **antes** do treino. Três protocolos com retreino foram comparados; em 17/10/2026 juntou-se um quarto, **sem retreino**, que serve de referência de quanto o fine-tuning recupera:

| protocolo | ponto de partida | poda calculada sobre | treino após a poda | cabeça | configs | runs |
|---|---|---|---|---|---|---|
| **Eixo 2: depois + 5 ép.** | `best.pt` do eixo 1 (treinado) | pesos treinados nos gestos | 5 épocas, lr 0,00375, sem warmup | podável | `prune_<arch>.yaml` | `eixo2_prune_*` |
| **Eixo 2b: antes + 30 ép.** | pesos ImageNet | pesos ImageNet | 30 épocas, receita completa do eixo 1 | **densa** (`exclude_head: true`; é aleatória, a magnitude não carrega informação; 0,15% dos parâmetros da ResNet-50) | `prune_first_<arch>.yaml` | `eixo2b_prunefirst_*` |
| **Eixo 2c: depois + 30 ép. (controle, LR rewinding)** | `best.pt` do eixo 1 | pesos treinados nos gestos | 30 épocas, receita completa (Renda et al., 2020) | podável | `prune_rewind_<arch>.yaml` | `eixo2c_rewind_*` |
| **Eixo 2d: sem retreino (podar e avaliar)** | `best.pt` do eixo 1 | pesos treinados nos gestos | **nenhum** (0 épocas; avaliação imediata no teste) | podável | (sem config; `src/compress/prune_eval.py --all`) | `eixo2d_noft_*` |

Por que o controle existe: a comparação 2 × 2b confunde a **ordem** da poda com o **orçamento** de retreino da rede esparsa (5 × 30 épocas). O 2c poda o modelo treinado e dá a ele a agenda completa; isola os dois fatores. O 2d responde à pergunta anterior a todas: quanto do erro introduzido pela poda o retreino de fato recupera. Nos três protocolos com retreino, máscaras fixas reaplicadas a cada passo; em todos, mesma métrica, mesmas seeds.

### 3.1 Eixo 2d: poda sem retreino (podar e avaliar)

**Procedimento** (`src/compress/prune_eval.py --all`, 17/10/2026). Para cada arquitetura, seed e nível: carrega-se o `best.pt` do eixo 1 da mesma seed, calculam-se as **mesmas máscaras globais por magnitude** da seção 2 (um limiar de |w| sobre todos os `Conv2d` e `Linear`, **cabeça incluída**; biases e BatchNorm excluídos), zeram-se os pesos e o modelo é avaliado **imediatamente no teste** (manifesto v2, AMP fp16, `channels_last`, mesmo caminho de `src/eval.py`). Nenhum passo de otimizador: **0 épocas**. Runs `runs/eixo2d_noft_<arch>_p<NN>_s<k>/`, com `metrics.json`, `run_meta.json`, `confusion_matrix.npy`, `cost.json` e o checkpoint podado no formato dos demais eixos. 30 células (2 arquiteturas × 5 níveis × 3 seeds), 0 falhas, ~40 s de GPU por célula (~25 min no total).

| esparsidade | erros R (s0, s1, s2) | acurácia R (%) | razão R | erros D (s0, s1, s2) | acurácia D (%) | razão D |
|---|---|---|---|---|---|---|
| baseline | 177, 165, 178 | 99,79 | 1,00 | 160, 167, 167 | 99,80 | 1,00 |
| 50% | 177, 181, 190 | 99,78 | **1,05 ± 0,05** | 161, 162, 177 | 99,80 | **1,01 ± 0,05** |
| 70% | 326, 377, 328 | 99,59 | **1,99 ± 0,26** | 217, 215, 205 | 99,75 | **1,29 ± 0,06** |
| 90% | 66.388, 77.983, 70.815 | 14,2 (6,7–20,6) | 415 ± 51 | 76.718, 71.898, 66.711 | 14,2 (8,2–20,2) | 436 ± 40 |
| 95% | 78.755, 76.128, 78.760 | 6,9 (5,8–9,0) | 450 ± 10 | 78.695, 78.760, 78.951 | 5,8 (5,6–5,9) | 479 ± 11 |
| 98% | 78.853, 78.760, 78.761 | 5,8 (5,7–5,8) | 455 ± 19 | 79.434, 78.873, 78.951 | 5,4 (5,0–5,7) | 481 ± 14 |

R = ResNet-50, D = DenseNet-121. Razão de erro vs o baseline denso da mesma seed (177/165/178 e 160/167/167 erros em 83.613 imagens), média ± desvio de 3 seeds; acurácia média das 3 seeds (faixa entre parênteses onde a variação é grande). Acaso para 18 classes = 1/18 = 5,6%. Fonte: `results/eixo2/prune_compare_runs.csv` e `prune_compare_summary.csv` (método `sem retreino (poda → avaliar)`).

**Leitura.** Sem nenhuma época de treino, a poda de **50% é gratuita** nas duas redes (1,05× / 1,01×, dentro do ruído entre seeds): metade dos pesos de menor magnitude não contribui para a decisão. A **70%** a rede ainda funciona mas já paga: **1,99× na ResNet-50 e 1,29× na DenseNet-121**. A **90% a rede quebra** (~14% de acurácia nas duas, com variação grande entre seeds: 6,7–20,6%), e a **95% e 98% o modelo responde ao acaso** (5–9% de acurácia; 1/18 = 5,6%). Posto ao lado da seção 2.1: o fine-tuning de 5 épocas **recupera tudo até 90%** (1,10× / 1,03×) e quase tudo a 95% (1,28× / 1,19×). O que a poda remove sem retreino é a calibração conjunta dos pesos restantes com o BatchNorm e a cabeça, não a capacidade da sub-rede: 5 épocas a lr 0,1× bastam para reaprendê-la. Em suma, **acima de 50% o retreino é obrigatório**, e o "joelho" de 98% da seção 2 é o ponto em que nem 5 épocas bastam. A DenseNet-121 é um pouco mais robusta à poda crua que a ResNet-50 (70%: 1,29× contra 1,99×; 50%: 1,01× contra 1,05×), coerente com a vantagem leve que mantém após o fine-tuning.

### 3.2 Tabela completa (razão de erro; erros por seed entre parênteses)

| arquitetura | esparsidade | sem retreino (eixo 2d) | depois + 5 ép. (eixo 2) | antes + 30 ép. (eixo 2b) | depois + 30 ép. (eixo 2c) |
|---|---|---|---|---|---|
| ResNet-50 | 50% | 1,05 ± 0,05 (177, 181, 190) | 0,97 ± 0,03 (166, 164, 174) | 0,99 (176) *1 seed* | — |
| ResNet-50 | 70% | 1,99 ± 0,26 (326, 377, 328) | 0,98 ± 0,02 (175, 163, 171) | 1,12 (199) *1 seed* | — |
| ResNet-50 | 90% | 415 ± 51 (66.388, 77.983, 70.815) | 1,10 ± 0,03 (190, 187, 192) | 1,08 (191) *1 seed* | — |
| ResNet-50 | 95% | 450 ± 10 (78.755, 76.128, 78.760) | 1,28 ± 0,07 (230, 221, 214) | 1,19 ± 0,05 (203, 206, 209) | 1,14 ± 0,07 (208, 197, 188) |
| ResNet-50 | 98% | 455 ± 19 (78.853, 78.760, 78.761) | 1,90 ± 0,17 (323, 345, 316) | 1,40 ± 0,16 (241, 261, 225) | **1,12 ± 0,02** (194, 186, 202) |
| DenseNet-121 | 50% | 1,01 ± 0,05 (161, 162, 177) | 0,97 ± 0,06 (159, 168, 151) | 1,06 (170) *1 seed* | — |
| DenseNet-121 | 70% | 1,29 ± 0,06 (217, 215, 205) | 0,98 ± 0,05 (158, 171, 156) | 1,07 (171) *1 seed* | — |
| DenseNet-121 | 90% | 436 ± 40 (76.718, 71.898, 66.711) | 1,03 ± 0,04 (163, 178, 166) | 1,11 (178) *1 seed* | — |
| DenseNet-121 | 95% | 479 ± 11 (78.695, 78.760, 78.951) | 1,19 ± 0,05 (193, 206, 190) | 1,17 ± 0,08 (202, 190, 186) | 1,10 ± 0,15 (204, 169, 171) |
| DenseNet-121 | 98% | 481 ± 14 (79.434, 78.873, 78.951) | 1,65 ± 0,05 (262, 285, 269) | 1,54 ± 0,09 (263, 246, 251) | **1,23 ± 0,09** (213, 198, 194) |

Fonte: `results/eixo2/prune_compare_summary.csv`, `results/eixo2/prune_compare_runs.csv`. Figura: `results/eixo2/figures/prune_compare` (eixo da razão em escala log, por causa da coluna sem retreino).

### 3.3 Custo de GPU por modelo podado

| arquitetura | sem retreino | depois + 5 ép. | antes + 30 ép. | depois + 30 ép. |
|---|---|---|---|---|
| ResNet-50 | ~40 s (só avaliação; + 86 min do treino denso, reaproveitado) | 14 min (+ 86 min do treino denso, reaproveitado para todos os níveis) | 81–85 min por nível | 84–85 min por nível (+ treino denso) |
| DenseNet-121 | ~35 s (+ 107 min do denso) | 17 min (+ 107 min do denso) | 99–105 min por nível | 102–105 min por nível (+ denso) |

Fonte: coluna `gpu_min` de `results/eixo2/prune_compare_summary.csv` (0 para o eixo 2d, que não treina) e `runs/queue_eixo2d.log` (tempo de avaliação por célula). "Antes" exige um treino completo por nível de esparsidade; "depois" reaproveita um único treino denso.

### 3.4 Conclusão: os quatro protocolos lado a lado

| arquitetura | esparsidade | sem retreino (2d) | depois + 5 ép. (2) | antes + 30 ép. (2b) | depois + 30 ép. (2c) |
|---|---|---|---|---|---|
| ResNet-50 | 90% | 415 ± 51 (acurácia 14%) | 1,10 ± 0,03 | 1,08 *(1 seed)* | — |
| ResNet-50 | 95% | 450 ± 10 (acaso) | 1,28 ± 0,07 | 1,19 ± 0,05 | **1,14 ± 0,07** |
| ResNet-50 | 98% | 455 ± 19 (acaso) | 1,90 ± 0,17 | 1,40 ± 0,16 | **1,12 ± 0,02** |
| DenseNet-121 | 90% | 436 ± 40 (acurácia 14%) | 1,03 ± 0,04 | 1,11 *(1 seed)* | — |
| DenseNet-121 | 95% | 479 ± 11 (acaso) | 1,19 ± 0,05 | 1,17 ± 0,08 | **1,10 ± 0,15** |
| DenseNet-121 | 98% | 481 ± 14 (acaso) | 1,65 ± 0,05 | 1,54 ± 0,09 | **1,23 ± 0,09** |

Razão de erro vs baseline denso da mesma seed; o eixo 2c não foi executado em 90% (nível em que o fine-tuning de 5 épocas já fica dentro do ruído).

**Sem retreino, a poda acima de 50% não é utilizável; com retreino, o que decide é o orçamento, não a ordem da poda.** Em 90% a rede crua responde com ~14% de acurácia e 5 épocas de fine-tuning a devolvem ao erro do baseline (1,10× / 1,03×). A 98%, retreinar 30 épocas depois da poda dá 1,12 ± 0,02× (ResNet-50) e 1,23 ± 0,09× (DenseNet-121): o joelho desaparece nas duas. Podar antes do treino fica no meio (1,40 ± 0,16× / 1,54 ± 0,09×), e o fine-tuning de 5 épocas é o pior dos protocolos com retreino (1,90× / 1,65×). Com orçamento igual, podar depois do treino é igual ou melhor que podar antes. A 95% os três métodos com retreino ficam dentro do ruído entre seeds (1,10–1,28). **O "joelho em 98%" do eixo 2 é uma propriedade do protocolo de 5 épocas, não das redes.** Na seed 0 isolada, a ResNet-50 podada antes parecia ganhar nas esparsidades extremas (98%: 1,36× vs 1,90×); com 3 seeds e o controle, o efeito ficou explicado pelo orçamento. A DenseNet-121 é um pouco mais robusta que a ResNet-50 à poda crua (70%: 1,29× vs 1,99×) e ao fine-tuning curto (98%: 1,65× vs 1,90×), mas um pouco pior nos protocolos de 30 épocas (98%: 1,54× vs 1,40× antes; 1,23× vs 1,12× depois).

Status das seeds: 95% e 98% têm 3 seeds nos três protocolos com retreino (eixo 2b: 10 runs da seed 0 em 20/09, seeds 1–2 de 22/09 a 01/10; eixo 2c: 12 runs em 21–22/09). Os níveis 50/70/90% do eixo 2b têm só a seed 0. O eixo 2d tem 3 seeds em todos os cinco níveis (30 células em 17/10).

## 4. Eixo 2: Quantização int8 pós-treinamento (PTQ)

Duas rotas, um artefato por backend, porque acurácia e latência de um modelo int8 dependem do backend e não se transferem entre eles. Em cada backend o baseline FP32 é medido no mesmo runtime. Calibração em ambas com **1.024 imagens de `fit`** (amostra determinística, seed 0), nunca do teste. Critério de escalada para QAT: queda de acurácia > 1 p.p.; não ocorreu.

**CPU** (`src/compress/quantize_cpu.py`): FX graph mode (`prepare_fx` → calibração → `convert_fx`), `qconfig_mapping` do backend `x86` (kernels fbgemm), fusão conv+bn+relu, observadores por tensor nas ativações e por canal nos pesos. Artefato TorchScript `model_int8_fbgemm.pt`, recarregado do disco e avaliado no teste completo em CPU (8 threads na avaliação). A DenseNet quantizou pelo FX sem tratamento especial das concatenações.

**GPU** (`src/compress/quantize_onnx_qdq.py`, `trt_build.py`, `trt_eval.py`): TensorRT 11 removeu a calibração implícita e as flags de precisão; as redes são fortemente tipadas e a precisão vem do grafo ONNX. FP32 a partir de `model.onnx` (opset 17; TF32 desligado); FP16 a partir de `model_fp16.onnx`; INT8 a partir de `model_qdq_int8.onnx` (ONNX Runtime `quantize_static`, formato QDQ, int8 simétrico, pesos por canal já em int8 com só `DequantizeLinear`, bias em float, convolução de entrada em float). Engines com perfil lote 1–32, workspace 4 GiB; o `EngineInspector` grava o histograma de tipos das camadas. Paridade ONNX × PyTorch verificada em 64 imagens de `val` (ResNet-50: max |Δ| 5,7e-6, 0 argmax divergentes; DenseNet-121: 2,9e-6, 0).

### 4.1 Resultados por backend e por seed

| arquitetura | backend | precisão | razão de erro | erros / baseline (s0, s1, s2) | acurácia (%) | artefato (MiB, × FP32) | img/s na avaliação |
|---|---|---|---|---|---|---|---|
| ResNet-50 | CPU fbgemm | int8 | 1,04 ± 0,07 | 170/177, 174/165, 196/178 | 99,785 | 23,0 (0,26) | 452 (8 thr) |
| ResNet-50 | TensorRT | fp32 | 1,00 ± 0,01 | 176/177, 163/165, 179/178 | 99,793 | 100,8 (1,12) | 1.965 |
| ResNet-50 | TensorRT | fp16 | 1,00 ± 0,01 | 176/177, 166/165, 179/178 | 99,792 | 45,5 (0,51) | 5.068 |
| ResNet-50 | TensorRT | int8 | **1,01 ± 0,03** | 173/177, 166/165, 184/178 | 99,791 | 23,9 (0,27) | 3.845 |
| DenseNet-121 | CPU fbgemm | int8 | **1,24 ± 0,07** | 210/160, 203/167, 198/167 | 99,756 | 7,7 (0,28) | 476 (8 thr) |
| DenseNet-121 | TensorRT | fp32 | 1,00 ± 0,00 | 160/160, 167/167, 168/167 | 99,803 | 34,3 (1,26) | 1.670 |
| DenseNet-121 | TensorRT | fp16 | 1,00 ± 0,00 | 160/160, 166/167, 167/167 | 99,803 | 15,0 (0,55) | 3.198 |
| DenseNet-121 | TensorRT | int8 | 1,07 ± 0,03 | 175/160, 174/167, 177/167 | 99,790 | 14,7 (0,54) | 1.102 |

Fonte: `results/eixo2/quant_summary.csv`, `results/eixo2/quant_runs.csv`. img/s medidos na avaliação com dataloader, lote 32; a latência controlada está no documento 04.

**Sanidade.** A engine FP32 reproduz a acurácia do PyTorch (ResNet s0: 176 vs 177 erros; DenseNet s0: 160 vs 160).

**Leitura.** A int8 não custa nada à ResNet-50 em nenhum backend. Na DenseNet-121 o custo depende do backend: **1,24× em CPU** e 1,07× no TensorRT com calibração robusta. A diferença CPU entre arquiteturas (Δ razão 0,20 > 2× desvio 0,14) é o primeiro sinal assimétrico entre arquiteturas do projeto. Hipótese para a CPU: a concatenação densa junta num mesmo tensor ativações de escalas distintas, e a escala única por tensor do PTQ estático perde resolução nos canais de menor amplitude. A acurácia difere entre backends para o mesmo modelo (ResNet s0: 170 erros em CPU, 173 no TensorRT; DenseNet s0: 210 vs 175); por isso é reportada por backend.

### 4.2 Varredura de calibradores e a troca entropia → percentil 99,99

**Como surgiu.** Na primeira rodada (14/09) o calibrador das ativações no grafo QDQ foi escolhido em `val` entre MinMax e entropia (KL) com uma única amostra: ResNet-50 61 erros em val com entropia contra 131 com MinMax; DenseNet 63 vs 67. A entropia foi adotada. Em 21/09, ao quantizar os 18 pontos do eixo 4 (calibração restrita à fração de cada run), o mesmo modelo deu de 194 a 2.596 erros no teste conforme as 1.024 imagens sorteadas. A calibração era instável à amostra.

**Varredura** (`scripts/calib_sweep.sh`, modelo `eixo4_resnet50_f075_s0`, 60 erros em `val` em FP32, quatro amostras de 1.024 imagens de `fit`):

| calibrador | amostra 0 | amostra 1 | amostra 2 | amostra 3 | faixa |
|---|---|---|---|---|---|
| MinMax | 966 | 604 | 813 | 808 | 604–966 |
| Entropia (KL) | 126 | 66 | 72 | 689 | 66–689 |
| **Percentil 99,99** | 64 | 62 | 62 | 63 | **62–64** |
| Percentil 99,999 | 62 | 67 | 67 | 68 | 62–68 |

Fonte: `results/eixo2/calib_sweep.csv` (erros em `val`, 19.511 imagens).

**Decisão.** Percentil 99,99, escolhido em `val` por robustez à amostra, sem tocar no teste. **28 células int8 TensorRT refeitas** (6 do eixo 1, 4 de poda+int8, 18 do eixo 4); os resultados com entropia ficam arquivados em `metrics_trt_int8_entropy.json` de cada run. A int8 de CPU (FX/fbgemm, observador por histograma) não é afetada. A latência int8 não muda (mesmo grafo; só as escalas mudam).

**O que a troca corrigiu nos números** (teste):

| célula | entropia (descartada) | percentil 99,99 |
|---|---|---|
| ResNet-50 TRT int8 (3 seeds) | 1,08 ± 0,08 | 1,01 ± 0,03 |
| DenseNet-121 TRT int8 (3 seeds) | 1,19 ± 0,08 | 1,07 ± 0,03 |
| ResNet-50 poda 90% + TRT int8 (s0) | 252 erros | 195 erros |
| DenseNet-121 poda 90% + TRT int8 (s0) | 242 erros | 185 erros |
| eixo 4, 75% s0, TRT int8 | 438 erros | 193 erros (denso 183) |

Duas afirmações da primeira rodada foram corrigidas: (1) "a int8 degrada a DenseNet nos dois backends" vale para a CPU (1,24×); no TensorRT é 1,07×; (2) "a int8 sobre modelos podados degrada mais no TensorRT que em CPU" era artefato da calibração por entropia.

**Incidente de memória.** Os calibradores por histograma do ONNX Runtime acumulam todas as ativações intermediárias do conjunto inteiro antes de montar os histogramas: 30 GB e 53 GB de RSS para a ResNet-50, dois encerramentos por OOM em 14/09 que derrubaram a sessão e a fila de poda. Correções: coletor tornado incremental (alimenta o `HistogramCollector` lote a lote, `max_intermediate_outputs = 2`), execução sob `systemd-run --user --scope -p MemoryMax=16G`, patches encadeados com `&&` e verificados por `grep` antes de rodar. Com isso o grafo Q/DQ da ResNet-50 sai em ~90 s e o da DenseNet em ~120 s.

### 4.3 Composição das camadas nas engines int8 (seed 0, percentil 99,99)

| arquitetura | camadas na engine | Int8 | Float | outras | nós `QuantizeLinear` no ONNX | tempo de build |
|---|---|---|---|---|---|---|
| ResNet-50 | 95 | 72 | 6 | 17 | 120 | 142 s |
| DenseNet-121 | 1.497 | 973 | 399 | 125 | 370 | 2.105 s |

Fonte: `runs/eixo1_<arch>_s0/trt/build_int8.json` (`EngineInspector`). Para comparação, a engine FP16 da DenseNet tem 774 camadas (772 em Half) e a FP32, 740.

## 5. Achados do eixo 2 a sustentar no texto

1. **90% dos pesos são dispensáveis** nas duas redes com fine-tuning de 5 épocas; o joelho de 98% é do orçamento curto (seção 3.4).
2. **Sem retreino, só a poda de 50% é gratuita** (1,05× / 1,01×). A 70% o erro já dobra na ResNet-50 (1,99×; DenseNet-121 1,29×), a 90% a rede quebra (~14% de acurácia) e a 95/98% responde ao acaso (1/18). As mesmas máscaras com 5 épocas de fine-tuning voltam a 1,10× / 1,03× em 90%: o retreino recupera a calibração conjunta dos pesos restantes, não a capacidade da sub-rede, que a poda preserva. Acima de 50%, retreinar é obrigatório (seção 3.1).
3. **Poda não-estruturada não acelera.** Kernels densos calculam os zeros: latência CPU dos podados = baseline (ResNet 10,9–11,4 ms vs 10,7 ms; DenseNet 15,8–16,5 vs 16,6 ms, 16 threads), MACs inalterados. O que cai é o artefato comprimido (21% do original a 90%) e os parâmetros efetivos. Detalhes no documento 04.
4. **A int8 é gratuita para a ResNet-50 e custa à DenseNet-121**: 24% a mais de erro em CPU, 7% no TensorRT.
5. **A engine int8 da DenseNet-121 é mais lenta que a FP16 no TensorRT** (3,19 vs 2,80 ms em lote 1; 17,2 vs 8,57 ms em lote 32; medição final de 18/10, mediana de 5 rodadas; na avaliação com dataloader 1.102 vs 3.198 img/s). Causa: 399 das 1.497 camadas ficam em float, e as conversões Q/DQ em torno das concatenações custam mais do que os kernels int8 economizam. Na ResNet-50 a int8 e a FP16 ficam a 3% em lote 1 (0,80 vs 0,82 ms) e a int8 é 1,3× mais rápida em lote 32 (3,10 vs 4,09 ms).
6. **A calibração da int8 é um hiperparâmetro** e precisa ser escolhida em validação com mais de uma amostra (seção 4.2).
7. **Poda e int8 compõem de forma aproximadamente aditiva** na ResNet-50 (90%: 1,07× → 1,14× CPU / 1,10× TensorRT). Na DenseNet-121 a int8 de CPU domina e 95% + int8 CPU sai do orçamento de 1,5× (1,58×). Tabela no documento 04, seção 1.
8. **Custo teórico e real divergem.** A DenseNet tem 30% menos MACs e 3,4× menos parâmetros, mas é mais lenta em todos os runtimes medidos.

## 6. Reprodução

```bash
scripts/queue.sh eixo1                                           # 6 runs, ~9,7 h
scripts/queue_prune.sh "0 1 2" "0.5 0.7 0.9 0.95 0.98"            # 30 runs, ~8,5 h
scripts/queue_prune_first.sh "0 1 2" "0.95 0.98"                  # eixo 2b
scripts/queue_rewind.sh "0 1 2" "0.95 0.98"                       # eixo 2c
python src/compress/prune_eval.py --all                           # eixo 2d: 30 células, ~25 min
python src/compress/quantize_cpu.py --checkpoint runs/eixo1_<arch>_s<k>/checkpoints/best.pt --threads 8
scripts/queue_trt.sh "runs/eixo1_<arch>_s<k>" "fp32 fp16 int8"    # ONNX -> Q/DQ -> engines -> teste
scripts/calib_sweep.sh                                            # results/eixo2/calib_sweep.csv
python results/eixo1/make_eixo1.py; python results/eixo2/make_pruning.py
python results/eixo2/make_quant.py; python results/eixo2/make_prune_compare.py
```

Pesos: releases `v0.2-eixo1` (6 × `best.pt` + 6 × `last.pt`) e `v0.3-eixo2` (30 modelos podados com máscaras + 10 artefatos int8 de CPU). As engines TensorRT não são publicadas (específicas da GPU e da versão); `make trt` as reconstrói.
