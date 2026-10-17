# 05: Validade, limitações e lições de método

Base para as seções de Discussão e Limitações do artigo. Fontes: `results/audit/` (`README.md`, `leakage_audit.json`, `holdout_summary.csv`, `clean_vs_suspect.csv`, `multiaccount_summary.csv`, `shuffled_labels_control.json`), `results/analise/` (`README.md`, `erros_overlap.json`, `confusion_pairs.csv`), `STATUS.md`.

## 1. Auditoria de vazamento: "a acurácia de 99,8% é real?"

Feita em 19/09/2026 a pedido do orientador, motivada pela acurácia de ~99,8%. Resposta curta: **sim**. O pipeline não vaza; o dataset tem um vazamento pequeno e mensurável (a mesma pessoa sob mais de um `user_id`), que explica no máximo 0,07 ponto percentual.

### 1.1 Verificações de manifesto (`scripts/audit_leakage.py`, sobre o manifesto v3)

| # | verificação | como | resultado |
|---|---|---|---|
| A1 | sujeitos disjuntos entre `fit`, `val` e `test` | interseção de `user_id` | fit∩test 0, val∩test 0, fit∩val 0 (11.805 / 2.268 / 6.044 sujeitos) |
| A2 | nenhuma imagem em dois splits | unicidade de `image_path` e de chave | 0 caminhos duplicados, 0 chaves em mais de um split |
| A3 | manifesto confere com a fonte | re-leitura independente dos JSONs brutos do HaGRID | 278.715 / 278.715 linhas encontradas, com `user_id` e classe idênticos |
| A4 | imagem em mais de uma classe | chaves presentes em 2+ JSONs de classe | 0 |
| A5 | o `user_id` do HaGRID é consistente entre seus splits oficiais | usuários em 2+ splits oficiais (train/val/test do HaGRID) | 0 |

Fonte: `results/audit/leakage_audit.json`. O item A5 também mostra como o nosso split cruza o oficial: do `train` oficial, 133.695 imagens foram para `fit`, 16.124 para `val` e 63.692 para `test`; do `val` oficial, 15.415 / 1.221 / 7.909; do `test` oficial, 26.481 / 2.166 / 12.012. O split do projeto é independente do oficial e só respeita o `user_id`.

### 1.2 Verificações de conteúdo

| # | verificação | como | resultado |
|---|---|---|---|
| B1 | duplicatas exatas | md5 dos 278.715 JPEGs processados | 0 grupos com 2+ arquivos; 0 entre treino e teste; 0 entre fit e val |
| B2 | quase-duplicatas treino × teste | dHash 64 bits, distância de Hamming ≤ 3 (banding 4×16 bits), seguido de diferença média absoluta (MAD) de miniaturas 32×32 | 29.616 pares brutos (4.955 imagens de teste, 5,9%); só 9.559 com o mesmo rótulo; **436 pares com MAD < 10 e mesmo rótulo, 87 imagens de teste (0,10%)** |

Fonte: `results/audit/leakage_audit.json`, `results/audit/near_duplicates.csv`, `near_duplicates.png`. Distribuição dos pares por Hamming: 0 → 233, 1 → 1.442, 2 → 6.546, 3 → 21.395. Dois terços dos pares do hash tinham rótulos diferentes (impossível ser a mesma foto): colisões em recortes de fundo liso. O hash sozinho não basta; a checagem por pixel separa.

### 1.3 Controles comportamentais

| # | controle | como | resultado |
|---|---|---|---|
| C1 | o pipeline vaza rótulo? | treino da ResNet-50 (seed 0, manifesto v2) com os rótulos de `fit` **permutados**, `val` e teste intactos | acurácia no teste **4,9%** (79.506 erros em 83.613), F1 macro 3,0%; acaso para 18 classes = 5,6% |
| C2 | o número se sustenta fora do split? | os 6 modelos do eixo 1 avaliados em um **holdout externo**: 278.702 imagens de 19.042 sujeitos que nunca entraram em nenhuma versão do manifesto | erro 0,235% (ResNet-50) e 0,216% (DenseNet-121); ver 1.5 |

Fonte: `results/audit/shuffled_labels_control.json` (`_audit_shuffled_labels`, `best.pt` da época 1), `results/audit/holdout_summary.csv`. O teste `tests/test_datamodule.py::test_shuffle_labels_permutes_only_fit` garante que o embaralhamento atinge só `fit`.

### 1.4 O vazamento que existe: contas duplicadas no HaGRID

Os 436 pares com MAD < 10 e mesmo rótulo são a mesma mão no mesmo cenário sob `user_id` diferentes: a mesma pessoa ou sessão com mais de uma conta na plataforma de coleta. O split por `user_id` é correto; é o identificador do dataset que não é uma identidade perfeita.

Impacto medido de forma conservadora (`scripts/audit_multiaccount.py`): todo **usuário** de teste com ao menos uma foto quase idêntica a uma de treino é marcado como suspeito, e **todas** as suas imagens são removidas. Modelos de seed 0.

| critério | pares | imagens de teste duplicadas | usuários de teste suspeitos | imagens removidas | fração do teste | usuários de treino envolvidos |
|---|---|---|---|---|---|---|
| estrito (MAD < 10) | 436 | 87 | 52 | 5.487 | 6,6% | 93 |
| frouxo (MAD < 20, com falsos positivos) | 1.425 | 610 | 292 | 34.983 | 41,8% | 375 |

Fonte: `results/audit/multiaccount_summary.csv`.

| conjunto de teste | n | ResNet-50 erros / erro (%) | DenseNet-121 erros / erro (%) |
|---|---|---|---|
| completo (o reportado) | 83.613 | 177 / 0,2117 | 160 / 0,1914 |
| sem 52 usuários suspeitos (MAD < 10) | 78.126 | 166 / 0,2125 | 152 / 0,1946 |
| só esses 52 usuários | 5.487 | 11 / 0,2005 | 8 / 0,1458 |
| sem 292 usuários suspeitos (MAD < 20) | 48.630 | 137 / **0,2817** | 126 / **0,2591** |
| só esses 292 usuários | 34.983 | 40 / 0,1143 | 34 / 0,0972 |
| holdout externo (19.042 sujeitos nunca usados) | 278.702 | 674 / 0,2418 | 585 / 0,2099 |

Fonte: `results/audit/clean_vs_suspect.csv`.

**Leitura.** No critério estrito nada muda (0,2125% vs 0,2117%). No frouxo, que descarta 42% do teste incluindo muitos falsos positivos, o erro sobe para 0,28% / 0,26%: **pior caso de 99,72% de acurácia**. Os usuários suspeitos são contribuidores pesados (≈120 imagens cada, contra 14 da média), e o erro menor neles pode ser tanto vazamento quanto fotos mais limpas; 0,28% é o limite superior honesto. O vazamento por contas duplicadas explica no máximo 0,07 p.p. e é declarado como limitação do dataset.

### 1.5 Holdout externo por seed

Os outros 50% dos sujeitos das 18 classes (19.042 sujeitos, 278.702 imagens) nunca apareceram em nenhuma versão do manifesto: nenhum treino, validação, calibração, máscara ou seleção os tocou. Recortados com a mesma função de pré-processamento (`scripts/make_holdout.py` → `data/holdout_unused/`).

| modelo | erros no teste (83.613) | erro no teste (%) | erros no holdout (278.702) | erro no holdout (%) | F1 macro no holdout |
|---|---|---|---|---|---|
| ResNet-50 s0 | 177 | 0,2117 | 674 | 0,2418 | 0,9976 |
| ResNet-50 s1 | 165 | 0,1973 | 630 | 0,2261 | 0,9978 |
| ResNet-50 s2 | 178 | 0,2129 | 663 | 0,2379 | 0,9976 |
| **ResNet-50 (média)** | | 0,2073 ± 0,0087 | | **0,2353 ± 0,0082** | |
| DenseNet-121 s0 | 160 | 0,1914 | 585 | 0,2099 | 0,9979 |
| DenseNet-121 s1 | 167 | 0,1997 | 608 | 0,2182 | 0,9978 |
| DenseNet-121 s2 | 167 | 0,1997 | 612 | 0,2196 | 0,9978 |
| **DenseNet-121 (média)** | | 0,1969 ± 0,0048 | | **0,2159 ± 0,0052** | |

Fonte: `results/audit/holdout_summary.csv`, `results/audit/holdout_<arch>_s<k>.json`.

Num conjunto 3,3× maior e independente, a acurácia é 99,76–99,78%; o erro é 10–14% maior que no teste do projeto, e a ordem entre arquiteturas se mantém. Na ResNet-50 s0, a maior confusão no holdout é `stop` ↔ `palm` (86 + 83 de 674 erros), a mesma do teste.

### 1.6 Por que o número é alto

A tarefa, neste formato, é fácil: o classificador recebe um recorte apertado da mão (não a cena), as 18 classes são visualmente distintas, há 175 mil imagens de treino e a rede parte de pesos ImageNet. O baseline de 5 épocas já chegava a 99,79%, e o artigo do HaGRID reporta F1 acima de 99% para classificadores ResNet (conferir a tabela exata antes de citar). Por isso o projeto reporta razão de erro, e não deltas de acurácia, nos eixos 2 a 4.

## 2. Análise qualitativa dos erros (teste, modelos de seed 0)

| | ResNet-50 | DenseNet-121 |
|---|---|---|
| erros no teste | 177 | 160 |
| erros em comum | 117 (66% dos seus) | 117 (73% dos seus) |
| em comum **com a mesma classe predita** | 109 de 117 | |
| imagens erradas por pelo menos uma | 220 | |
| confiança média nos erros / nos acertos | 0,84 / 0,9995 | 0,85 / 0,9994 |

Fonte: `results/analise/erros_overlap.json`; predições por imagem em `results/analise/preds_test_<arch>_s0.csv` (`src/eval.py --save-preds`).

Duas redes com conectividades opostas, treinadas de forma independente, falham nas mesmas 117 imagens e concordam sobre a classe em 109 delas. Isso não é comportamento de erro de modelo; é comportamento de **rótulo questionável**. A inspeção visual de `erros_ambas.png` confirma: imagem rotulada `three` com quatro dedos estendidos, `stop` com os dedos afastados (a definição de `palm`), `two_up` com os dedos em V (`peace`), `palm` com o polegar escondido (`four`). A proporção exata exigiria re-anotação manual (não feita). O que os dados sustentam: uma fração grande do erro residual de ~0,2% é o piso de ruído de rótulo do HaGRID.

**Confusões mais frequentes** (`results/analise/confusion_pairs.csv`):

| arquitetura | verdadeiro → predito | n | fração dos erros |
|---|---|---|---|
| ResNet-50 | stop → palm | 32 | 18,1% |
| ResNet-50 | palm → stop | 31 | 17,5% |
| ResNet-50 | two_up → peace | 9 | 5,1% |
| ResNet-50 | peace → two_up | 8 | 4,5% |
| ResNet-50 | three → four | 7 | 4,0% |
| ResNet-50 | three → peace | 6 | 3,4% |
| DenseNet-121 | palm → stop | 27 | 16,9% |
| DenseNet-121 | stop → palm | 25 | 15,6% |
| DenseNet-121 | three → peace | 8 | 5,0% |
| DenseNet-121 | three → four | 6 | 3,7% |
| DenseNet-121 | two_up → peace | 6 | 3,7% |

`stop` ↔ `palm` responde por **36% dos erros da ResNet-50** (63 de 177) e 33% dos da DenseNet-121 (52 de 160): as duas classes diferem apenas pela abertura dos dedos. Todas as confusões frequentes são entre gestos que diferem por um detalhe de pose, e são as mesmas nas duas arquiteturas.

**Consequências para a leitura do trabalho.** (1) Explica por que ResNet-50 e DenseNet-121 empatam no eixo 1: ambas chegaram ao piso. (2) Reforça a razão de erro: a degradação por compressão aparece como erros novos somados a um piso comum. (3) A acurácia "verdadeira" é provavelmente maior que a medida.

## 3. Limitações

1. **Poda não-estruturada não acelera.** O ganho é só de parâmetros efetivos e artefato comprimido; MACs e latência ficam iguais ao baseline em CPU e GPU densos. A poda estruturada, que reduziria MACs, não foi aplicada porque na DenseNet as concatenações exigem propagação de índices por todo o bloco denso; a comparação entre arquiteturas ficaria assimétrica.
2. **Latência GPU preliminar.** Medida com a sessão gráfica aberta (campo `note` em cada JSON). A medição definitiva em TTY, após reconstruir as engines (`make trt`), está pendente. A decisão do eixo 3 depende de diferenças de latência de 0,9% a 15% entre finalistas: pode trocar as duas primeiras posições, não a arquitetura nem a precisão.
3. **Extra B com uma seed** (ResNet-50, seed 0, 25% dos dados). A direção do efeito é clara (1,69× e 2,03× contra 1,07× e 1,14×); a magnitude pede mais seeds.
4. **Eixo 2b em 50/70/90% com uma seed.** Só 95% e 98% têm 3 seeds nos três protocolos. A conclusão (orçamento de retreino decide) está sustentada em 95/98%, onde a diferença importa.
5. **Poda + int8 com uma seed** (seed 0) em 90% e 95%.
6. **Dataset com ruído de rótulo.** O piso de ~0,2% é em boa parte rótulo questionável (seção 2); a acurácia absoluta é um limite inferior e as comparações são relativas.
7. **`user_id` não é identidade.** Contas duplicadas no HaGRID produzem 0,10% de quase-duplicatas entre treino e teste; pior caso de 99,72% (seção 1.4).
8. **Regime de fine-tuning.** Todos os treinos partem de pesos ImageNet. A curva do eixo 4 mede redução de dados nesse regime, que é o cenário realista de aplicação (Kornblith et al., 2019), não a eficiência de dados de treino do zero.
9. **Eixo 4 só na ResNet-50**, por critério do protocolo (melhor configuração do eixo 3). A curva da DenseNet-121 (extra A, ~10 h de GPU) ficou em espera.
10. **Dois ajustes de protocolo pendentes de confirmação do orientador:** retirada da parada antecipada no eixo 4 (documento 04, § 3.4) e tolerância de empate de 5% na decisão do eixo 3 (documento 04, § 2).
11. **Subconjunto de 50% e 18 classes.** Os números absolutos não são comparáveis diretamente com os do artigo do HaGRID (quadro inteiro, 33 classes na v2, split oficial).
12. **Uma GPU, um hardware.** Latências valem para RTX 3080 Ti e Ryzen 9 9950X com TensorRT 11.3 e PyTorch 2.11; a FX graph mode quantization está marcada como legada no PyTorch 2.11.
13. **Não-determinismo residual** de `cudnn.benchmark = True` nos kernels de backward; dados, init e augmentation são determinísticos.
14. **Augmentation mínima** (random crop + hflip), por decisão de protocolo. Não se mediu o efeito de augmentations mais fortes.

## 4. Lições de método

1. **Parada antecipada com paciência curta trunca a agenda cosseno.** O erro em `val` oscila ±0,05 p.p. entre épocas; cinco épocas sem recorde ocorrem por acaso. 13 de 18 runs do eixo 4 pararam com o lr ainda alto (até 56% do pico), e um ponto de 100% fez 277 erros em vez de 165. Com agenda cosseno, a seleção do melhor checkpoint por validação já cumpre o papel da parada antecipada; o número de épocas deve ser fixo.
2. **A calibração da int8 é um hiperparâmetro e precisa de validação com mais de uma amostra.** Com entropia, o mesmo modelo foi de 194 a 2.596 erros no teste conforme a amostra de 1.024 imagens; parecia bom com uma única amostra. Percentil 99,99, escolhido em `val` em quatro amostras, varia de 62 a 64 erros onde o denso faz 60. Duas afirmações sobre arquiteturas foram corrigidas depois da troca.
3. **`user_id` de um dataset crowdsourced não é identidade.** O split por sujeito é necessário mas não suficiente; o efeito do identificador imperfeito foi medido (quase-duplicatas, remoção de usuários suspeitos, holdout externo) em vez de presumido.
4. **Memória da calibração.** Calibradores por histograma do ONNX Runtime retêm todas as ativações intermediárias do conjunto de calibração (30–53 GB para a ResNet-50 com 1.024 imagens). Dois OOM derrubaram a sessão e a fila de GPU. Solução: coletor incremental lote a lote e todo processo de CPU com memória imprevisível sob `systemd-run --user --scope -p MemoryMax=16G`.
5. **Medir o baseline no mesmo runtime da célula comprimida.** O TensorRT FP16 já dá 2,8× sobre o TensorRT FP32; comparar int8 TensorRT com PyTorch eager misturaria efeito de quantização com efeito de runtime. Em CPU, baseline eager e int8 fbgemm no mesmo processo.
6. **Acurácia int8 por backend.** O mesmo modelo dá 170 erros em fbgemm e 173 no TensorRT (ResNet s0), 210 e 175 (DenseNet s0). Não assumir paridade.
7. **Comparar protocolos com orçamento igual.** "Podar antes × depois" só ficou interpretável depois do controle com orçamento igual (eixo 2c). Sem ele, a conclusão da seed 0 (ResNet ganha podando antes) estaria errada.
8. **Fixar o critério de decisão antes dos resultados** e registrar qualquer refinamento como proposta, ao lado da leitura literal (`decision.md`).
9. **Infraestrutura de execução.** Filas longas desacopladas da sessão (`setsid nohup`; uma fila lançada em background da ferramenta morreu na época 29/30 em 30/09); tudo retomável de `last.pt` com replay exato dos lotes; `gpu_wait.sh` ignorando processos de desktop (o compositor Wayland aparece como processo de compute e travou a espera); globs do zsh sem correspondência abortam o comando inteiro (um `rm -f` não rodou; usar `find -delete` e conferir).
10. **Rastreabilidade mínima por run:** seed, versão e sha256 do manifesto, commit git e estado da árvore, versões, GPU. Os dois primeiros runs do eixo 1 começaram com árvore suja e exigiram uma nota de proveniência; os seguintes saíram limpos porque o commit foi feito durante a fila.
11. **Todo número em `results/` sai de um script a partir de `runs/`.** Nenhuma tabela é digitada à mão; `make results` regenera tudo. Isso foi o que permitiu refazer 28 células int8 e republicar todas as tabelas no mesmo dia.
