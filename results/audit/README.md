# Auditoria de vazamento — "a acurácia de 99,8% é real?"

Feita em 2026-09-19 a pedido do orientador. Resposta curta: **sim**. O pipeline não vaza; o dataset tem um vazamento pequeno e
mensurável (a mesma pessoa sob mais de um `user_id`), que explica no máximo 0,07 ponto percentual.

## 1. O que foi verificado e o que deu

| # | verificação | como | resultado |
|---|---|---|---|
| A1 | sujeitos disjuntos entre `fit`, `val` e `test` | interseção de `user_id` no manifesto v3 | 0 / 0 / 0 (11.805 / 2.268 / 6.044 sujeitos) |
| A2 | nenhuma imagem em dois splits | unicidade de caminho e de chave | 0 duplicadas |
| A3 | manifesto confere com a fonte | re-leitura independente dos JSONs brutos do HaGRID | 278.715 / 278.715 linhas com `user_id` e classe idênticos |
| A4 | imagem em mais de uma classe | chaves em 2+ JSONs de classe | 0 |
| A5 | o `user_id` do HaGRID é consistente | usuários em 2+ splits oficiais do HaGRID | 0 |
| B1 | duplicatas exatas entre treino e teste | md5 dos 278.715 JPEGs processados | 0 |
| B2 | fotos quase idênticas entre treino e teste | dHash 64 bits (dist. ≤ 3) + diferença média de pixel (MAD) | **87 imagens de teste (0,10%)** com MAD < 10 — ver § 2 |
| C1 | o pipeline vaza rótulo? | treino com rótulos **embaralhados**, teste intacto | 4,9% de acerto (acaso = 5,6%) |
| C2 | o número se sustenta fora do nosso split? | 6 modelos avaliados em **278.702 imagens de 19.042 sujeitos nunca usados** | erro 0,216% / 0,235% — ver § 3 |

## 2. O vazamento que existe: contas duplicadas no HaGRID

O dHash apontou 29.616 pares teste×treino, mas dois terços tinham rótulos diferentes (impossível ser a mesma foto): colisões do hash em
crops de fundo liso. A checagem por pixel separa: só **436 pares** têm MAD < 10 com o mesmo rótulo, e são de fato a mesma mão no mesmo
cenário sob `user_id` diferentes (`near_duplicates.png`) — a mesma pessoa/sessão com mais de uma conta na plataforma de coleta.
O split por `user_id` é correto; é o identificador do dataset que não é uma identidade perfeita.

Impacto, medido de forma conservadora: todo **usuário** de teste com ao menos uma foto quase idêntica a uma de treino é marcado como
suspeito e **todas** as suas imagens são removidas (modelos de seed 0):

| conjunto de teste | n | ResNet-50 erro | DenseNet-121 erro |
|---|---|---|---|
| completo (o reportado) | 83.613 | 0,2117% | 0,1914% |
| sem 52 usuários suspeitos (MAD < 10) | 78.126 | 0,2125% | 0,1946% |
| só esses 52 usuários | 5.487 | 0,2005% | 0,1458% |
| sem 292 usuários suspeitos (MAD < 20, critério frouxo, com falsos positivos) | 48.630 | 0,2817% | 0,2591% |
| só esses 292 usuários | 34.983 | 0,1143% | 0,0972% |

No critério estrito nada muda. No frouxo — que descarta 42% do teste, incluindo muitos falsos positivos — o erro sobe para 0,28% / 0,26%:
**pior caso 99,72% de acurácia**. Os usuários suspeitos são contribuidores pesados (≈120 imagens cada, contra 14 da média), e o erro
menor neles pode ser tanto vazamento quanto fotos mais limpas; o número de 0,28% é o limite superior honesto.

## 3. Teste externo: os 50% do HaGRID que nunca entraram no projeto

O subconjunto de trabalho usa 50% dos sujeitos das 18 classes. Os outros 19.042 sujeitos (278.702 imagens) nunca apareceram em nenhuma
versão do manifesto: nenhum treino, validação, calibração, máscara ou seleção os tocou. Recortados com a mesma função de pré-processamento:

| arquitetura (3 seeds) | erro no teste do projeto | erro no holdout externo |
|---|---|---|
| ResNet-50 | 0,2073 ± 0,0087% | 0,2353 ± 0,0082% |
| DenseNet-121 | 0,1969 ± 0,0048% | 0,2159 ± 0,0052% |

Num conjunto 3,3× maior e independente, a acurácia é 99,76–99,78%. A ordem entre arquiteturas se mantém.

## 4. Por que o número é alto

A tarefa, neste formato, é fácil: o classificador recebe um **recorte apertado da mão** (não a cena), as 18 classes são visualmente
distintas, há 175 mil imagens de treino e a rede parte de pesos ImageNet. O baseline de 5 épocas já chegava a 99,79%, e o artigo do
HaGRID reporta F1 acima de 99% para classificadores ResNet em quadro inteiro (a conferir a tabela exata para citação). Os ~170 erros
residuais concentram-se em `stop`↔`palm` e em imagens borradas ou cortadas. É por isso que o projeto reporta **razão de erro** e não
deltas de acurácia nos eixos 2–4.

## 5. Reprodução

```bash
python scripts/audit_leakage.py          # A1–A5, B1–B2 -> leakage_audit.json, near_duplicates.{csv,png}
python scripts/audit_multiaccount.py     # § 2 -> multiaccount_summary.csv, clean_vs_suspect_eval.csv
python scripts/make_holdout.py           # recorta os sujeitos nunca usados -> data/holdout_unused/
python src/eval.py --checkpoint runs/eixo1_<arch>_s<k>/checkpoints/best.pt --split test \
       --manifest data/holdout_unused/manifest_holdout.csv --out results/audit/holdout_<arch>_s<k>.json
python src/train.py --config configs/train_resnet50.yaml --run-name _audit_shuffled --epochs 2 \
       --limit-fit 20000 --limit-val 2000 --shuffle-labels --eval-test     # C1
```
