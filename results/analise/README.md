# Análise qualitativa dos erros (teste, modelos de seed 0)

Gerado por `make_erros.py` a partir de `preds_test_<arch>_s0.csv` (`src/eval.py --save-preds`).

## As duas arquiteturas erram as mesmas imagens

| | ResNet-50 | DenseNet-121 |
|---|---|---|
| erros no teste (83.613 imagens) | 177 | 160 |
| erros em comum | 117 (66% dos seus erros) | 117 (73% dos seus erros) |
| em comum **com a mesma classe predita** | 109 de 117 | |
| imagens erradas por pelo menos uma | 220 | |
| confiança média nos erros / nos acertos | 0,84 / 0,9995 | 0,85 / 0,9994 |

Duas redes com conectividades opostas, treinadas de forma independente, que falham nas mesmas 117 imagens e concordam entre si sobre a
classe em 109 delas: isso não é comportamento de erro de modelo, é comportamento de **rótulo questionável**. A inspeção visual de
`erros_ambas.png` confirma o padrão — imagem rotulada `three` com quatro dedos estendidos, `stop` com os dedos afastados (a definição de
`palm`), `two_up` com os dedos em V (`peace`), `palm` com o polegar escondido (`four`). A proporção exata exigiria re-anotação manual
(não feita); o que os dados sustentam é que **uma fração grande do erro residual de ~0,2% é o piso de ruído de rótulo do HaGRID**, não
capacidade do modelo.

Consequências para a leitura do trabalho: (1) explica por que ResNet-50 e DenseNet-121 empatam no eixo 1 — ambas chegaram ao piso;
(2) reforça o uso da **razão de erro**: a degradação por compressão aparece como erros *novos* somados a um piso comum;
(3) a acurácia "verdadeira" é provavelmente maior que a medida.

## Confusões mais frequentes (`confusion_pairs.csv`)

`stop` ↔ `palm` responde por 36% dos erros da ResNet-50 (32 + 31 de 177): as duas classes diferem apenas pela abertura dos dedos. Seguem
`two_up` ↔ `peace` (dedos juntos vs em V), `three` → `four` e `three` → `peace`. Todas são confusões entre gestos que diferem por um
detalhe de pose, e são as mesmas nas duas arquiteturas.

## Arquivos

- `erros_resnet50.{pdf,png}`, `erros_densenet121.{pdf,png}` — os 24 erros mais confiantes de cada arquitetura
- `erros_ambas.{pdf,png}` — 24 das 117 imagens erradas pelas duas
- `confusion_pairs.csv`, `erros_overlap.json`
