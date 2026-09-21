# Mapa do código — que arquivo faz o quê

Guia para explicar o projeto. Os arquivos aparecem na ordem em que são usados. Dentro de cada um, só as funções que importam.

## Visão geral: o caminho de uma foto

```
fotos + anotações do HaGRID
   │  validate_raw.py      confere se os dados brutos estão íntegros
   │  preprocess.py        escolhe as pessoas, separa treino/prova, recorta as mãos  ──►  manifest.csv
   │  make_inner_split.py  separa a validação dentro do treino                       ──►  manifest_v2.csv
   │  make_masks.py        marca as frações de 75/50/25/10/5% das fotos              ──►  manifest_v3.csv
   ▼
datamodule.py   lê o manifesto e entrega as fotos em lotes para a rede
train.py        ensina a rede, acompanha na validação, guarda o melhor modelo      ──►  runs/<nome>/best.pt
eval.py         dá a prova final a um modelo pronto                                ──►  metrics.json
   ▼
compress/       deixa o modelo menor: apagar conexões (pruning.py) e compactar (quantize_*.py, trt_*.py)
measure/        mede tamanho e contas (cost.py) e velocidade (latency.py)
reduce_data.py  repete o treino com menos fotos (etapa 4)
   ▼
results/*/make_*.py   juntam tudo em tabelas e gráficos          app/webcam_demo.py   demonstração ao vivo
```

**Ideia central:** todo experimento lê a lista de fotos de um **manifesto** (uma planilha CSV), nunca as pastas. É isso que garante que todos os modelos viram exatamente as mesmas fotos e que nenhuma pessoa aparece no treino e na prova.

---

## 1. Dados — `src/`

### `validate_raw.py` — confere os dados brutos antes de qualquer coisa
| função | o que faz |
|---|---|
| `bbox_is_valid` | diz se a caixa em volta da mão é válida (4 números, tamanho positivo, dentro da foto) |
| `pick_annotation` | acha, numa foto, a caixa da mão que faz o gesto; ignora a mão marcada "sem gesto" |
| `main` | percorre os 18 gestos × 3 pastas; conta fotos, pessoas, caixas faltando; grava `results/raw_validation.csv` |

### `preprocess.py` — monta o subconjunto, separa treino/prova e recorta as mãos
| função | o que faz |
|---|---|
| `load_records` | lê as anotações dos 18 gestos e devolve a lista de fotos utilizáveis, sempre na mesma ordem |
| `pick_bbox` | caixa da mão do gesto (mesma lógica de `pick_annotation`) |
| `user_class_counts` | conta quantas fotos de cada gesto cada pessoa tem |
| `select_users_balanced` | **escolhe pessoas** (nunca fotos soltas) até juntar a fração pedida de cada gesto. Usada 3 vezes: 50% do total, 70% para treino, 10% para validação |
| `crop_box` | calcula o recorte: caixa da mão + 10% de folga, transformada em quadrado e empurrada para dentro da foto se sair |
| `process_one` | recorta uma foto, redimensiona para 256×256 e salva em JPEG. Pula se já existe |
| `build_plan` | junta tudo: quais fotos entram e se cada uma é treino ou prova |
| `write_manifest` | grava o `manifest.csv` ordenado, para sair idêntico em qualquer execução |
| `manifest_hash` | "impressão digital" (sha256) do manifesto, gravada em cada treino |
| `main` | executa o pipeline; `--sample N` faz um ensaio com N fotos; asserts param tudo se uma pessoa cair nos dois grupos |

### `make_inner_split.py` — cria o manifesto v2
Separa 10% das **pessoas** do treino como validação (coluna `inner_split`: `fit`, `val` ou `test`). A prova fica intocada. Reaproveita `select_users_balanced`.

### `make_masks.py` — cria o manifesto v3 (etapa 4)
| função | o que faz |
|---|---|
| `mask_seed` | semente do sorteio, derivada da impressão digital do v2 e do número da repetição |
| `add_masks` | para cada repetição, embaralha as fotos de cada gesto **uma vez**; a fração de 25% são as primeiras 25% dessa fila, a de 50% as primeiras 50%… Por isso as frações ficam uma dentro da outra e equilibradas por gesto |
| `check` | confere: máscaras só no treino, uma dentro da outra, proporção certa por gesto |

### `datamodule.py` — entrega as fotos para a rede
| função | o que faz |
|---|---|
| `read_manifest`, `class_names` | lê o CSV; lista os 18 gestos em ordem alfabética (o número de cada gesto nunca muda) |
| `split_frames` | devolve as três listas (treino, validação, prova) e **para tudo** se achar uma pessoa em duas delas; aplica a máscara de fração só no treino |
| `frame_for_split` | devolve a lista de um grupo pelo nome |
| `ManifestDataset` | abre uma foto do disco e devolve (imagem, número do gesto) |
| `build_transforms` | **o que é feito com a foto antes de entrar na rede** (ver a seção "Por que 224 e por que espelhar") |
| `build_dataloaders` | monta os três "entregadores" de lotes de 96 fotos, com 16 processos lendo do disco em paralelo |
| `build_eval_loader` | entregador de um grupo só, para avaliação |
| `run_mask_column` | descobre com que fração de fotos um treino da etapa 4 foi feito |

### `runinfo.py` e `metrics.py` — apoio
| função | o que faz |
|---|---|
| `git_info`, `env_info` | registram a versão do código e dos programas usados em cada treino |
| `manifest_version` | `manifest_v2.csv` → 2 |
| `compute_metrics` | a partir das respostas certas e das dadas: acerto, F1 (média do desempenho por gesto), taxa de erro, tabela de confusões |
| `write_json` | salva um resultado em arquivo |

## 2. Treino e prova — `src/`

### `train.py` — ensina a rede
| função | o que faz |
|---|---|
| `set_seed` | fixa os sorteios para o treino ser repetível |
| `build_model` | carrega a ResNet-50 ou a DenseNet-121 já treinada em fotos gerais e troca a última camada por uma de 18 saídas |
| `build_optimizer` | o "professor": como os números da rede são corrigidos a cada lote (SGD com momentum) |
| `build_scheduler` | o tamanho da correção ao longo do treino: sobe devagar nas 2 primeiras rodadas e depois vai diminuindo até zero |
| `train_one_epoch` | uma passada por todas as fotos de treino; se o modelo foi podado, reaplica os zeros a cada passo |
| `evaluate` | mede o modelo num grupo de fotos, sem ensinar nada |
| `sanity_check` | ensaio de 100 passos: o erro cai? a memória cabe? qual a velocidade? |
| `main` | laço principal: treina, mede na validação, salva `last.pt` sempre e `best.pt` quando melhora; no fim dá a prova ao `best.pt`. Opções: `--resume` continua de onde parou, `--sparsity` poda, `--frac-column` usa menos fotos |

### `eval.py` — dá a prova a um modelo já pronto
`main` carrega um `best.pt`, avalia no grupo pedido e grava `metrics_<grupo>.json`. Com `--save-preds` grava a resposta foto por foto (usado na análise dos erros).

### `reduce_data.py` — etapa 4
| função | o que faz |
|---|---|
| `run_name`, `train_cmd` | montam o nome e o comando de cada treino (fração × repetição) |
| `main` | roda os 18 treinos em fila; pula os prontos e retoma os interrompidos; `--dry-run` só lista |

## 3. Deixar o modelo menor — `src/compress/`

### `pruning.py` — apagar conexões
| função | o que faz |
|---|---|
| `prunable_weights` | lista os pesos que podem ser apagados (camadas de convolução e a final) |
| `global_magnitude_masks` | junta **todos** os pesos numa fila, ordena pelo tamanho e marca os menores para apagar. Uma régua só para a rede inteira |
| `apply_masks` | zera os pesos marcados. Chamada depois de cada passo do treino, senão eles "voltam" |
| `sparsity_report` | confere quantos zeros existem de verdade |

### `quantize_cpu.py` — modelo compacto para o processador
`main`: prepara a rede, passa 1.024 fotos de treino para medir a faixa de valores de cada camada (calibração), converte os números de 32 para 8 bits, salva o modelo compacto e dá a prova final a ele. `evaluate_cpu` é a avaliação rodando no processador.

### `export_onnx.py` — converte para um formato universal
| função | o que faz |
|---|---|
| `export` | salva o modelo em ONNX, formato que outros programas entendem |
| `parity` | confere que o modelo convertido dá as mesmas respostas que o original |

### `quantize_onnx_qdq.py` — modelo compacto para a placa de vídeo
| função | o que faz |
|---|---|
| `IncrementalMinMax`, `_incremental_histogram_collect` | fazem a calibração em pedaços pequenos. Sem isso ela usava 53 GB de memória |
| `Reader` | entrega as fotos de calibração |
| `main` | gera o ONNX com as marcas de 8 bits, usando o método de calibração escolhido por ser estável (percentil 99,99) |

### `trt_build.py`, `trt_runtime.py`, `trt_eval.py` — TensorRT (acelerador da NVIDIA)
| função | o que faz |
|---|---|
| `make_fp16_onnx` | versão de meia precisão do modelo |
| `build` | constrói o "motor" otimizado para esta placa de vídeo, em precisão total, meia ou 8 bits |
| `inspect_engine` | confere em que precisão cada camada ficou |
| `TRTRunner` | roda um motor pronto |
| `trt_eval.main` | dá a prova final ao motor |

## 4. Medir — `src/measure/`

| arquivo / função | o que faz |
|---|---|
| `cost.py` · `count_params` | quantos números o modelo tem, e quantos não são zero |
| `cost.py` · `count_macs` | quantas contas ele faz para uma foto |
| `cost.py` · `disk_sizes` | tamanho do arquivo, normal e comprimido |
| `latency.py` · `measure` | roda o modelo 50 vezes para aquecer e 300 para medir; devolve o tempo típico e o tempo dos piores casos |
| `latency.py` · `load_artifact` | carrega qualquer tipo de modelo (original, compacto, ONNX, TensorRT) de forma uniforme |

## 5. Juntar resultados — `results/`

Cada script lê as pastas `runs/` e gera CSV, gráficos e a tabela do README daquela etapa. **Nenhum número é digitado à mão.**

| arquivo | o que produz |
|---|---|
| `eixo1/make_eixo1.py` | comparação dos dois modelos; `signal_check` diz se a diferença é maior que a variação entre repetições |
| `eixo2/make_pruning.py` | erros por nível de poda; `knees` acha o ponto em que o modelo quebra |
| `eixo2/make_quant.py` | erros do modelo compacto, por programa que o executa |
| `eixo2/make_prune_compare.py` | apagar antes × depois × controle |
| `eixo3/make_eixo3.py` | tabela geral (qualidade × tamanho × velocidade); `decide` aplica a regra e nomeia a versão vencedora |
| `eixo4/make_eixo4.py` | curva de erros por quantidade de fotos; `powerlaw` ajusta a lei matemática da curva |
| `analise/make_erros.py` | quais fotos os dois modelos erram, e a grade de exemplos |

## 6. Ferramentas — `scripts/` e `app/`

| arquivo | o que faz |
|---|---|
| `queue.sh`, `queue_prune.sh`, `queue_prune_first.sh`, `queue_rewind.sh`, `queue_trt.sh` | filas: rodam vários treinos em sequência, pulando os prontos e retomando os interrompidos |
| `gpu_wait.sh` | espera a placa de vídeo ficar livre antes de começar um treino |
| `verify_manifest.py`, `verify_determinism.sh` | conferem que os manifestos estão íntegros e saem idênticos em qualquer execução |
| `audit_leakage.py`, `audit_multiaccount.py`, `make_holdout.py` | a auditoria: fotos repetidas, pessoas com dois cadastros, prova com pessoas de fora do projeto |
| `calib_sweep.sh` | compara 4 métodos de calibração do modelo compacto |
| `measure_cpu_latency.sh`, `measure_gpu_latency.sh` | medem a velocidade de todas as versões |
| `plot_curves.py`, `compact_log.py`, `md2pdf.py`, `freeze.sh`, `publish_release.sh`, `smoke.sh` | curvas de treino; limpar registros; gerar PDF; fixar versões dos programas; publicar modelos; ensaio geral de 3 minutos |
| `app/webcam_demo.py` | demonstração: `choose_model` mostra o menu, `Predictor` carrega qualquer tipo de modelo, `frame_source` lê a câmera, `draw` desenha o resultado |

`tests/` tem 26 testes automáticos: pessoas nunca em dois grupos, máscaras corretas, poda com a quantidade certa de zeros, treino que continua igual depois de interrompido, medição de velocidade estável.

---

## Por que 224 e por que espelhar

As fotos são guardadas em 256×256, mas a rede recebe 224×224. É de propósito, e está em `build_transforms` (`src/datamodule.py`).

**No treino**, cada vez que uma foto é usada:
1. **Recorte aleatório de 224×224** dentro dos 256. A mão aparece um pouco mais à esquerda, à direita, acima ou abaixo a cada vez.
2. **Espelhamento horizontal** em metade das vezes, como num espelho.

**Para quê:** a rede vê a mesma foto 30 vezes durante o treino. Se ela fosse sempre idêntica, a rede poderia decorar a foto em vez de aprender o gesto. Com as pequenas variações, ela nunca vê exatamente a mesma imagem duas vezes, e é obrigada a aprender o que não muda: o formato da mão. É como estudar com exercícios parecidos em vez de decorar o gabarito. O nome técnico é *aumento de dados*.

**Por que 224:** é o tamanho em que essas redes foram originalmente treinadas, então é onde elas funcionam melhor. Guardar em 256 deixa uma folga de 32 pixels para o recorte poder "passear".

**Por que espelhar é seguro aqui:** no HaGRID, mão esquerda e mão direita têm o mesmo nome de gesto. Um "ok" espelhado continua sendo "ok".

**Por que NÃO giramos nem viramos de cabeça para baixo:** existem gestos como `peace` e `peace_inverted`, que são o mesmo gesto girado. Girar a foto transformaria um no outro e ensinaria a resposta errada. Um teste automático garante que ninguém adicione rotação por engano.

**Na validação e na prova não há sorteio:** usa-se sempre o recorte central de 224. Assim o resultado de um modelo é o mesmo toda vez que é medido.
