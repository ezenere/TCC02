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

### 2.1 Poda (pruning)

**Técnica.** Poda **global, não-estruturada, por magnitude** (`src/compress/pruning.py`):
um único limiar de |w| sobre todos os pesos de `Conv2d` e `Linear` do modelo (biases e
BatchNorm excluídos), de modo que a fração `s` de menor magnitude é zerada. O limiar global
deixa a esparsidade se distribuir de forma desigual entre camadas — as camadas mais
redundantes perdem mais pesos — o que é a vantagem do critério global sobre o por-camada.

**Máscaras explícitas.** As máscaras são tensores booleanos persistidos no checkpoint
(`prune_masks`), e não a reparametrização de `torch.nn.utils.prune` (que instala hooks
`weight_orig`/`weight_mask` e contamina exportação ONNX e quantização). Após **cada**
`optimizer.step()` do fine-tuning as máscaras são reaplicadas (`apply_masks`), garantindo
que os zeros permaneçam; ao final do run a esparsidade obtida é medida a partir dos zeros
reais dos pesos e um assert falha se divergir do alvo em mais de 0,1 p.p.

**Protocolo.** Para cada arquitetura e seed: parte-se do `best.pt` do eixo 1 da **mesma
seed**, aplica-se a poda no nível `s ∈ {50%, 70%, 90%}` (estendendo a 95% e 98% se a razão
de erro em 90% ainda for < 2×), e faz-se fine-tuning de 5 épocas em `fit` com SGD
(lr 0,00375 = 0,1× o inicial, cosine sem warmup, demais hiperparâmetros idênticos ao eixo 1),
seleção por F1 macro em `val` e avaliação única no teste (`configs/prune_*.yaml`,
`src/train.py --init-from ... --sparsity ...`). Reporta-se acurácia, F1 macro, taxa de erro,
**razão de erro em relação ao baseline da mesma seed**, parâmetros não-nulos e tamanho do
`state_dict` comprimido (gzip), que reflete a esparsidade.

**Limitação registrada: poda estruturada em DenseNet.** A poda estruturada (remoção de
filtros/canais inteiros, a que de fato reduz MACs e latência sem hardware esparso) é
diretamente aplicável à ResNet, mas na DenseNet cada camada concatena as saídas de todas as
anteriores do bloco; remover um canal de uma camada altera a entrada de todas as seguintes e
das transições, exigindo propagação de índices por todo o bloco denso. Por isso o protocolo
usa poda **não-estruturada** nas duas arquiteturas — comparável entre elas — e registra
que, nesse regime, o ganho de latência em GPU/CPU densos é nulo ou pequeno: o que a poda
reduz é o número de parâmetros efetivos e o tamanho do artefato comprimido. Esta é uma
constatação a reportar, não uma falha de implementação.

**Testes.** `tests/test_pruning.py`: esparsidade obtida = alvo ± 0,1 p.p. em 50/70/90%;
limiar global (esparsidade difere entre camadas); os pesos removidos são os de menor
magnitude; máscaras sobrevivem a passos do otimizador com momentum e weight decay;
round-trip por checkpoint preserva máscaras e saídas.

### 2.2 Quantização

**Duas rotas, um artefato por backend.** A quantização é medida em dois runtimes distintos, porque
acurácia e latência de um modelo int8 dependem do backend e não se transferem entre eles:
(a) **CPU** — PTQ estática int8 nativa do PyTorch; (b) **GPU** — engine TensorRT com calibração int8
(seção a preencher). Em cada backend o baseline FP32 é medido **no mesmo runtime** (PyTorch eager em
CPU; TensorRT FP32/FP16 em GPU), para que o ganho medido seja da precisão numérica e não da troca de
runtime.

**PTQ em CPU (`src/compress/quantize_cpu.py`).** FX graph mode (`prepare_fx` → calibração →
`convert_fx`) com o `qconfig_mapping` padrão do backend `x86` (kernels fbgemm): fusão
conv+bn+relu, observadores por tensor nas ativações e por canal nos pesos. **Calibração com 1.024
imagens de `fit`** (amostra determinística, seed 0) — nunca do teste. O modelo convertido é
serializado em TorchScript (`model_int8_fbgemm.pt`), e é esse artefato — recarregado do disco —
que é avaliado no teste completo em CPU (`torch.set_num_threads` declarado). A DenseNet-121
quantizou pelo mesmo caminho FX, sem tratamento especial das concatenações; o fallback previsto
(quantização estática via ONNX Runtime) não foi necessário. Critério de escalada: se a queda de
acurácia excedesse 1 p.p., passar a QAT — não ocorreu (seed 0: ResNet-50 razão de erro 0,96;
DenseNet-121 ver `results/eixo2/`).

**Reportado por célula:** acurácia, F1 macro, taxa de erro e razão de erro vs baseline da mesma
seed; tamanho do artefato serializado (TorchScript int8 vs `state_dict` FP32); backend, nº de
threads e throughput da avaliação. A FX graph mode quantization está marcada como legada no
PyTorch 2.11 (migração para `torchao`); a versão exata está pinada em `requirements.txt` e
registrada em cada `metrics_int8_fbgemm.json`.

### 2.3 TensorRT (GPU)
_(a preencher: engines FP32/FP16/INT8, calibrador de entropia com as mesmas 1.024 imagens, perfis batch 1 e 32)_

## 3. Eixo 3 — Benchmark

**Células.** {ResNet-50, DenseNet-121} × {baseline, poda@{50,70,90,95,98}%, int8-CPU (fbgemm),
int8-GPU (TensorRT), poda@joelho + int8}. Cada célula existe em 3 seeds (a técnica é aplicada ao
modelo do eixo 1 da mesma seed), e tudo é reportado como média ± desvio entre seeds.

**Métricas de qualidade.** Acurácia, F1 macro e taxa de erro no teste; **razão de erro** em relação ao
baseline da mesma seed — a métrica primária, porque o teto está saturado (~0,2% de erro) e deltas de
acurácia são ilegíveis. A leitura é feita como **localização do joelho**: o primeiro nível de compressão
em que a razão de erro excede 1,5×.

**Métricas de custo** (`src/measure/cost.py`): parâmetros totais e **não-nulos** (os zerados pela poda
não contam), MACs a 224×224 (fvcore; 1 multiply-add = 1), tamanho em disco do artefato que de fato
seria implantado — `state_dict` FP32 para o baseline, o mesmo comprimido com gzip para os podados
(a esparsidade não-estruturada só se materializa em disco após compressão), TorchScript int8 e engine
TensorRT para os quantizados.

**Latência** (`src/measure/latency.py`): entrada sintética, 50 iterações de aquecimento e 300 medidas por
lote; p50, p95 e média; lotes 1 e 32. **CPU** (Ryzen 9 9950X): PyTorch eager FP32 e TorchScript int8
(fbgemm) no mesmo processo, com 1 e 16 threads declarados. **GPU** (RTX 3080 Ti): PyTorch eager
FP32/FP16 e engines TensorRT FP32/FP16/INT8 — o baseline é medido também em TensorRT, para que o ganho
de int8 não se confunda com o ganho de runtime. Cada JSON de latência grava hardware, versões, threads
e uma nota manual sobre o estado da sessão gráfica; as medições de GPU são feitas em TTY com o
compositor fechado. Os modelos podados são medidos uma vez para documentar que kernels densos não
exploram esparsidade não-estruturada (latência ≈ baseline).

**Decisão** (`results/eixo3/make_eixo3.py --decide`, critério fixado em 2026-09-14 antes dos
resultados): entre as células com razão de erro média ≤ 1,5×, vence a de **menor latência p50 em GPU
(TensorRT, lote 1)**; empate → menor artefato em disco. A configuração vencedora alimenta o eixo 4;
`results/eixo3/decision.md` lista as finalistas e o motivo de cada derrota.

**Figuras.** Fronteiras de Pareto razão de erro × latência (CPU e GPU em painéis separados), × tamanho
em disco e × MACs, com barras de erro entre seeds.

## 4. Eixo 4 — Redução de dados anotados
_(a preencher)_
