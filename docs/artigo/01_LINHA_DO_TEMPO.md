# 01: Linha do tempo do projeto (28/08/2026 a 18/10/2026)

Cronologia do que foi implementado, executado e decidido, com incidentes e correções. Fontes: `STATUS.md` (diário), `git log` (fuso −03:00), logs de fila em `runs/queue_*.log`, `runs/*/run_meta.json`, `data/processed/MANIFESTS.md`.

**Nota sobre datas.** O diário `STATUS.md` rotula os dias pelo plano de trabalho (D0914, D0915, D0922...). A execução correu adiantada: os horários dos logs de fila e dos commits mostram, por exemplo, que a fila do eixo 1 começou no domingo 13/09 às 15:49, não na terça 15/09. Esta linha do tempo usa as datas dos logs e dos commits. Quando o diário registra um fato sem horário de máquina (reuniões, pedidos do orientador), a data é a do diário.

## Visão geral

| período | o que aconteceu | GPU |
|---|---|---|
| 28–29/08 | dados, pré-processamento, baseline de evidência, release `ResNet` | ~16 min |
| 30/08–12/09 | planejamento; sem execução registrada | 0 |
| 13/09 | manifesto v2, `train.py` definitivo, início do eixo 1 | eixo 1 (início) |
| 14/09 | eixo 1 fechado; poda (30 runs); int8 CPU; incidente de OOM; TensorRT; manifesto v3; reunião (8 decisões) | ~20 h |
| 15–18/09 | ONNX, latência CPU parcial, pauta; sem GPU | 0 |
| 19/09 | pedidos do orientador: auditoria, poda antes do treino, app de webcam; eixo 2b seed 0 lançado | eixo 2b (início) |
| 20/09 | releases v0.2 e v0.3; eixo 2b seed 0 fechado; latência CPU e GPU (preliminar); decisão do eixo 3; eixo 4 lançado | ~26 h |
| 21/09 | anomalia do early stopping, correção e continuação; calibração instável, varredura e 28 células refeitas; eixo 2c (12 runs); relatório | ~24 h |
| 22/09 | eixo 2c fechado; eixo 2b seeds 1–2 iniciado e pausado a pedido do autor | ~4 h |
| 23–29/09 | pausa | 0 |
| 30/09–01/10 | retomada; fila morta por limite de tempo e relançada desacoplada; eixo 2b fechado; extra B; release v0.5 | ~8,7 h |
| 02–16/10 | sem GPU; pendências com o orientador | 0 |
| 17/10 | eixo 2d: poda sem retreino (30 avaliações, 0 épocas); redação desta documentação; latência GPU medida em TTY pelo autor (`scripts/latency_tty.sh`) | ~25 min + ~22 min |
| 18/10 | experimento de variância da latência GPU; `--rounds` no harness; todas as células de GPU remedidas (5 rodadas); decisão do eixo 3 confirmada | ~15 min |

Total aproximado de GPU em treinos, engines e avaliações: **≈ 84 h**, soma dos logs de fila (≈ 83 h até 01/10 + ~25 min do eixo 2d em 17/10 + ~40 min de engines e medições de latência em 17–18/10). Runs de treino: 6 (eixo 1) + 30 (eixo 2) + 16 (eixo 2b) + 12 (eixo 2c) + 18 (eixo 4) + 1 (extra B) + 1 (controle de rótulos embaralhados) + 1 (baseline de 5 épocas) = 85, todos com 0 falhas de execução (as interrupções foram externas e retomadas de `last.pt`). Além deles, 30 células de avaliação sem treino (eixo 2d, poda sem retreino), também com 0 falhas.

## Dia a dia

### 2026-08-28 (qui): dados e pré-processamento (etapa E2)

- `src/validate_raw.py`: 0 imagens ausentes, 0 bboxes malformadas, 1 imagem descartada por só ter mãos `no_gesture` (`results/raw_validation.csv`).
- `src/preprocess.py`: subconjunto de 50% das 18 classes por `user_id` (guloso balanceado por classe, seed 20250828) → **278.715 imagens, 20.117 sujeitos**; split 70/30 por sujeito (treino 195.102 / 14.073 sujeitos; teste 83.613 / 6.044; interseção 0); crop quadrado com margem de 10% (3.265 janelas deslocadas, 87 com lado reduzido), 256×256 JPEG q90, 4,2 GB.
- Manifesto v1 (`manifest.csv`, sha `df8e43a3…`); `scripts/verify_determinism.sh` confirma manifesto byte-idêntico em dois processos com `PYTHONHASHSEED` distintos.
- Decisão registrada: no overflow do crop, reduzir o lado a `min(W, H)` em vez de pad por replicação (a bbox nunca é cortada).
- Artefatos: `results/subset_stats.csv` (teste entre 29,35% e 30,74% em todas as classes), figura de amostras `results/figures/fig2_amostras_classes`.

### 2026-08-29 (sex): baseline de evidência e primeiro commit

- `src/datamodule.py`, `src/train.py` (`--sanity N`), `make_baseline_figures.py`.
- `runs/resnet50_baseline_5ep/`: ResNet-50, 5 épocas, manifesto v1, última época: **99,79% acurácia / 99,79% F1 macro**, 185,7 s/época. Teto quase saturado.
- Release `ResNet` publicada no GitHub (`best.pt` + `last.pt`, 180 MB cada). Commit `40f79c3` ("First Commit"), repositório `github.com/ezenere/TCC02`.

### 2026-08-30 a 2026-09-12

Sem execução registrada. Planejamento das decisões a levar ao orientador (métrica de erro, critério do eixo 3, receita do eixo 4, 3 seeds, validação interna, célula poda + int8, tags).

### 2026-09-13 (dom): "Dia 0" do diário: manifesto v2, treino definitivo, eixo 1 lançado

- Auditoria do repositório: 1 commit, remote OK, `.gitignore` exclui `data/`, checkpoints e `*.pt`; `requirements.txt` fiel ao env (152 pins); `run_meta.json` do baseline sem `git_commit` (corrigido no mesmo dia).
- **Manifesto v2** (`3c32da4`): 10% dos sujeitos do treino em `val` (seed 20250830 = base + 2): fit 175.591 / 11.805 sujeitos, val 19.511 / 2.268, test idêntico. Sha `ed5c1eb7…`. Byte-idêntico entre processos; `verify_manifest.py --version 2` sem violações. Registro `data/processed/MANIFESTS.md`.
- **`train.py` definitivo** (`590f3d6`): fit/val/test, `best.pt` por F1 macro em val, warmup 2 épocas + cosseno, `--seed`, `--resume` (replay exato dos lotes; `persistent_workers` desligado para isso), `--frac-column`, `--patience`, `git_commit` e `git_dirty` no `run_meta`. `src/eval.py`. Testes (`tests/`, 9 verdes). Smoke de 100 steps: ResNet-50 4,26 GiB / 1.093 img/s; DenseNet-121 6,10 GiB / 895 img/s.
- Correção ao plano: ResNet-50 com cabeça de 18 classes tem 23,5 M parâmetros (não 25,6 M); DenseNet-121 7,0 M.
- **Fila do eixo 1 lançada às 15:49** (`scripts/queue.sh eixo1`, `a3344d6`): 6 runs seriais, s0 → s1 → s2, ResNet antes de DenseNet. Run 1 `eixo1_resnet50_s0` terminou 17:15 (86 min): 177 erros, best época 28 (`db058a8`). O run começou com árvore suja (`40f79c3`); commit `714b431` feito durante o run 2 para que os runs 3–6 saíssem limpos.
- `src/compress/pruning.py` (`40dd401`): poda global por magnitude com máscaras explícitas, `apply_masks` após cada step, assert de esparsidade; 7 testes verdes; smoke de 1 época a 50%.
- Runs 2–5 do eixo 1 durante a noite (`7593510`, `4c404c5`, `b88e6d4`, `8a8bbd4`).

### 2026-09-14 (seg): dia mais denso: eixo 1 fechado, poda, int8, incidente, TensorRT, reunião

**Madrugada.**
- 01:32 fila do eixo 1 concluída: 6 runs, **9,7 h de GPU, 0 falhas** (`bcafa82`). DenseNet 0,197 ± 0,005% vs ResNet 0,207 ± 0,009% de erro: **sem sinal** (|Δ| 0,010 p.p. < 2× desvio 0,017). `results/eixo1/`.
- 01:35–03:19 poda seed 0 em 50/70/90% (6 runs, 1,7 h; `868db0e`). Razão de erro em 90%: ResNet 1,07×, DenseNet 1,02× → regra automática dispara a extensão para **95% e 98%** (`b453a76`). 03:20 fila 95/98 iniciada.
- Antecipado em CPU: `quantize_cpu.py` (FX, fbgemm, 1.024 imagens de `fit`), seed 0: ResNet 170/177 erros (0,96×), DenseNet 210/160 (1,31×). Primeiro sinal assimétrico entre arquiteturas (`4750116`). Exportação ONNX com paridade (`a164be5`).

**Incidente (tarde).** Dois OOM, 30 GB e 53 GB de RSS, na calibração por entropia do ONNX Runtime, que retém todas as ativações intermediárias das 1.024 imagens. O segundo foi erro de execução: um patch para MinMax falhou na asserção e o quantizador rodou com o código antigo. Os OOM derrubaram a sessão e a fila de poda; o run `eixo2_prune_resnet50_p98_s0` retomou do `last.pt` (época 2) às 13:48. **Correções:** coletor de histogramas incremental (lote a lote) e correção de um bug de flush do ORT; toda execução desse tipo sob `systemd-run --user --scope -p MemoryMax=16G` (regra nova); patches encadeados com `&&` e verificados por `grep` antes de rodar. Grafo Q/DQ da ResNet-50 (177 `QuantizeLinear`) e da DenseNet-121 (556) em 28 s cada, memória contida.

**TensorRT 11.** Sem `IInt8EntropyCalibrator2` nem flags `FP16`/`INT8`: redes fortemente tipadas, a precisão vem do ONNX (fp32 / fp16 convertido / QDQ). `trt_build.py` reescrito (`bb6c2bb`, `56110bd`). Engine FP32 da ResNet s0: 176 erros vs 177 do PyTorch, paridade OK.

**Tarde.** Fila de poda seeds 1–2 (14:19 → ~20:00). Total da poda: **30 runs, ~8,5 h, 0 falhas** (`10d9402`). Joelho (razão > 1,5×) em 98% nas duas arquiteturas. int8 CPU seeds 1–2 (`771fe9b`): ResNet 1,04 ± 0,07×, DenseNet 1,24 ± 0,07× (Δ 0,20 > 2× desvio: sinal). Manifesto v3 (`da17d0f`, sha `40f09516…`, 25 máscaras aninhadas) e `src/reduce_data.py` (`aa471b9`). Harness de latência (`1f1872b`), consolidação do eixo 3 com regra de decisão (`626078e`) e do eixo 4 com lei de potência (`f758896`). `requirements.txt` com `tensorrt-cu12 11.3` (`b0d6549`).

**Noite.** Engines TensorRT da seed 0, calibrador escolhido em `val` com uma amostra (entropia 61 erros vs MinMax 131 na ResNet; 63 vs 67 na DenseNet) → entropia adotada (`2b31948`, `12c2f3e`). Resultados seed 0: TRT FP32/FP16 1,00× nas duas; INT8 ResNet 1,02× (6.227 img/s), DenseNet 1,19× (1.763 img/s, **mais lenta que a FP16** a 3.183 img/s). 20:50–21:23 fila TRT seeds 1–2 (`edf3eed`); 21:23–21:42 poda@90/95 + int8 TRT (seed 0). `results/eixo2/README.md` com tabelas geradas (`53ad1e8`). Medição de latência CPU iniciada em background (interrompida na metade: ResNet completa, DenseNet só 1 thread).

**Reunião com o orientador (14/09): 8 decisões aprovadas.**
1. Métrica primária de degradação: taxa de erro e razão de erro vs baseline; eixo 3 como localização do joelho de esparsidade.
2. "Melhor configuração" para o eixo 4: menor latência GPU (TensorRT, lote 1) entre as células com razão ≤ 1,5×; empate → menor artefato.
3. Eixo 4: 30 épocas fixas + early stopping por validação interna ("reduz dados, mantém receita").
4. Eixo 1 com 3 seeds por arquitetura (0, 1, 2).
5. Validação interna por sujeitos (manifesto v2) confirmada.
6. Célula extra poda + int8 no benchmark autorizada.
7. Feriados 12/10 e 02/11: pauta por escrito.
8. Tags `v0.2-eixo1`, `v0.3-eixo2`, …; release `ResNet` mantida; publicação via `scripts/publish_release.sh` (curl + token; `gh` não instalado).
Decisões menores do dia: linhas de teste do manifesto v2 recebem `inner_split = "test"`; seeds de treino são 0/1/2; `metrics.json` = teste no `best.pt`; MACs pelo fvcore (1 MAC = 1).

### 2026-09-15 a 2026-09-18 (ter–sex): trabalho de CPU, sem GPU

- `src/compress/export_onnx.py`: opset 17, batch dinâmico, paridade ORT × PyTorch em 64 imagens de `val` (ResNet max |Δ| 5,7e-6; DenseNet 2,9e-6; 0 argmax divergentes). Observação: em CPU o ORT é 35% mais rápido que o eager na ResNet e 30% mais lento na DenseNet; a latência precisa ser declarada por backend.
- Pauta da reunião de 21/09 preparada (eixo 1 fechado; níveis de poda; interpretação do eixo 3; limitação da poda estruturada em DenseNet).
- Pendente: publicar `v0.2-eixo1` (manual); latência CPU completa; TensorRT em TTY.

### 2026-09-19 (sáb): pedidos do orientador: auditoria, poda antes do treino, app de webcam

- **Auditoria de vazamento completa** (`7f736a2`, `f6d7e94`; `results/audit/README.md`): sujeitos disjuntos; manifesto = JSONs brutos em 100% das 278.715 linhas; 0 duplicatas exatas (md5); controle de rótulos embaralhados = 4,9% (acaso 5,6%); **achado:** `user_id` não é identidade (87 imagens de teste, 0,10%, quase idênticas a imagens de treino sob outro `user_id`); remoção de usuários suspeitos: erro inalterado no critério estrito, 0,28% no frouxo (pior caso 99,72%); **holdout externo** de 278.702 imagens de 19.042 sujeitos nunca usados: 0,235% ResNet / 0,216% DenseNet.
- **Eixo 2b** (poda antes do treino): configs `prune_first_<arch>.yaml`, `prune.exclude_head: true`, `scripts/queue_prune_first.sh`. Fila seed 0 lançada às 22:03 (10 runs de 30 épocas, 5 níveis × 2 arquiteturas, ETA 16 h).
- App de webcam `app/webcam_demo.py` (PyTorch, int8 TorchScript, ONNX Runtime, TensorRT; 18/18 corretos em teste headless nos quatro backends).
- `.gitignore` corrigido (`4fe486f`: linhas `TODO` e `*.onnx` coladas). Análise qualitativa dos erros (`4ff42d0`): 117 das ~170 imagens erradas são comuns às duas arquiteturas, 109 com a mesma predição; `stop` ↔ `palm` = 36%. `README.md` da raiz, `docs/RUNBOOK.md`, `Makefile`, `scripts/smoke.sh` (`7f4f019`). 24 testes de CPU verdes.
- Decisão por padrão (pendente de objeção): fração 100% do eixo 4 é run próprio com a mesma receita.

### 2026-09-20 (dom): releases, eixo 2b fechado na seed 0, latência, decisão preliminar, eixo 4 lançado

- Releases **`v0.2-eixo1`** (tag em `bcafa82`, 12 assets, 1,4 GiB; `5be74c3`) e **`v0.3-eixo2`** (tag em `edf3eed`, 30 modelos podados + 10 artefatos int8 CPU, 40 assets, 4,0 GiB; `9303a62`). `publish_release.sh` corrigido para nomear assets pelo diretório do run (`e435fcd`).
- 14:04 **eixo 2b seed 0 concluído**: 10 runs, 0 falhas, ~16 h (`c5552d9`). Antes × depois (razão): ResNet 98% 1,36 × 1,90; DenseNet 98% 1,64 × 1,65. Confusão declarada: ordem × orçamento (30 × 5 épocas). Controle previsto: podar depois com 30 épocas.
- **Latência CPU** completa, máquina em repouso (14:17): ResNet FP32 51,7 / 10,7 ms (1 / 16 thr), int8 8,0 / 1,38 ms; DenseNet 41,3 / 16,6, int8 7,2 / 2,86; podados = baseline. **Latência GPU preliminar** (14:36, sessão gráfica aberta, uma rodada; substituída pela medição final de 18/10): TRT ResNet FP32 2,65 / FP16 0,94 / INT8 0,80 ms; DenseNet 4,23 / 2,76 / 3,17 (int8 mais lenta que fp16). `trt_runtime.py` passou a usar stream CUDA dedicado. Aviso do TensorRT sobre engines de outro "modelo de dispositivo": reconstruir com `make trt` antes da medição final em TTY.
- **Decisão do eixo 3, preliminar** (`1a0b269`, `results/eixo3/decision.md`): leitura literal → ResNet-50 poda 90% + int8 TRT (0,793 ms, 1,42× com calibração por entropia), por 0,9% sobre ResNet-50 int8 TRT (0,800 ms); refinamento proposto com tolerância de 5% → ResNet-50 int8 TRT. Arquitetura ResNet-50 em qualquer leitura.
- 14:38 **fila do eixo 4 lançada**: `reduce_data.py`, ResNet-50, 6 frações × 3 seeds, 30 épocas + paciência 5 (`c620694`: calibração int8 respeita a fração do run; `43e47ae`).

### 2026-09-21 (seg): early stopping descartado, calibração refeita, eixo 2c, relatório

**Madrugada.**
- 01:23 eixo 4 terminou, 18/18, 0 falhas. **Anomalia:** 100% da seed 2 com 277 erros (177 e 182 nas outras), pior que 75% e 50% da mesma seed. **Diagnóstico:** parada antecipada com paciência 5 parou o run na época 15 com o lr a 56% do pico; 13 dos 18 runs pararam antes da época 30 e eram os piores de cada fração. **Correção:** `--patience 0`, 30 épocas fixas, `best.pt` por `val` (`1879413`, `72df340`); os 13 runs continuados de `last.pt` até a época 30 (01:23 → 03:07); resultados antigos em `metrics_es5.json`. Sinalizado ao orientador como ajuste de uma decisão fechada.
- 03:07–03:58 int8 TensorRT dos 18 pontos (entropia). **Achado:** o mesmo modelo foi de 194 a 2.596 erros no teste conforme a amostra de calibração. **Varredura em `val`** (`scripts/calib_sweep.sh`, 4 amostras): percentil 99,99 → 62–64 erros; 99,999 → 62–68; entropia → 66–689; MinMax → 604–966 (denso 60). **Novo padrão: percentil 99,99** (`28309f6`, `2c713cc`). **28 células refeitas** (6 do eixo 1, 4 de poda+int8, 18 do eixo 4), 0 falhas (`fd0b42f`): ResNet TRT int8 1,08 → 1,01×; DenseNet 1,19 → 1,07×; poda 90% + int8 TRT ResNet 252 → 195 erros. Duas afirmações dos READMEs corrigidas.
- **Incidente `gpu_wait`** (04:39, `fa6c89a`): a fila do eixo 2c, lançada com `scripts/gpu_wait.sh` para compartilhar a GPU com o autor, ficou travada ~40 min porque o compositor `kwin_wayland` aparece no `nvidia-smi` como processo de compute. Correção: ignorar processos com menos de 500 MiB.
- **Incidente menor:** um `rm -f` com glob sem correspondência não rodou (o zsh aborta o comando inteiro); detectado antes de a fila reaproveitar engines antigas; limpo com `find -delete` e conferido. Regra registrada.
- 04:39–11:39 **eixo 2c seed 0** (controle por LR rewinding, 4 runs; `12c1057`, `0fd1fee`): a 98%, depois + 30 ép. dá 1,10× (ResNet) e 1,33× (DenseNet) contra 1,36× / 1,64× antes + 30 ép. e 1,90× / 1,65× depois + 5 ép. **O orçamento de retreino decide, não a ordem.**

**Dia.**
- Eixo 4 fechado na curva densa (`5a4da53`): 75% 1,05 · 50% 1,28 · 25% 1,41 · 10% 2,33 · 5% 3,13; erro ∝ N^−0,38 (R² 0,98); int8 acompanha (N^−0,40).
- Decisão do eixo 3 atualizada com a int8 recalibrada: **ResNet-50 int8 TensorRT**, razão 1,01×, 0,80 ms, 23,9 MiB (preliminar à época; confirmada em 18/10).
- 11:41 → 22/09 00:39: eixo 2c seeds 1–2 (8 runs).
- `RESULTS.md` (`cbee9de`); relatório em linguagem simples com figuras próprias e PDF (`9080977`, `de37a87`, `c76528a`; `docs/relatorio/`); `docs/MAPA_CODIGO.md` (`ff346a2`).

### 2026-09-22 (ter): eixo 2c fechado; pausa a pedido do autor

- 00:39 **eixo 2c completo**: 12 runs, 3 seeds, 0 falhas. A 98%: 1,12 ± 0,02× (ResNet) e 1,23 ± 0,09× (DenseNet). Conclusão sustentada com 3 seeds (`652f3bc`).
- 00:39 fila do eixo 2b seeds 1–2 (95/98%) iniciada; 2 runs concluídos até 03:57 (`p98_s1` das duas arquiteturas), `resnet50_p95_s1` em andamento.
- 05:00 **pausa** a pedido do autor (GPU necessária). Nada do projeto roda na GPU até 30/09.

### 2026-09-23 a 2026-09-29

Pausa. Sem execução.

### 2026-09-30 (qua): retomada; fila morta por limite de tempo

- 15:41 fila do eixo 2b relançada (`queue_prune_first.sh "1 2" "0.98 0.95"`, com `gpu_wait`) como comando em background da ferramenta de execução. **Incidente:** o comando foi encerrado ao atingir um limite de tempo; fila e treino morreram na época 29/30 do run `resnet50_p95_s1`. Nada perdido: 16:12 relançada **desacoplada da sessão** (`setsid nohup … > runs/detached_2b_extraB.log`), e o run retomou do `last.pt` e terminou em 3 min. Regra registrada: filas longas (> 30 min) sempre desacopladas.
- Textos atualizados com o eixo 2c em 3 seeds (`01e6de8`); `scripts/extra_b.sh` (`70f32f5`); release **`v0.5-eixo4`** publicada (18 × `best.pt` do eixo 4, 3,2 GiB; `d938919`).
- Extra A (eixo 4 na DenseNet-121, ~10 h) deixado em espera por decisão do autor.

### 2026-10-01 (qui): eixo 2b fechado e extra B

- 00:34 **eixo 2b completo em 95/98% com 3 seeds** (6 runs restantes, 8,4 h): a 98%, 1,40 ± 0,16× (ResNet) e 1,54 ± 0,09× (DenseNet), entre o fine-tuning de 5 épocas e o retreino de 30 (`6e5c4e2`).
- 00:34–00:48 **Extra B** (ResNet-50 seed 0, 25% dos dados): poda 90% custa 1,69× (vs 1,07× com 100%), poda + int8 CPU 2,03× (vs 1,14×), int8 sozinha 1,15× (vs 0,96×). `results/analise/interacao.csv`.
- Relatório e diário atualizados (`61fcd3a`). GPU livre.

### 2026-10-02 a 2026-10-16

Sem execução de GPU. Pendências com o autor e o orientador: (1) latência GPU em TTY, após `make trt` reconstruir as engines (feita em 17/10); (2) confirmar a tolerância de empate de 5% na decisão do eixo 3 (refinada em 18/10); (3) confirmar a retirada da parada antecipada no eixo 4; (4) decidir o extra A.

### 2026-10-17 (sáb): eixo 2d (poda sem retreino) e documentação de referência

- Redação desta documentação de referência (`docs/artigo/`, `28834e7`).
- **Eixo 2d: poda sem retreino.** Novo `src/compress/prune_eval.py`: aplica as máscaras globais por magnitude do eixo 2 (cabeça incluída) ao `best.pt` do eixo 1 da mesma seed e avalia direto no teste, sem nenhuma época. **30 avaliações** (2 arquiteturas × 5 níveis × 3 seeds), ~40 s cada, **~25 min de GPU**, 0 falhas; runs `runs/eixo2d_noft_*` com `metrics.json`, `run_meta.json`, `confusion_matrix.npy`, `cost.json` e checkpoint podado. Resultado: 50% é gratuito sem retreino (1,05× / 1,01×); 70% custa 1,99× (ResNet-50) / 1,29× (DenseNet-121); a 90% a rede quebra (~14% de acurácia) e a 95/98% responde ao acaso (1/18). O fine-tuning de 5 épocas recupera tudo até 90%: acima de 50%, o retreino é obrigatório. Integrado como coluna da tabela comparativa (`make_prune_compare.py`, figura `prune_compare` em escala log), `results/eixo2/README.md`, `docs/METODOLOGIA.md` § 2.1 e diário (`b66bdc5`). GPU livre.
- **Latência GPU em TTY, pelo autor** (21:37–21:59; `scripts/latency_tty.sh`, `8ade7b5`): sessão gráfica parada (`systemctl stop sddm`), pré-condições verificadas (sem DISPLAY, sem outro processo na GPU), engines TensorRT da seed 0 reconstruídas com a GPU ociosa (baseline FP32/FP16/INT8 das duas redes + poda 90/95% + int8), 18 medições gravadas com a nota "sessão gráfica fechada (TTY)". **Resultado inesperado:** as três células int8 da ResNet-50, que rodam o mesmo grafo, saíram em 0,920 / 0,923 / **0,797** ms (int8, poda 90% + int8, poda 95% + int8) e a FP16 (0,827) ficou à frente da int8 — a leitura literal passaria a apontar poda 95% + int8. As demais células não diferiram da medição de 20/09 (eager FP32 1,81 ms nas duas; TensorRT FP32 2,645 nas duas).

### 2026-10-18 (dom, madrugada): variância da latência GPU, rodadas repetidas, remedição e decisão confirmada

- **Experimento de variância** (`results/eixo3/build_variance.md`, sessão gráfica aberta): (1) a mesma engine int8 da ResNet-50 construída três vezes varia ~4% (0,886–0,926 ms); (2) as três engines de TTY medidas em sequência ficam em 0,923–0,930 ms — o 0,797 ms de 17/10 não se reproduz, foi variância **de medição**; (3) FP16 × INT8 alternadas em 4 rodadas, clock SM estável em 1.980–2.010 MHz: cada medição isolada de lote 1 alterna entre dois patamares, ≈ 0,80 e ≈ 0,93 ms, para **qualquer** engine (ruído de ~15%, maior que a diferença entre as duas engines); a int8 é mais rápida que a FP16 nas 4 rodadas e por 25% em lote 32.
- **Harness** (`src/measure/latency.py`, `59f4132`): `--rounds N` (padrão 5 em GPU), mediana do p50 entre rodadas, p50 de cada rodada e dispersão gravados no JSON. **Regra de decisão** (`make_eixo3.py --decide`, `graph_family`): células com o mesmo grafo compartilham a latência (mediana do grupo), além da tolerância de 5% já proposta.
- 00:27–00:38 **todas as 18 células de GPU remedidas** com 5 rodadas (`66621e1`; sessão gráfica aberta; os JSONs de TTY arquivados como `*_tty1round.json`). Entre TTY e sessão aberta não há diferença sistemática. Números finais (p50 lote 1 / lote 32): ResNet-50 eager FP32 1,803 / 23,2, eager FP16 1,480 / 10,6, TensorRT FP32 2,652 / 14,74, FP16 0,823 / 4,09, INT8 0,798 / 3,10, poda 90% + int8 0,814, poda 95% + int8 0,805; DenseNet-121 eager FP32 3,497 / 25,6, eager FP16 3,327 / 13,3, TensorRT FP32 4,206 / 18,13, FP16 2,799 / 8,57, INT8 3,192 / 17,21, poda 90% + int8 3,397, poda 95% + int8 3,339.
- **Decisão do eixo 3 confirmada** (`results/eixo3/decision.md`): **ResNet-50 int8 TensorRT**, razão 1,01×, 0,805 ms (mediana do grafo; esta engine 0,798), 23,9 MiB; a leitura literal aponta a mesma célula. Vencedora inalterada em relação à preliminar de 20/09: o eixo 4 já estava na configuração certa. Textos atualizados (README, RESULTS, METODOLOGIA, relatório, documentação de referência). GPU livre.

## Incidentes e correções (resumo)

| data | incidente | causa | correção |
|---|---|---|---|
| 14/09 | dois OOM (30 e 53 GB) derrubaram a sessão e a fila de poda | calibração por entropia do ONNX Runtime retém todas as ativações intermediárias; o segundo OOM veio de um patch que falhou e deixou o código antigo rodar | coletor incremental; `systemd-run --scope -p MemoryMax=16G`; patches com `&&` + `grep` |
| 19/09 | `user_id` não é identidade | contas duplicadas no HaGRID | quase-duplicatas medidas; usuários suspeitos removidos (pior caso 99,72%); holdout externo |
| 21/09 | 13 de 18 runs do eixo 4 truncados; 100% s2 com 277 erros | early stopping com paciência 5 sobre `val` que oscila ±0,05 p.p.; agenda cosseno cortada com lr até 56% do pico | `--patience 0`; continuação dos 13 runs até a época 30; `metrics_es5.json` arquivado |
| 21/09 | mesmo modelo de 194 a 2.596 erros em int8 TensorRT | calibração por entropia instável à amostra de 1.024 imagens | varredura em `val`; percentil 99,99; 28 células refeitas |
| 21/09 | `gpu_wait` travado ~40 min | `kwin_wayland` aparece como processo de compute | ignorar processos < 500 MiB |
| 21/09 | `rm -f` não executou | glob sem correspondência aborta o comando no zsh | `find -delete` + conferência |
| 30/09 | fila e treino mortos na época 29/30 | comando em background da ferramenta encerrado por limite de tempo | `setsid nohup`, logs em `runs/`, retomada de `last.pt` |
| 13/09 | dois runs com árvore suja no `run_meta` | fila lançada antes do commit | commit durante a fila; nota de proveniência |
| 17/10 | medição única em TTY inverteu a ordem de três células int8 da ResNet-50 que rodam o mesmo grafo (0,797 vs 0,920 ms) | a latência de lote 1 em GPU alterna entre dois patamares ~15% apart entre medições, com clock estável | `--rounds 5` (mediana) no harness; remedição de todas as células; latência compartilhada entre células do mesmo grafo |

## Decisões do orientador e ajustes sinalizados

| data | decisão | status |
|---|---|---|
| 14/09 | 8 decisões (acima) | fechadas |
| 19/09 | auditoria de vazamento; poda antes do treino (eixo 2b); app de webcam | executadas |
| 21/09 | retirada do early stopping no eixo 4 (ajuste de parâmetro da decisão 3 de 14/09) | sinalizado; confirmação pendente |
| 20/09 | tolerância de empate de 5% na decisão do eixo 3 (refinamento da decisão 2) | proposto; confirmação pendente; vencedora igual à leitura literal após 18/10 |
| 18/10 | células com o mesmo grafo compartilham a latência (mediana); latência GPU em 5 rodadas (refinamento da decisão 2) | sinalizado; confirmação pendente; não altera a vencedora |
| 14/09 | célula poda + int8 em 90% e 95% (não em 98%, onde a poda sozinha já excede 1,5×) | registrada |
| 20/09 | fração 100% do eixo 4 como run próprio | adotada por padrão |
| 30/09 | extra A (eixo 4 na DenseNet-121) | em espera |
