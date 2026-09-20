# STATUS

Diário de execução do projeto. Atualizado ao fim de cada dia de trabalho.

## 2026-09-14 (seg) — Dia 0

**Concluído:** D0914-1 (auditoria), D0914-3 (release já publicada — ver desvios), D0914-4 (manifesto v2).
**Em andamento:** —
**Bloqueado:** — (reunião realizada; decisões abaixo).
**Desvios do plano:** (1) release publicada com tag `ResNet`, não `v0.1`; assets `best.pt` + `last.pt` (180 MB cada). Sugiro manter e seguir `v0.2-eixo1` daqui em diante, como o README prevê. (2) `gh` CLI não está instalado — `scripts/publish_release.sh` usa `curl` + token (decidido 14/09).
**Decisões tomadas hoje (fora do CLAUDE.md):** linhas de teste do manifesto v2 recebem `inner_split = "test"` (em vez de vazio) para o código downstream nunca precisar tratar NaN; seed do split interno = seed base + 2 (20250830).
**GPU hoje:** 0 h (conforme instrução).

### Auditoria do repositório (D0914-1)

| item | estado |
|---|---|
| git | 1 commit (`40f79c3 First Commit`), branch `master`, remote `github.com/ezenere/TCC02`. Untracked: `CLAUDE.md`. Modificado: `.gitignore` (linha `TODO` adicionada pelo humano → **`TODO/` fica fora do git**). |
| tags / releases | tag `ResNet` → Release "ResNet-50 baseline" (2026-08-29), `best.pt` + `last.pt`. Nenhuma tag `v*`. |
| `.gitignore` | exclui `data/`, `runs/**/checkpoints/`, `*.pt`, `TODO`. Logs, configs, métricas e figuras versionados — conforme CLAUDE.md. |
| `requirements.txt` | **fiel ao env `tcc`** (152 pins idênticos ao `pip freeze` normalizado). Falta só o cabeçalho de 4 linhas que `scripts/freeze.sh` gera — cosmético. |
| manifesto | v1 `manifest.csv` sha `df8e43a3…` (igual ao `run_meta.json` do baseline). **v2 criado hoje** `manifest_v2.csv` sha `ed5c1eb7…`. Registro em `data/processed/MANIFESTS.md`. `verify_manifest.py --version 1` e `--version 2`: 0 violações. |
| runs | `runs/resnet50_baseline_5ep/` completo (config, run_meta, metrics.csv, tb, matriz de confusão; checkpoints locais 188 MB × 2). |
| `run_meta.json` | **falta `git_commit` e versão do torchvision** — exigidos pelo README do TODO. Corrigir em D0915-1 (`train.py` grava `git rev-parse HEAD` e `git status --porcelain` vazio/sujo). |
| ausentes | `tests/`, `docs/`, `notebooks/`, `Makefile`, `README.md` raiz, `src/eval.py`, `src/compress/`, `src/measure/`, `results/eixo*/`. Todos previstos em dias posteriores. |
| pacotes ausentes | `pytest` (D0915-4), `fvcore` (D0916-5), `onnx`/`onnxruntime`/`tensorrt` (D0923-3). Instalar no dia da tarefa e rodar `scripts/freeze.sh`. |
| GPU | ociosa (1,1 GB usados pelo desktop). ResNet-50 medido 3,1 min/época; DenseNet-121 ~3,7 min (probe). |

**Correção ao TODO (D0916 critério):** "ResNet-50 ≈ 25,6 M par." vale para a cabeça ImageNet de 1000 classes; com cabeça de 18 classes o total é **23,6 M** (medido no checkpoint). DenseNet-121 com 18 classes ≈ 7,0 M (vs 8,0 M). MACs não mudam de forma relevante.

**CLAUDE.md:** a proposta de `manifest_masks.csv` foi retirada; máscaras do eixo 4 entram como `manifest_v3.csv`, registrado em `MANIFESTS.md` (feito 14/09).

### Manifesto v2 (D0914-4) — critério de conclusão atingido

```
fit    175.591 imgs   11.805 sujeitos
val     19.511 imgs    2.268 sujeitos   (10,00% do treino; 9,54%–10,10% por classe)
test    83.613 imgs    6.044 sujeitos   (idêntico à v1, linha a linha)
```
Determinismo: byte-idêntico em dois processos com `PYTHONHASHSEED` distintos.
Gerador: `src/make_inner_split.py`. Verificador: `scripts/verify_manifest.py --version 2` → OK.

### Pauta — reunião de 14/09 (levar como propostas fechadas)

1. **Saturação do baseline (99,79% acc / 99,79% F1 em 5 épocas).** Proposta já no CLAUDE.md: reportar **taxa de erro e razão de erro vs baseline** como métrica primária de degradação; o eixo 3 é formulado como **localização do joelho de esparsidade** por arquitetura. Pedir apenas: *confirmar*.
   - Evidência: `results/figures/resnet50_baseline_5ep_curvas.pdf`; confusão residual concentrada em `stop`↔`palm` (F1 0,9925 / 0,9936).
2. **Critério de "melhor configuração" para o eixo 4.** Proposta: **menor latência GPU (TensorRT, batch 1) entre as configurações com razão de erro ≤ 1,5× do baseline**; empate → menor tamanho em disco. Alternativa se o professor preferir CPU como alvo: mesma regra com latência CPU 1 thread. Pedir: *escolher GPU ou CPU como latência de referência e confirmar o limiar 1,5×*.
3. **Receita do eixo 4.** Proposta: **épocas fixas (30) + early stopping por validação interna** ("reduz dados, mantém receita"), em vez de steps fixos. Consequência: frações pequenas treinam em minutos; a curva mede eficiência de dados, não orçamento de compute. Pedir: *confirmar*.
4. **Eixo 1 com 3 seeds por arquitetura** (6 runs × 30 épocas ≈ 12–13 h de GPU, já medido). Necessário para o eixo 3 ter barra de erro e para responder "ResNet vs DenseNet difere mais que o ruído entre seeds?". Pedir: *confirmar*.
5. **(nova) Validação interna.** Já implementada hoje conforme CLAUDE.md: 10% dos **sujeitos** do treino (2.268 sujeitos / 19.511 imgs), seleção e early stopping em val, teste só no número final. Informar, não perguntar.
6. **(nova) Célula extra "poda + int8" no benchmark** (D0930-1): barata (~0,5 h GPU) e responde "as técnicas compõem?". Pedir: *autorizar* — entra no CLAUDE.md se sim.
7. **(logística) Feriados 12/10 e 02/11** caem em segunda — reunião remarcada ou pauta por escrito?
8. **(logística) Convenção de tags/releases**: `v0.2-eixo1`, `v0.3-eixo2`, … com uma release por eixo. A release atual (`ResNet`) fica como está.

### Decisões da reunião de 14/09 (todas as propostas aprovadas)

1. Métrica primária de degradação: **taxa de erro e razão de erro vs baseline**; eixo 3 formulado como localização do joelho de esparsidade.
2. "Melhor configuração" para o eixo 4: **menor latência GPU (TensorRT, batch 1) entre as configurações com razão de erro ≤ 1,5×**; empate → menor tamanho em disco.
3. Eixo 4: **30 épocas fixas + early stopping por validação interna** (mesma receita, menos dados).
4. Eixo 1: **3 seeds por arquitetura** (0, 1, 2).
5. Validação interna por sujeitos (manifesto v2) — confirmada.
6. Célula extra **poda + int8** no benchmark — autorizada.
7. Feriados 12/10 e 02/11: pauta por escrito ao professor.
8. Tags/releases: `v0.2-eixo1`, `v0.3-eixo2`, …; release atual (`ResNet`) mantida. Publicação via `scripts/publish_release.sh` (curl + token).

Registradas no `CLAUDE.md` em 14/09.

### Próximo dia (D0915)
`train.py` definitivo (inner_split, best por val, early stopping opcional, warmup 2 épocas, `--seed`, `--resume`, `--frac-column`, `git_commit` no run_meta), `eval.py`, configs definitivas, `tests/`, smoke `[GPU]` de 100 steps nas duas arquiteturas — **liberado**.

## 2026-09-15 (ter) — Dia 1

**Concluído:** D0915-1..5; D0916-3, D0916-4, D0916-5 (antecipados, CPU); `scripts/queue.sh` e `scripts/publish_release.sh` (curl + token).
**Em andamento:** D0916-1/2 + D0917-1 `[GPU]` — fila `scripts/queue.sh eixo1` (6 runs seriais: s0 → s1 → s2, ResNet antes de DenseNet em cada seed). Log: `runs/queue_eixo1.log`; por run: `runs/eixo1_<arch>_s<seed>/train.log`. ETA total ≈ 12–13 h.
**Bloqueado:** —
**Desvios do plano:** D0917-1 antecipado para a mesma fila (GPU serial, sem motivo para esperar). `persistent_workers` desligado no dataloader — necessário para `--resume` replicar exatamente a augmentation (custo ~1 s/época).
**Decisões tomadas hoje (fora do CLAUDE.md):** seeds de treino são 0/1/2 (inteiros pequenos; a seed 20250828 fica só para os dados). `metrics.json` de um run = teste avaliado no `best.pt` (selecionado por F1 macro em val); `metrics_val_best.json` guarda a val. `cost.py` reporta MACs pelo fvcore (1 multiply-add = 1).

### Smoke com a config definitiva (D0915-5)
```
                 loss 100 steps   lr warmup           VRAM pico   throughput   época estimada
ResNet-50        2.907 → 2.811    2.1e-5 → 1.0e-3     4,26 GiB    1.093 img/s  2,7 min
DenseNet-121     2.990 → 2.097    2.1e-5 → 1.0e-3     6,10 GiB      895 img/s  3,3 min
```
Testes: `pytest tests/` → 9 passed (datamodule: sujeitos disjuntos, máscara só no fit, sem rotação; smoke: loss cai, resume replica a época 2 com |Δloss| < 5e-3).

### Custo estático (D0916-5, baseline ResNet-50 com cabeça de 18 classes)
params 23,54 M · MACs 4,11 G @224 · state_dict FP32 90,1 MiB (gzip 83,7 MiB). Critério do TODO corrigido de 25,6 M → 23,6 M.

### Run 1 concluído — `eixo1_resnet50_s0` (86 min)
teste (best.pt = época 28, escolhido por val): **acc 99,788% · F1 macro 99,791% · erro 0,2117% (177 / 83.613)**.
Val: melhor F1 0,99700 na época 28; queda transitória na época 2 (pico do lr após o warmup), recuperada na 3.
163 s/época, 1.077 img/s. `make_eixo1.py`, `plot_curves.py`, `cost.py` e `compact_log.py` validados sobre este run.

**⚠ commitar agora.** O `run_meta.json` deste run registra `git_commit = 40f79c3` com **árvore suja** (`git_dirty: true`): o código que produziu o resultado ainda não está em nenhum commit. Os próximos runs da fila são processos novos e vão registrar o commit que existir no momento em que começarem — commitar antes de a DenseNet s0 terminar (~2 h) faz os runs 2–6 saírem limpos. O run 1 fica documentado como "commit 40f79c3 + alterações não commitadas equivalentes ao commit seguinte".

### Run 2 concluído — `eixo1_densenet121_s0` (107 min)
teste (best.pt = época 30, escolhido por val): **acc 99,809% · F1 macro 99,810% · erro 0,1914% (160 / 83.613)**.
204 s/época, 861 img/s. Custo: 6,97 M parâmetros, 2,86 GMACs @224, 27,2 MiB FP32 (critério ≈7,0 M / ≈2,9 G atingido).
Razão de erro DenseNet/ResNet (seed 0) = 0,90 — ainda uma seed; sinal indeterminado. A DenseNet melhorou até a última época (best = 30), a ResNet estabilizou na 28.
Runs 1 e 2 registram commit `40f79c3` com árvore suja: o código é o de `714b431` (commitado durante o run 2). Runs 3–6 registram `714b431` limpo.

### Antecipado (CPU, durante a fila): eixo 2 — poda (D0921-2..5)
`src/compress/pruning.py` (máscaras explícitas, limiar global por magnitude, `apply_masks` após cada step), integração no
`train.py` (`--init-from`, `--sparsity`; máscaras persistidas no checkpoint; assert de esparsidade ao final), `configs/prune_*.yaml`,
`tests/test_pruning.py` (7 verdes), `docs/METODOLOGIA.md` § 2.1 com a limitação da poda estruturada em DenseNet.
Smoke GPU (1 época, 320 imgs, init do baseline, 50%): esparsidade obtida 0,5000, 54 camadas mascaradas, 11,8 M não-nulos.
A varredura `[GPU]` (D0922) espera a fila do eixo 1 terminar.

### Run 3 concluído — `eixo1_resnet50_s1` (86 min)
teste: **acc 0.99803, F1 0.99805, erro 0.1973% (165 erros), best epoca 28**. Commit registrado: limpo.

### Run 4 concluído — `eixo1_densenet121_s1` (106 min)
teste: **acc 0.99800, F1 0.99802, erro 0.1997% (167 erros), best epoca 26**. Commit registrado: limpo.

### Run 5 concluído — `eixo1_resnet50_s2` (87 min)
teste: **acc 0.99787, F1 0.99789, erro 0.2129% (178 erros), best epoca 21**. Commit registrado: limpo.

## 2026-09-16 → 18 (qua–sex) — Dias 2–4, eixo 1 fechado

**Concluído:** D0916-1/2, D0917-1/2/3, D0918-1/2/3/5. Fila de 6 runs terminou em 01:32 de 14/09 (9,7 h de GPU, 0 falhas).
**Bloqueado:** D0918-4 (manual) — publicar a release `v0.2-eixo1` (instrução abaixo).
**Desvios do plano:** nenhum; o eixo 1 fechou ~4 dias antes do previsto. Poda (D0921-2..5) já antecipada.

### Resultado do eixo 1 (teste, média ± std de 3 seeds)
```
DenseNet-121  acc 99,803 ± 0,005 %   F1 99,805 ± 0,005 %   erro 0,1969 ± 0,0048 %   (160, 167, 167 erros)   203 s/época
ResNet-50     acc 99,793 ± 0,009 %   F1 99,795 ± 0,009 %   erro 0,2073 ± 0,0087 %   (177, 165, 178 erros)   164 s/época
razão de erro DenseNet/ResNet = 0,95  →  SEM SINAL (|Δ| 0,010 p.p. < 2×std 0,017 p.p.)
```
Fonte: `results/eixo1/eixo1_summary.csv`; leitura em `results/eixo1/README.md`.

### Release `v0.2-eixo1` (manual)
```bash
git tag -a v0.2-eixo1 -m "Eixo 1: ResNet-50 e DenseNet-121, 3 seeds, 30 épocas"
export GITHUB_TOKEN=...   # token com escopo repo
scripts/publish_release.sh v0.2-eixo1 "Eixo 1 — treinos definitivos (2 arquiteturas × 3 seeds)" \
  runs/eixo1_resnet50_s0/checkpoints/best.pt    runs/eixo1_resnet50_s0/checkpoints/last.pt \
  runs/eixo1_resnet50_s1/checkpoints/best.pt    runs/eixo1_resnet50_s1/checkpoints/last.pt \
  runs/eixo1_resnet50_s2/checkpoints/best.pt    runs/eixo1_resnet50_s2/checkpoints/last.pt \
  runs/eixo1_densenet121_s0/checkpoints/best.pt runs/eixo1_densenet121_s0/checkpoints/last.pt \
  runs/eixo1_densenet121_s1/checkpoints/best.pt runs/eixo1_densenet121_s1/checkpoints/last.pt \
  runs/eixo1_densenet121_s2/checkpoints/best.pt runs/eixo1_densenet121_s2/checkpoints/last.pt
```
12 assets (6 × 180 MB ResNet, 6 × 54 MB DenseNet ≈ 1,4 GB). Os assets são nomeados `<run>__best.pt` / `<run>__last.pt`.

### Pauta — reunião de 21/09
1. **Eixo 1 fechado:** arquiteturas equivalentes em erro (0,197% vs 0,207%, sem sinal pelo critério de 2× std); DenseNet com 3,4× menos parâmetros e 30% menos MACs, mas 24% mais lenta por época. Mostrar `eixo1_comparativo.pdf`.
2. **Níveis de esparsidade da poda:** 50 / 70 / 90% (CLAUDE.md); regra automática já implementada: se a razão de erro em 90% ficar < 2×, estender a 95 e 98%. Pedir: *confirmar*.
3. **Interpretação do eixo 3:** como não há sinal entre arquiteturas, o benchmark responde "qual técnica de compressão degrada menos, em qual arquitetura" — a pergunta continua bem posta. Informar.
4. **Poda estruturada em DenseNet:** registrada como limitação (concatenações); poda não-estruturada nas duas. Informar.
5. Próximo bloco de GPU: varredura de poda (D0922–D0923), ~5 h.

### Antecipado (CPU): exportação ONNX (D0923-3 parcial, D0924-2)
`src/compress/export_onnx.py` — opset 17, batch dinâmico, checker + paridade ORT-CPU vs PyTorch FP32 em 64 imagens de `val`.
ResNet-50 s0: 89,7 MiB, 122 nós, max|Δ| 5,7e-6, 0 argmax divergentes. DenseNet-121 s0: 27,0 MiB, 372 nós, max|Δ| 2,9e-6, 0 divergentes.
Observação para o eixo 3: em CPU o ORT é 35% mais rápido que o PyTorch eager na ResNet (86 vs 64 img/s, 8 threads) mas 30% mais lento na DenseNet
(42 vs 60 img/s) — o backend interage com a arquitetura; a latência precisa ser declarada por backend, como o CLAUDE.md já exige.
Pendente: TensorRT (instalar quando a GPU estiver livre). `.gitignore`: `*.onnx`, `*.engine`, `*.plan`.

## 2026-09-22 (ter) — Dia 6, poda seed 0 (antecipado para 14/09)

**Concluído:** D0922-1, D0922-2, D0922-3, D0922-4. Seis runs de poda (seed 0) em 1,7 h de GPU, 0 falhas; esparsidade obtida = alvo em todos.
**Em andamento:** D0923-1 `[GPU][NOITE]` — fila: seed 0 a 95/98% + seeds 1–2 em 50/70/90/95/98% (24 runs, ETA ≈ 7 h).
**Decisão automática (D0922-4):** extensão para **95% e 98%** disparada — razão de erro em 90%: DenseNet 1,02×, ResNet 1,07× (< 2×).

### Poda, seed 0 (razão de erro vs baseline da mesma seed; teste)
```
                 50%          70%          90%        não-nulos @90%   gzip @90%
ResNet-50        0,94×        0,99×        1,07×      2,40 M (10%)     17,2 MiB (0,21×)
DenseNet-121     0,99×        0,99×        1,02×      0,77 M (11%)     5,7 MiB (0,22×)
```
Nenhum joelho até 90% em nenhuma arquitetura. A DenseNet, já 3,4× menor, tolera 90% de poda com 0,77 M pesos não-nulos.

### Antecipado (CPU, durante a varredura de poda): PTQ int8 em CPU (D0925-1..3, seed 0)
`src/compress/quantize_cpu.py` — FX graph mode, backend x86/fbgemm, calibração com 1.024 imagens de `fit`, artefato TorchScript
avaliado do disco no teste completo (8 threads). DenseNet quantizou pelo FX sem fallback (D0925-2 não acionado).
```
                 erros int8 / baseline   razão   acc int8    artefato          throughput CPU
ResNet-50        170 / 177               0,96×   99,797%     23,0 MiB (0,26×)  440 img/s
DenseNet-121     210 / 160               1,31×   99,749%     7,7 MiB (0,28×)   472 img/s
```
Queda de acurácia < 1 p.p. nas duas → sem QAT. **Achado:** a int8 degrada a DenseNet (1,31×) e não a ResNet (0,96×) — o primeiro sinal
assimétrico entre arquiteturas do projeto; verificar nas seeds 1–2 (CPU, ~4 min cada) e no backend TensorRT.
Pendente: FP32 em CPU no mesmo processo para a comparação de latência (`--eval-fp32`) — fica para o harness de latência (D1001).

### PTQ int8 CPU — seeds 1 e 2 concluídas (buffer 26–27/09 antecipado)
```
                 razão de erro (3 seeds)   erros int8 / baseline por seed
ResNet-50        1,04 ± 0,07×              170/177, 174/165, 196/178
DenseNet-121     1,24 ± 0,07×              210/160, 203/167, 198/167
```
Δ razão = 0,20 > 2×std (0,14) → **sinal**: a quantização int8 (fbgemm, PTQ por tensor nas ativações) custa mais à DenseNet.
Hipótese a registrar no texto: as concatenações densas juntam ativações de escalas diferentes num só tensor, e a escala
única por tensor do PTQ estático perde resolução nos canais de menor amplitude. Verificável no TensorRT (calibrador de entropia) — D1001+.

## 2026-09-14 (tarde) — incidente e TensorRT

**Incidente:** dois OOM (30 GB e 53 GB de RSS) na calibração por **entropia** do ONNX Runtime, que retém todas as ativações
intermediárias das 1.024 imagens. O segundo foi erro de execução meu: o patch para MinMax falhou na asserção e o quantizador rodou
mesmo assim com o código antigo. Os OOM derrubaram a sessão e a fila de poda (o run `resnet50_p98_s0` retomou do `last.pt`, época 2).
**Correções:** (1) quantizador reescrito com MinMax incremental + correção do bug de flush do ORT; (2) toda execução desse tipo sob
`systemd-run --scope -p MemoryMax=16G` (regra nova no CLAUDE.md); (3) patches encadeados com `&&` e verificados por `grep` antes de rodar.
**Resultado:** `model_qdq_int8.onnx` para ResNet-50 (177 QuantizeLinear) e DenseNet-121 (556) em 28 s cada, memória contida.
**TensorRT 11:** sem `IInt8EntropyCalibrator2` nem flags `FP16`/`INT8` — redes fortemente tipadas; precisão vem do ONNX
(fp32 / fp16 convertido / QDQ). Engine FP32 da ResNet s0 construída e avaliada: erro 0,2105% (176) vs 0,2117% (177) do PyTorch — paridade OK.
`trt_build.py` reescrito; engines FP16/INT8 e as demais serão construídas com a GPU ociosa, após a fila de poda.

### Antecipado (CPU): manifesto v3 e driver do eixo 4 (D1006-2/3/4)
`manifest_v3.csv` (sha `40f09516…`): 25 máscaras `frac_{75,50,25,10,5}_s{0..4}`, estratificadas por classe (±0,1 p.p.), aninhadas,
só em `fit`; val/test idênticos à v2. `scripts/verify_manifest.py --version 3` → 0 violações; byte-idêntico entre processos.
`src/reduce_data.py`: fila retomável (pula runs com `metrics.json`), seeds 0–2 antes de 3–4, frações da maior para a menor;
encadeia denso → poda → fine-tuning quando `--sparsity` é dado. `train.py` ganhou `--manifest` e `--patience`.
**Para a pauta:** a fração **100%** entra no eixo 4 como run próprio (30 épocas + early stopping, paciência 5), para que todos os pontos
da curva tenham a mesma receita — custa ~1,75 h × seed a mais do que reutilizar os runs do eixo 1 (que não tiveram early stopping).
Alternativa: reutilizar o eixo 1 como ponto 100% e registrar a diferença de receita. Aguardo decisão.

## 2026-09-14 (noite) — poda fechada com 3 seeds (D0923-1, D0924-1)

30 runs (2 arch × 5 níveis × 3 seeds), 0 falhas, ~8,5 h de GPU. Razão de erro vs baseline da mesma seed (média ± std), teste:
```
esparsidade   ResNet-50        DenseNet-121     não-nulos R / D     gzip R / D (× baseline)
50%           0,97 ± 0,03      0,97 ± 0,06      11,8 M / 3,5 M      0,61 / 0,61
70%           0,98 ± 0,02      0,98 ± 0,05      7,1 M / 2,2 M       0,42 / 0,43
90%           1,10 ± 0,03      1,03 ± 0,04      2,4 M / 0,77 M      0,21 / 0,22
95%           1,28 ± 0,07      1,19 ± 0,05      1,2 M / 0,43 M      0,14 / 0,17
98%           1,90 ± 0,17      1,65 ± 0,05      0,52 M / 0,22 M     0,10 / 0,13
```
**Joelho (primeiro nível com razão > 1,5×): 98% nas duas.** Até 90% a degradação está dentro do ruído entre seeds; 95% já é
visível mas dentro do orçamento de 1,5× do critério do eixo 3. A DenseNet tolera cada nível um pouco melhor que a ResNet
(1,03 vs 1,10 em 90%; 1,65 vs 1,90 em 98%) apesar de partir de 3,4× menos parâmetros.
Célula poda+int8 (D0930-1): rodada em 90% e 95% (o último nível dentro do orçamento), não em 98%, onde a poda sozinha já
excede o critério — registrar a escolha na reunião.

## 2026-09-14 (noite) — TensorRT fechado na seed 0 (D0928-2/3/4, D0929-1, D0930-1 parcial)

**Rota int8 do TensorRT 11 (o que funcionou):** rede fortemente tipada; ONNX Q/DQ do ORT com pesos int8 + só `DequantizeLinear`
(`AddQDQPairToWeight=False`), bias não quantizado (TRT rejeita DQ sobre Int32), conv de entrada mantida em float (3 canais; sem kernel
int8 para o bloco fundido Q+conv+relu+maxpool). Calibração escolhida **em `val`**, sem tocar no teste:
```
calibrador (ativações)   ResNet val erros   DenseNet val erros    (FP32 em val: ~59 / ~54)
MinMax                   131                67
Entropia (KL) incremental 61                63     ← escolhido; memória contida por alimentar histogramas lote a lote
```
**Resultados no teste, seed 0** (razão vs PyTorch da mesma seed; img/s em lote 32, avaliação com dataloader):
```
                     ResNet-50                        DenseNet-121
TRT FP32 (TF32 off)  176 erros  0,99×   1.945 img/s   160 erros  1,00×   1.665 img/s
TRT FP16             176 erros  0,99×   4.488 img/s   160 erros  1,00×   3.183 img/s
TRT INT8 (entropia)  180 erros  1,02×   6.227 img/s   190 erros  1,19×   1.763 img/s
int8 CPU fbgemm      170 erros  0,96×                 210 erros  1,31×
```
**Achados:** (1) a int8 degrada a DenseNet nos dois backends (1,19× TRT, 1,31× CPU) e não a ResNet (1,02× / 0,96×) — o efeito é da
arquitetura, não do backend; (2) a engine int8 da DenseNet é **mais lenta que a FP16** (1.763 vs 3.183 img/s): 376 camadas ficam em float
e as conversões Q/DQ em torno das concatenações custam mais do que os kernels int8 economizam; na ResNet a int8 é 1,4× a FP16;
(3) a acurácia difere entre backends (ResNet 180 vs 170 erros; DenseNet 190 vs 210), como o CLAUDE.md previa — reportar por backend.
**Em andamento:** fila TensorRT das seeds 1–2 (FP32/FP16/INT8) e poda@90/95 + INT8 TRT (seed 0) — `runs/queue_trt.log`.

## 2026-09-14 (madrugada de 15/09) — TensorRT com 3 seeds e poda+int8 (D0929-2, D0930-1)

Fila `scripts/queue_trt.sh`: 8 lotes, 0 falhas. Teste, razão de erro vs PyTorch da mesma seed (média ± std, 3 seeds):
```
                 TRT FP32          TRT FP16          TRT INT8          int8 CPU        img/s TRT fp16 / int8 (lote 32, c/ dataloader)
ResNet-50        1,00 ± 0,01       1,00 ± 0,01       1,08 ± 0,08       1,04 ± 0,07     5.068 / 6.189
DenseNet-121     1,00 ± 0,00       1,00 ± 0,00       1,19 ± 0,08       1,24 ± 0,07     3.198 / 1.753
```
Poda + int8 (seed 0; razão vs baseline):
```
                 poda só    +int8 CPU   +int8 TRT
ResNet-50 90%    1,07       1,14        1,42
ResNet-50 95%    1,30       1,34        1,48
DenseNet 90%     1,02       1,37        1,51
DenseNet 95%     1,21       1,58        1,92
```
**Achados:** a int8 do TensorRT sobre modelos podados degrada mais que a de CPU (ResNet 90%: 1,42 vs 1,14) — a calibração por entropia
sobre pesos 90% esparsos produz escalas piores que o observador por histograma do FX; hipótese a registrar. As duas técnicas compõem de
forma aproximadamente aditiva na ResNet; na DenseNet a int8 domina e 95%+int8 sai do orçamento de 1,5× nos dois backends.
**Em andamento (background):** latência CPU de todas as células (`scripts/measure_cpu_latency.sh`, máquina ociosa) e, na sequência,
GPU **preliminar** com a sessão gráfica aberta (`NOTE` registra isso; a medição válida é a tua em TTY, que sobrescreve os JSONs).

## 2026-09-19 (sáb) — pedidos do orientador: auditoria, poda antes do treino, app de webcam

**Concluído:** auditoria de vazamento completa (`results/audit/README.md`); app `app/webcam_demo.py` (4 backends, 18/18 no teste headless);
código e configs do eixo 2b; `.gitignore` corrigido (as linhas `TODO` e `*.onnx` estavam coladas — `TODO/` nunca chegou a ser commitado).
**Em andamento:** `[GPU]` fila `scripts/queue_prune_first.sh "0" "0.9 0.95 0.98 0.7 0.5"` — 10 runs de 30 épocas, ETA ≈ 16 h. Log: `runs/queue_eixo2b_prune_first.log`.
**Pendente (máquina ociosa):** latência CPU — a medição de 15/09 foi interrompida na metade (ResNet completa; DenseNet só 1 thread); GPU em TTY (manual).
**Bloqueado (decisão):** fração 100% do eixo 4; release `v0.2-eixo1`; seeds 1–2 do eixo 2b (+32 h de GPU) dependem do resultado da seed 0.

### Auditoria — resumo
Sujeitos disjuntos, manifesto = JSONs brutos em 100% das linhas, 0 duplicatas exatas, controle de rótulos embaralhados = 4,9% (acaso).
**Achado:** o `user_id` do HaGRID não é identidade perfeita — a mesma pessoa aparece sob contas diferentes; 87 imagens de teste (0,10%) são
quase idênticas a imagens de treino. Removendo todos os usuários suspeitos: erro inalterado no critério estrito (0,2125% vs 0,2117%),
0,28% no critério frouxo (pior caso 99,72%). Holdout externo (278.702 imagens, 19.042 sujeitos nunca usados): erro 0,235% ResNet / 0,216% DenseNet.
