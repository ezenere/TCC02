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
