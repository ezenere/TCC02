# Runbook — como rodar, quanto demora, como retomar

Todos os comandos a partir da raiz, com o env `tcc` ativo e `PYTHONPATH=src` (o `Makefile` já exporta). `make help` lista os alvos.

## Dados

1. Baixar `hagridv2_512.zip` (~119 GB) e `annotations.zip` do repositório `hukenovs/hagrid`; extrair em `data/raw/HaGRIDv2_dataset_512/` e `data/raw/annotations/`.
2. `python src/validate_raw.py` — confere JSONs × imagens × bboxes (≈15 s).
3. `python src/preprocess.py` — amostra 50% dos sujeitos, split 70/30 por sujeito, recorta 278.715 mãos (≈70 s, 24 processos) → `data/processed/` (4,2 GB) e `manifest.csv`.
4. `python src/make_inner_split.py` (v2: validação interna) e `python src/make_masks.py` (v3: máscaras do eixo 4).
5. `make verify` — os sha256 devem bater com `data/processed/MANIFESTS.md`.

## Tempos de referência (RTX 3080 Ti, batch 96, AMP)

| etapa | tempo | VRAM |
|---|---|---|
| treino ResNet-50, 30 épocas + val | ≈ 86 min (164 s/época) | 4,3 GiB |
| treino DenseNet-121, 30 épocas + val | ≈ 107 min (203 s/época) | 6,2 GiB |
| poda + fine-tuning de 5 épocas | 15 min (ResNet) / 19 min (DenseNet) | idem |
| int8 CPU (calibração + teste completo, 8 threads) | ≈ 4 min | — |
| ONNX → Q/DQ (entropia, 1.024 imagens) | 2–3 min, **sob teto de 16 GB** | — |
| engines TensorRT fp32/fp16/int8 | ResNet ≈ 40 s; DenseNet ≈ 7 min | 4 GiB de workspace |
| avaliação no teste (83.613 imagens) | ≈ 1 min em GPU | — |

## Filas e retomada

Todas as filas são seriais (uma GPU) e **retomáveis**: um run com `metrics.json` é pulado; um run interrompido continua de
`checkpoints/last.pt` com a mesma ordem de lotes e a mesma augmentation. Basta repetir o comando.

| fila | comando | log |
|---|---|---|
| eixo 1 | `scripts/queue.sh eixo1 ["0 1 2"]` | `runs/queue_eixo1.log` |
| poda depois | `scripts/queue_prune.sh "<seeds>" "<níveis>"` | `runs/queue_eixo2_prune.log` |
| poda antes | `scripts/queue_prune_first.sh "<seeds>" "<níveis>"` | `runs/queue_eixo2b_prune_first.log` |
| TensorRT | `scripts/queue_trt.sh "<runs>" "fp32 fp16 int8"` | `runs/queue_trt.log` |
| eixo 4 | `python src/reduce_data.py --config configs/train_<arch>.yaml --seeds 0 1 2 [--sparsity S]` (`--dry-run` lista) | `runs/queue_eixo4_<arch>.log` |

Cada run tem o seu `train.log`; `python scripts/compact_log.py <log>` remove os quadros de barra de progresso.

## Cuidados

- **Memória.** Calibração/quantização ONNX roda sob `systemd-run --user --scope -p MemoryMax=16G`: os calibradores por histograma do ONNX Runtime, sem o modo incremental deste repositório, passam de 50 GB.
- **Engines TensorRT** são construídas com a GPU ociosa (a seleção de kernels é por cronometragem) e são específicas da GPU e da versão do TensorRT.
- **Latência.** CPU: máquina em repouso (`make latency-cpu`). GPU: em um TTY com a sessão gráfica fechada: `NOTE="sessão gráfica fechada (TTY)" make latency-gpu`. O campo `note` fica gravado em cada JSON.
- **Um run só vale** com `run_meta.json` registrando commit git limpo, versão e sha do manifesto. Commitar antes de lançar filas.
- **Teste intocável.** Nenhuma escolha (época, calibrador, hiperparâmetro) usa o split de teste; tudo isso usa `val`.

## Releases de pesos

```bash
git tag -a v0.2-eixo1 -m "Eixo 1"
GITHUB_TOKEN=... scripts/publish_release.sh v0.2-eixo1 "Eixo 1 — treinos definitivos" runs/eixo1_*/checkpoints/{best,last}.pt
```
