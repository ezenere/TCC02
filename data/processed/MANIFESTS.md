# Registro de versões do manifesto

Regra (TODO/README.md): o manifesto é imutável por versão. Toda mudança gera um
arquivo novo, uma linha nova aqui e um sha256 novo; cada run grava em
`run_meta.json` qual versão usou. Nunca sobrescrever. Verificação:
`python scripts/verify_manifest.py --version N`.

| versão | arquivo | sha256 | criado | gerado por | colunas | conteúdo |
|---|---|---|---|---|---|---|
| v1 | `manifest.csv` | `df8e43a346d095a5f8de9e8ea42cf13740e560a3a631350c3b98cc910c95c2d5` | 2026-08-28 | `src/preprocess.py` (seed 20250828) | `image_path, label, user_id, split` | 278.715 imagens; treino 195.102 (14.073 sujeitos) / teste 83.613 (6.044 sujeitos); 18 classes; sobreposição de sujeitos = 0 |
| v2 | `manifest_v2.csv` | `ed5c1eb749ec9e723055d1ce7284df37403df29dc16c0d81337b36a59dba374d` | 2026-09-14 | `src/make_inner_split.py` (seed 20250830 = base+2) | v1 + `inner_split ∈ {fit, val, test}` | Linhas e colunas da v1 idênticas, na mesma ordem. Treino dividido por **sujeito**: fit 175.591 imgs (11.805 sujeitos) / val 19.511 imgs (2.268 sujeitos; 9,54%–10,10% de cada classe). Teste inalterado, marcado `test`. |

## Uso por versão

- **v1** — baseline de evidência `runs/resnet50_baseline_5ep` (Release `ResNet`).
- **v2** — todos os treinos definitivos dos eixos 1–3: treinar em `fit`, selecionar/early-stop em `val`, reportar em `test` uma única vez ao final.
- **v3** (planejada, D1006-2) — v2 + máscaras `frac_{75,50,25,10,5}_s{0..4}`, amostradas por imagem, estratificadas por classe, aninhadas, **dentro de `fit` apenas**.
