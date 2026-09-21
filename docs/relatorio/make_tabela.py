"""Appendix of the report: every measured accuracy in one place, from the result CSVs.

    python docs/relatorio/make_tabela.py      # rewrites the marked block of RELATORIO.md

Rows = model versions, columns = mean accuracy on the final test, each repetition (seed)
and the mean number of wrong photos. Nothing is typed by hand; "—" = not run / still running.
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
MD = Path(__file__).resolve().parent / "RELATORIO.md"
NAME = {"resnet50": "ResNet-50", "densenet121": "DenseNet-121"}
CELLS = [("baseline", "Modelo original"),
         ("prune-50", "Apagar 50% + ajuste curto"), ("prune-70", "Apagar 70% + ajuste curto"),
         ("prune-90", "Apagar 90% + ajuste curto"), ("prune-95", "Apagar 95% + ajuste curto"),
         ("prune-98", "Apagar 98% + ajuste curto"),
         ("int8-cpu", "Compacto, no processador"), ("trt-fp32", "Placa de vídeo, precisão total"),
         ("trt-fp16", "Placa de vídeo, meia precisão"), ("trt-int8", "Compacto, na placa de vídeo"),
         ("prune-90+int8-cpu", "Apagar 90% + compacto (processador)"), ("prune-95+int8-cpu", "Apagar 95% + compacto (processador)"),
         ("prune-90+trt-int8", "Apagar 90% + compacto (placa de vídeo)"), ("prune-95+trt-int8", "Apagar 95% + compacto (placa de vídeo)")]
def head(title: str) -> list[str]:
    """The network name lives in the header cell, so it can never be separated from its table."""
    return [f"| {title} | Acerto médio | Rep. 1 | Rep. 2 | Rep. 3 | Erros (média) |", "|---|---|---|---|---|---|"]


def pct(v) -> str:
    return "—" if pd.isna(v) else f"{100 * v:.3f}%".replace(".", ",")


def row(label: str, g: pd.DataFrame) -> str:
    by = g.set_index("seed")
    seeds = [pct(by.acc.get(k, float("nan"))) for k in (0, 1, 2)]
    return f"| {label} | **{pct(g.acc.mean())}** | {' | '.join(seeds)} | {g.n_errors.mean():.0f} |"


def main() -> int:
    cells = pd.read_csv(ROOT / "results/eixo3/eixo3_table.csv")
    prune = pd.read_csv(ROOT / "results/eixo2/prune_compare_runs.csv")
    e4 = pd.read_csv(ROOT / "results/eixo4/eixo4_runs.csv")
    out = ['Prova final: 83.613 fotos. "Ajuste curto" = 5 rodadas de treino depois de apagar; "compacto" = números '
           'guardados em 8 bits. "—" = não executado ou ainda rodando.', ""]
    for arch in ("resnet50", "densenet121"):
        out += head(NAME[arch])
        def add_cells(selected):
            for cell, label in selected:
                g = cells[(cells.arch == arch) & (cells.cell == cell)]
                if len(g):
                    out.append(row(label, g))
        add_cells(CELLS[:6])                                     # original + the five pruning levels
        for key, label in (("antes", "Apagar {p}% antes de treinar (30 rodadas)"), ("contro", "Apagar {p}% e treinar tudo de novo (30 rodadas)")):
            sub = prune[(prune.arch == arch) & prune.method.str.startswith(key)]
            for p in sorted(sub.sparsity.unique()):
                out.append(row(label.format(p=int(p)), sub[sub.sparsity == p]))
        add_cells(CELLS[6:])                                     # compact versions and combinations
        out.append("")
    out += head("ResNet-50 com menos fotos (etapa 4)")
    for stage, tag in (("dense", "modelo original"), ("dense+trt-int8", "modelo compacto")):
        for frac in sorted(e4.frac.unique(), reverse=True):
            g = e4[(e4.stage == stage) & (e4.frac == frac)]
            if len(g):
                n = f"{int(g.n_fit.iloc[0]):,}".replace(",", ".")
                out.append(row(f"{frac}% das fotos ({n} imagens), {tag}", g))
    block = '<div class="long" markdown="1">\n\n' + "\n".join(out) + "\n\n</div>"

    txt = MD.read_text(encoding="utf-8")
    a, b = "<!-- tabela:start -->", "<!-- tabela:end -->"
    pre, rest = txt.split(a, 1)
    txt = pre + a + "\n" + block + "\n" + b + rest.split(b, 1)[1]
    # keep the "in progress" line honest
    done = sum(len(re.findall(r"end   eixo2[bc]_\w+_s[12] rc=0", p.read_text())) for p in
               (ROOT / "runs").glob("queue_eixo2[bc]*.log"))
    txt = re.sub(r"Rodando: \d+ de 16 treinos concluídos, cerca de \d+ horas restantes",
                 f"Rodando: {done} de 16 treinos concluídos, cerca de {round((16 - done) * 100 / 60)} horas restantes", txt)
    MD.write_text(txt, encoding="utf-8")
    print(f"tabela: {sum(1 for l in out if l.startswith('| ') and not l.startswith('| Versão'))} linhas | fila extra: {done}/16")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
