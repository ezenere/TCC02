"""Markdown -> PDF (A4) with tables and images, via python-markdown + headless Chromium.

    python scripts/md2pdf.py docs/relatorio/RELATORIO.md [saida.pdf]
"""

from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

import markdown

CSS = """
@page { size: A4; margin: 18mm 17mm 18mm 17mm; }
* { box-sizing: border-box; }
body { font-family: 'Noto Sans', 'DejaVu Sans', 'Liberation Sans', Arial, sans-serif; font-size: 10.4pt; line-height: 1.5;
       color: #1a1a19; margin: 0; }
h1 { font-size: 21pt; line-height: 1.2; margin: 0 0 4pt; }
h1 + p { color: #52514e; margin-top: 0; }
h2 { font-size: 14pt; margin: 20pt 0 6pt; padding-bottom: 3pt; border-bottom: 1.5px solid #2a78d6; break-after: avoid; }
h3 { font-size: 11.5pt; margin: 14pt 0 4pt; break-after: avoid; }
p, li { orphans: 3; widows: 3; }
ul { padding-left: 16pt; margin: 6pt 0; } li { margin: 3pt 0; }
table { border-collapse: collapse; width: 100%; margin: 8pt 0 12pt; font-size: 9.6pt; break-inside: avoid; }
th { background: #eef3fa; text-align: left; font-weight: 600; }
th, td { border: 1px solid #d9d8d3; padding: 4.5pt 7pt; vertical-align: top; }
tr:nth-child(even) td { background: #fafaf8; }
img { display: block; max-width: 100%; max-height: 68mm; margin: 8pt auto 10pt; break-inside: avoid; }
img[src$='gestos.jpg'] { max-height: 66mm; }
strong { font-weight: 650; }
code { font-family: 'DejaVu Sans Mono', monospace; font-size: 9pt; background: #f2f1ee; padding: 0 3pt; border-radius: 3px; }
"""


def main() -> int:
    src = Path(sys.argv[1]).resolve()
    out = Path(sys.argv[2]).resolve() if len(sys.argv) > 2 else src.with_suffix(".pdf")
    body = markdown.markdown(src.read_text(encoding="utf-8"), extensions=["tables", "fenced_code", "sane_lists"])
    html = (f"<!doctype html><html lang='pt-BR'><head><meta charset='utf-8'><base href='{src.parent.as_uri()}/'>"
            f"<title>{src.stem}</title><style>{CSS}</style></head><body>{body}</body></html>")
    with tempfile.NamedTemporaryFile("w", suffix=".html", delete=False, dir=src.parent, encoding="utf-8") as fh:
        fh.write(html)
        tmp = Path(fh.name)
    try:
        subprocess.run(["chromium", "--headless=new", "--no-sandbox", "--disable-gpu", "--no-pdf-header-footer",
                        "--run-all-compositor-stages-before-draw", "--virtual-time-budget=10000",
                        f"--print-to-pdf={out}", tmp.as_uri()], check=True, capture_output=True, timeout=180)
    finally:
        tmp.unlink(missing_ok=True)
    print(f"{out}  ({out.stat().st_size / 1024:.0f} KiB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
