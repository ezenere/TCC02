"""Collapse tqdm carriage-return progress in a train.log, keeping one line per bar.

    python scripts/compact_log.py runs/eixo1_resnet50_s0/train.log   # rewrites in place
"""
import re
import sys
from pathlib import Path

for arg in sys.argv[1:]:
    path = Path(arg)
    raw = path.read_text(errors="replace")
    out, last_bar = [], None
    for chunk in re.split(r"[\r\n]", raw):
        if not chunk.strip():
            continue
        if re.search(r"\d+%\|.*\| \d+/\d+", chunk):        # a tqdm bar frame
            last_bar = chunk                                # keep only the final frame
            continue
        if last_bar is not None:
            out.append(last_bar)
            last_bar = None
        out.append(chunk)
    if last_bar is not None:
        out.append(last_bar)
    before = len(raw)
    path.write_text("\n".join(out) + "\n")
    print(f"{path}: {before / 1e6:.1f} MB -> {path.stat().st_size / 1e6:.2f} MB ({len(out)} linhas)")
