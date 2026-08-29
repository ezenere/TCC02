#!/usr/bin/env bash
set -euo pipefail
PY="${PY:-/home/ezenere/miniconda3/envs/tcc/bin/python}"

{
  echo "# TCC com env congelado. Reproduzir com:"
  PYVER="$("$PY" -c "import sys; print('%d.%d' % sys.version_info[:2])")"
  echo "#   conda create -n tcc python=$PYVER -y"
  echo "#   pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu128"
  echo "# GPU alvo: RTX 3080 Ti (sm_86)."
  "$PY" -m pip freeze | "$PY" -c '
import re, sys
from importlib.metadata import version, PackageNotFoundError
for line in sys.stdin:
    line = line.rstrip("\n")
    m = re.match(r"^([A-Za-z0-9._-]+) @ (file://|git\+).*", line)
    if m:
        try:
            print(f"{m.group(1)}=={version(m.group(1))}")
            continue
        except PackageNotFoundError:
            pass
    print(line)
'
} > requirements.txt
echo "requirements.txt: $(grep -c '==' requirements.txt) pacotes pinados"
