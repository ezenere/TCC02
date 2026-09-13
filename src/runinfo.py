"""Provenance helpers shared by train.py and eval.py."""

from __future__ import annotations

import platform
import subprocess
from pathlib import Path


def git_info() -> dict:
    def run(*args):
        try:
            return subprocess.check_output(["git", *args], text=True,
                                           stderr=subprocess.DEVNULL).strip()
        except Exception:                                   # noqa: BLE001
            return None
    commit = run("rev-parse", "HEAD")
    dirty = run("status", "--porcelain")
    return {"git_commit": commit,
            "git_dirty": bool(dirty) if dirty is not None else None}


def env_info() -> dict:
    import torch
    import torchvision
    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "hostname": platform.node(),
    }


def manifest_version(path: str | Path) -> int:
    """manifest.csv -> 1, manifest_v2.csv -> 2, ..."""
    stem = Path(path).stem
    return int(stem.split("_v")[1]) if "_v" in stem else 1
