"""End-to-end smoke of train.py: loss falls, resume replays the same run."""

import csv
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
CFG = ROOT / "configs/test_smoke.yaml"
RUNS = ROOT / "runs/_smoke"
PY = sys.executable

pytestmark = pytest.mark.skipif(
    not (ROOT / "data/processed/manifest_v2.csv").exists(), reason="data not on disk")


def has_cuda() -> bool:
    import torch
    return torch.cuda.is_available()


def train(run_name: str, epochs: int, resume: bool = False, eval_test: bool = False):
    cmd = [PY, "src/train.py", "--config", str(CFG), "--run-name", run_name,
           "--epochs", str(epochs), "--limit-fit", "500", "--limit-val", "200"]
    if resume:
        cmd.append("--resume")
    if eval_test:
        cmd.append("--eval-test")
    env = {"PYTHONPATH": "src", "PATH": "/usr/bin:/bin", "HOME": str(Path.home())}
    return subprocess.run(cmd, cwd=ROOT, env=env, capture_output=True, text=True)


def rows(run_name: str):
    with open(RUNS / run_name / "metrics.csv") as fh:
        return list(csv.DictReader(fh))


@pytest.fixture(scope="module", autouse=True)
def clean():
    shutil.rmtree(RUNS, ignore_errors=True)
    yield
    shutil.rmtree(RUNS, ignore_errors=True)


@pytest.mark.skipif(not has_cuda(), reason="needs a GPU")
def test_loss_falls_and_artifacts_exist():
    r = train("smoke_a", epochs=2)
    assert r.returncode == 0, r.stderr[-2000:]
    a = rows("smoke_a")
    assert len(a) == 2
    assert float(a[1]["train_loss"]) < float(a[0]["train_loss"])
    run = RUNS / "smoke_a"
    for f in ("config.yaml", "run_meta.json", "metrics.csv", "metrics_val_best.json",
              "checkpoints/last.pt", "checkpoints/best.pt"):
        assert (run / f).exists(), f
    import json
    meta = json.loads((run / "run_meta.json").read_text())
    assert meta["manifest_version"] == 2 and meta["git_commit"]


@pytest.mark.skipif(not has_cuda(), reason="needs a GPU")
def test_resume_replays_the_same_run():
    r1 = train("smoke_b", epochs=1)
    assert r1.returncode == 0, r1.stderr[-2000:]
    r2 = train("smoke_b", epochs=2, resume=True)
    assert r2.returncode == 0, r2.stderr[-2000:]
    a, b = rows("smoke_a"), rows("smoke_b")
    assert len(b) == 2
    # Same seed, deterministic cuDNN: the resumed epoch 2 must match run A's.
    for key in ("train_loss", "val_f1_macro", "val_loss"):
        assert abs(float(a[1][key]) - float(b[1][key])) < 5e-3, (key, a[1][key], b[1][key])
