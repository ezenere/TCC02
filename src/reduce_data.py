"""Axis 4 driver: train the winning configuration on nested fractions of `fit`.

    python src/reduce_data.py --config configs/train_resnet50.yaml --dry-run
    python src/reduce_data.py --config configs/train_resnet50.yaml --seeds 0 1 2
    python src/reduce_data.py --config configs/train_densenet121.yaml --seeds 3 4 --sparsity 0.9

Per (fraction F, seed K): `train.py --manifest manifest_v3.csv --frac-column
frac_F_sK --seed K --patience P --eval-test` -> runs/eixo4_<arch>_f<F>_s<K>/.
Fraction 100 is trained the same way without a mask, so every point of the
curve shares the recipe (30 epochs + early stopping on val). The seed K
governs mask, head init, data order and augmentation.

If the axis-3 winner is a pruned model (--sparsity S): dense run on the
fraction -> prune at S -> fine-tune on the same fraction
(runs/eixo4_<arch>_f<F>_s<K>_p<S>/, config configs/prune_<arch>.yaml).

Resumable: a run with metrics.json is skipped; an interrupted one resumes from
last.pt. Order: seeds 0-2 first over all fractions (large to small), then 3-4.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
FRACTIONS = (100, 75, 50, 25, 10, 5)
PY = sys.executable


def run_name(arch: str, frac: int, seed: int, sparsity: float | None = None) -> str:
    base = f"eixo4_{arch}_f{frac:03d}_s{seed}"
    return base if sparsity is None else f"{base}_p{int(round(100 * sparsity)):02d}"


def train_cmd(config: Path, arch: str, frac: int, seed: int, manifest: str, patience: int,
              epochs: int | None, init_from: Path | None = None, sparsity: float | None = None,
              prune_config: Path | None = None) -> tuple[str, list[str]]:
    name = run_name(arch, frac, seed, sparsity)
    cmd = [PY, "src/train.py", "--config", str(prune_config if sparsity is not None else config),
           "--seed", str(seed), "--run-name", name, "--manifest", manifest,
           "--patience", str(patience), "--resume", "--eval-test"]
    if frac != 100:
        cmd += ["--frac-column", f"frac_{frac}_s{seed}"]
    if epochs is not None and sparsity is None:
        cmd += ["--epochs", str(epochs)]
    if sparsity is not None:
        cmd += ["--sparsity", str(sparsity), "--init-from", str(init_from)]
    return name, cmd


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True, help="dense training config of the winner's arch")
    ap.add_argument("--manifest", default="data/processed/manifest_v3.csv")
    ap.add_argument("--fractions", type=int, nargs="+", default=list(FRACTIONS))
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=None, help="override (default: config, 30)")
    ap.add_argument("--sparsity", type=float, default=None, help="winner is pruned at this level")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    cfg = yaml.safe_load(args.config.read_text())
    arch = cfg["model"]["arch"]
    prune_config = ROOT / f"configs/prune_{arch}.yaml"
    log = ROOT / "runs" / f"queue_eixo4_{arch}.log"
    env = {**__import__("os").environ, "PYTHONPATH": "src", "TQDM_MININTERVAL": "60"}

    plan = []
    for seed in args.seeds:
        for frac in sorted(args.fractions, reverse=True):
            name, cmd = train_cmd(args.config, arch, frac, seed, args.manifest, args.patience, args.epochs)
            plan.append((name, cmd))
            if args.sparsity is not None:
                init = ROOT / "runs" / name / "checkpoints" / "best.pt"
                pname, pcmd = train_cmd(args.config, arch, frac, seed, args.manifest, args.patience, None,
                                        init_from=init, sparsity=args.sparsity, prune_config=prune_config)
                plan.append((pname, pcmd))

    if args.dry_run:
        for i, (name, cmd) in enumerate(plan, 1):
            status = "feito" if (ROOT / "runs" / name / "metrics.json").exists() else "pendente"
            print(f"{i:>2}. {name:<40} {status:<8} {' '.join(cmd[2:])}")
        print(f"\n{len(plan)} runs")
        return 0

    def say(msg: str) -> None:
        line = f"{time.strftime('%F %T')}  {msg}"
        print(line, flush=True)
        with open(log, "a") as fh:
            fh.write(line + "\n")

    say(f"=== eixo4 {arch} frações {args.fractions} seeds {args.seeds} sparsity {args.sparsity} ===")
    for name, cmd in plan:
        run_dir = ROOT / "runs" / name
        if (run_dir / "metrics.json").exists():
            say(f"skip  {name}")
            continue
        run_dir.mkdir(parents=True, exist_ok=True)
        say(f"start {name}")
        t0 = time.time()
        with open(run_dir / "train.log", "a") as fh:
            rc = subprocess.run(cmd, cwd=ROOT, env=env, stdout=fh, stderr=subprocess.STDOUT).returncode
        say(f"end   {name} rc={rc}  {(time.time() - t0) / 60:.0f} min")
        if rc != 0:
            say(f"FALHOU {name} — abortando a fila (retomar com o mesmo comando)")
            return rc
        subprocess.run([PY, "src/measure/cost.py", "--checkpoint", str(run_dir / "checkpoints/best.pt")],
                       cwd=ROOT, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    say(f"=== eixo4 {arch} concluído ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
