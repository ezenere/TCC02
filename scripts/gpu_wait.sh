#!/usr/bin/env bash
# Block until the GPU is free for a training run: no compute process from anyone and
# at least NEED_MB of free memory, on STABLE consecutive checks 20 s apart.
#   scripts/gpu_wait.sh [NEED_MB=8000] [STABLE=3]
NEED_MB="${1:-8000}"; STABLE="${2:-3}"; ok=0; waited=0
while [ "$ok" -lt "$STABLE" ]; do
  free="$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)"
  apps="$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -c .)"
  if [ "${free:-0}" -ge "$NEED_MB" ] && [ "$apps" -eq 0 ]; then ok=$((ok + 1)); else ok=0; fi
  [ "$ok" -lt "$STABLE" ] && { sleep 20; waited=$((waited + 20)); }
done
[ "$waited" -gt 60 ] && echo "gpu_wait: esperou ${waited}s pela GPU" || true
