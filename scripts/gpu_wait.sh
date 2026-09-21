#!/usr/bin/env bash
# Block until the GPU is free for a training run: no HEAVY compute process from anyone
# and at least NEED_MB of free memory, on STABLE consecutive checks 20 s apart.
# Desktop processes (kwin_wayland, Xwayland...) register as small compute apps and are
# ignored: only apps holding more than HEAVY_MB count as "someone is using the GPU".
#   scripts/gpu_wait.sh [NEED_MB=8000] [STABLE=3] [HEAVY_MB=500]
NEED_MB="${1:-8000}"; STABLE="${2:-3}"; HEAVY_MB="${3:-500}"; ok=0; waited=0
while [ "$ok" -lt "$STABLE" ]; do
  free="$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)"
  heavy="$(nvidia-smi --query-compute-apps=used_memory --format=csv,noheader,nounits | awk -v t="$HEAVY_MB" '$1+0 > t' | grep -c .)"
  if [ "${free:-0}" -ge "$NEED_MB" ] && [ "$heavy" -eq 0 ]; then ok=$((ok + 1)); else ok=0; fi
  [ "$ok" -lt "$STABLE" ] && { sleep 20; waited=$((waited + 20)); }
done
[ "$waited" -gt 60 ] && echo "gpu_wait: esperou ${waited}s pela GPU" || true
