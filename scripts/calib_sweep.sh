#!/usr/bin/env bash
# Calibration robustness sweep on ONE run, judged on VAL (never test): 4 activation calibrators x 4
# calibration samples -> TensorRT int8 engine -> val errors. Result: results/eixo2/calib_sweep.csv
#   scripts/calib_sweep.sh runs/eixo4_resnet50_f075_s0
cd /home/ezenere/Documents/Faculdade/TCC; P=/home/ezenere/miniconda3/envs/tcc/bin/python; export PYTHONPATH=src
R="$1"; OUT="$R/trt/calib_sweep"; mkdir -p "$OUT"
for spec in "entropy 0" "minmax 0" "percentile 99.99" "percentile 99.999"; do set -- $spec; m=$1; pc=$2
  for cs in 0 1 2 3; do tag="_sw_${m}${pc/./}_cs${cs}"
    [ -f "$OUT/val${tag}.json" ] && continue
    systemd-run --user --scope -q -p MemoryMax=16G $P src/compress/quantize_onnx_qdq.py --run $R --method $m --percentile ${pc/0/99.999} --calib-seed $cs --tag $tag --smoke-n 32 >/dev/null 2>&1 \
    && $P src/compress/trt_build.py --run $R --precision int8 --qdq $R/model_qdq_int8${tag}.onnx --engine-tag $tag --force >/dev/null 2>&1 \
    && $P src/compress/trt_eval.py --run $R --precision int8 --engine-tag $tag --split val --out "$OUT/val${tag}.json" >/dev/null 2>&1
    rm -f $R/model_qdq_int8${tag}.onnx $R/trt/model_int8${tag}.engine $R/qdq_int8${tag}.json $R/trt/build_int8${tag}.json
  done
done
$P - "$OUT" <<'PY'
import json, sys, glob, re, pandas as pd
rows = []
for f in glob.glob(sys.argv[1] + "/val_sw_*.json"):
    m = re.search(r"_sw_([a-z]+)(\d*)_cs(\d)", f); d = json.load(open(f))
    rows.append({"metodo": m[1] + (" " + m[2][:2] + "." + m[2][2:] if m[2] and m[1] == "percentile" else ""), "cs": int(m[3]), "erros_val": d["n_errors"]})
t = pd.DataFrame(rows).pivot_table(index="metodo", columns="cs", values="erros_val").astype(int)
t["media"] = t.mean(1).round(1); t["max"] = t.max(1); print(t.to_string())
PY
