#!/bin/bash
# Full ablation sweep for Section 6 (paper/main.tex):
#   methods: manual, yolo_only, adb, adb_al, adb_no_sam2, adb_no_clip, adb_no_sam2_no_clip
#   seeds:   0, 1, 2  (mean +/- std)
#   epochs:  50 (CPU)
#
# Each run writes data/motorcycle_coco/result_<method>_e50_s<seed>.json
# Resumable: skips a run if its result JSON already exists.
set -uo pipefail
cd "$(dirname "$0")/.."

EPOCHS=50
METHODS="manual yolo_only adb adb_al adb_no_sam2 adb_no_clip adb_no_sam2_no_clip"
SEEDS="0 1 2"

for method in $METHODS; do
  for seed in $SEEDS; do
    out="data/motorcycle_coco/result_${method}_e${EPOCHS}_s${seed}.json"
    if [ -f "$out" ]; then
      echo "[skip] $out already exists"
      continue
    fi
    echo "[run] method=$method seed=$seed epochs=$EPOCHS  $(date)"
    python3 tools/train_eval_motorcycle.py --method "$method" --epochs "$EPOCHS" --seed "$seed"
    echo "[done] method=$method seed=$seed  $(date)"
  done
done

echo "[sweep complete] $(date)"
