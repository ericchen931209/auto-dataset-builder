"""
Cross-domain validation of Neural DQS (Section 5/6 limitation: "generalization
to datasets with substantially different class distributions ... has not been
evaluated").

For each of the 7 motorcycle-dataset training-label variants produced for the
ADB pipeline ablation (Section 6.4), compute the 6 DQS features and the
Neural DQS predicted score using the model trained on COCO128 degradation
variants (data/dqs_training_data_v5.csv), and correlate the predicted score
against the actual mean mAP@0.5 (over 3 seeds, 50 epochs) measured for that
variant.

This is an out-of-domain test: the model was never trained on any data from
this single-class motorcycle dataset.
"""
import csv
import glob
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))

from models.dqs.feature_extractor import extract_features
from models.dqs.neural_dqs import predict, FEATURE_NAMES

ROOT = "data/motorcycle_coco/yolo_dataset"
METHODS = ["manual", "yolo_only", "adb", "adb_al", "adb_no_sam2", "adb_no_clip", "adb_no_sam2_no_clip"]


def mean_map50(method: str) -> float:
    vals = []
    for f in glob.glob(f"data/motorcycle_coco/result_{method}_e50_s*.json"):
        vals.append(json.load(open(f))["map50"])
    return float(np.mean(vals)) if vals else float("nan")


def main():
    rows = []
    for method in METHODS:
        img_dir = os.path.join(ROOT, method, "images", "train")
        lbl_dir = os.path.join(ROOT, method, "labels", "train")
        feats = extract_features(img_dir, lbl_dir)
        feat_vec = [getattr(feats, name) for name in FEATURE_NAMES]
        dqs_score = predict(feat_vec)
        actual = mean_map50(method)
        rows.append({
            "method": method,
            "dqs_score": dqs_score,
            "actual_map50": actual,
            **{name: getattr(feats, name) for name in FEATURE_NAMES},
        })
        print(f"{method:22s} DQS={dqs_score:.4f}  actual_mAP50={actual:.4f}")

    dqs_vals = np.array([r["dqs_score"] for r in rows])
    map_vals = np.array([r["actual_map50"] for r in rows])
    pearson_r = float(np.corrcoef(dqs_vals, map_vals)[0, 1])
    print(f"\nPearson r (DQS vs actual mAP50, n={len(rows)}): {pearson_r:.4f}")

    out = {"pearson_r": pearson_r, "n": len(rows), "rows": rows}
    out_path = "data/motorcycle_coco/dqs_cross_domain_check.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
