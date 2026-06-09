"""
DQS Feature Importance via SHAP (Figure 5 generator)

Loads trained Neural DQS model + training data, computes SHAP values,
and saves a bar chart for the paper.

Usage:
    python tools/explain_dqs.py --data data/dqs_training_data_v5.csv
    python tools/explain_dqs.py --data data/dqs_training_data_v5.csv --out dqs_shap.png
"""
import argparse
import csv
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

FEATURE_COLS = [
    "annotation_quality", "sharpness", "clip_diversity",
    "lighting_diversity", "pose_diversity", "class_balance",
]
FEATURE_LABELS = ["AQ\n(Annotation)", "IQ\n(Image Quality)", "CD\n(CLIP Diversity)",
                  "LD\n(Lighting)", "PD\n(Pose)", "CB\n(Class Balance)"]


def load_X(csv_path: str) -> np.ndarray:
    rows = []
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            try:
                x = [float(row[c]) for c in FEATURE_COLS]
                if float(row["map50"]) > 0:
                    rows.append(x)
            except (KeyError, ValueError):
                continue
    return np.array(rows, dtype=np.float32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data",  required=True, help="Path to dqs_training_data.csv")
    parser.add_argument("--model", default="models/dqs/neural_dqs_model.pkl")
    parser.add_argument("--out",   default="dqs_shap.png")
    parser.add_argument("--nsamples", type=int, default=200,
                        help="SHAP KernelExplainer nsamples (higher = slower but more accurate)")
    args = parser.parse_args()

    print(f"Loading data from {args.data}…")
    X = load_X(args.data)
    print(f"  {len(X)} samples loaded")

    print(f"Computing SHAP importance (nsamples={args.nsamples})…")
    from models.dqs.neural_dqs import compute_shap_importance
    importance = compute_shap_importance(X, model_path=args.model, nsamples=args.nsamples)

    print("\nSHAP mean |φ| per feature:")
    sorted_items = sorted(importance.items(), key=lambda x: x[1], reverse=True)
    for name, val in sorted_items:
        bar = "█" * int(val * 200)
        print(f"  {name:<25} {val:.5f}  {bar}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        names  = [k for k, _ in sorted_items]
        values = [v for _, v in sorted_items]
        labels = [FEATURE_LABELS[FEATURE_COLS.index(n)] for n in names]

        colors = ["#2196F3" if n in ("clip_diversity", "sharpness") else "#90CAF9" for n in names]

        fig, ax = plt.subplots(figsize=(7, 4))
        bars = ax.barh(labels[::-1], values[::-1], color=colors[::-1], edgecolor="white", height=0.6)
        ax.set_xlabel("Mean |SHAP value|  (impact on predicted mAP)", fontsize=11)
        ax.set_title("Neural DQS — Feature Importance (SHAP)", fontsize=13, fontweight="bold")

        for bar, val in zip(bars, values[::-1]):
            ax.text(bar.get_width() + 0.0005, bar.get_y() + bar.get_height() / 2,
                    f"{val:.4f}", va="center", ha="left", fontsize=9)

        ax.set_xlim(0, max(values) * 1.25)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        plt.tight_layout()
        plt.savefig(args.out, dpi=150)
        print(f"\nFigure 5 saved to {args.out}")
        plt.close()

    except ImportError:
        print("[info] matplotlib not installed, skipping plot")


if __name__ == "__main__":
    main()
