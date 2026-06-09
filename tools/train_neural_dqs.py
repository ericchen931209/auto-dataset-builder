"""
Neural DQS Training Script (V1.0)

Takes the CSV output from collect_dqs_data.py and trains the Neural DQS MLP.
Saves the trained model and produces a correlation report for the paper.

Usage:
    python3 tools/train_neural_dqs.py --data dqs_training_data.csv
    python3 tools/train_neural_dqs.py --data dqs_training_data.csv --plot
"""
import argparse
import csv
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

FEATURE_COLS = [
    "annotation_quality", "diversity", "lighting_diversity",
    "pose_diversity", "class_balance",
]
TARGET_COL = "map50"
DEFAULT_MODEL_PATH = "models/dqs/neural_dqs_model.pkl"


def load_data(csv_path: str) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Returns (X, y, variant_ids)."""
    rows = []
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            try:
                x = [float(row[c]) for c in FEATURE_COLS]
                y = float(row[TARGET_COL])
                if y > 0:   # skip rows where YOLO failed
                    rows.append((x, y, row["variant_id"]))
            except (KeyError, ValueError):
                continue

    if not rows:
        raise ValueError("No valid rows found in CSV")

    X = np.array([r[0] for r in rows])
    y = np.array([r[1] for r in rows])
    ids = [r[2] for r in rows]
    return X, y, ids


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data",  required=True, help="Path to dqs_training_data.csv")
    parser.add_argument("--model", default=DEFAULT_MODEL_PATH, help="Output model .pkl path")
    parser.add_argument("--plot",  action="store_true", help="Show scatter plot")
    args = parser.parse_args()

    print(f"\n{'='*60}")
    print(f"  Neural DQS Training")
    print(f"{'='*60}\n")

    print(f"Loading data from {args.data}…")
    X, y, ids = load_data(args.data)
    print(f"  {len(y)} records | mAP50 range: {y.min():.4f}–{y.max():.4f}\n")

    from models.dqs.neural_dqs import train

    feature_list = [list(row) for row in X]
    map_scores   = list(y)

    os.makedirs(Path(args.model).parent, exist_ok=True)

    print("Training MLP regressor…")
    metrics = train(feature_list, map_scores, save_path=args.model)

    print(f"\n{'─'*60}")
    print(f"  Training results")
    print(f"{'─'*60}")
    for k, v in metrics.items():
        print(f"  {k:<30} {v}")
    print(f"{'─'*60}\n")

    # Feature importance via correlation
    print("Feature–mAP Pearson correlations:")
    for i, col in enumerate(FEATURE_COLS):
        r = np.corrcoef(X[:, i], y)[0, 1]
        bar = "█" * int(abs(r) * 20)
        sign = "+" if r >= 0 else "-"
        print(f"  {col:<25} {sign}{abs(r):.4f}  {bar}")

    report = {
        "n_samples": len(y),
        "mAP50_range": [round(float(y.min()), 4), round(float(y.max()), 4)],
        "train_metrics": metrics,
        "feature_correlations": {
            col: round(float(np.corrcoef(X[:, i], y)[0, 1]), 4)
            for i, col in enumerate(FEATURE_COLS)
        },
    }

    report_path = str(Path(args.data).with_suffix(".report.json"))
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nReport saved to {report_path}")
    print(f"Model saved to  {args.model}\n")

    if args.plot:
        try:
            import matplotlib.pyplot as plt
            from models.dqs.neural_dqs import predict

            y_pred = np.array([predict(list(row), model_path=args.model) for row in X])
            plt.figure(figsize=(7, 6))
            plt.scatter(y, y_pred, alpha=0.7, edgecolors="k", linewidths=0.5)
            mn, mx = min(y.min(), y_pred.min()), max(y.max(), y_pred.max())
            plt.plot([mn, mx], [mn, mx], "r--", linewidth=1.5, label="perfect fit")
            plt.xlabel("Actual mAP@0.5")
            plt.ylabel("Predicted DQS")
            plt.title(f"Neural DQS vs Actual mAP  (Pearson r = {metrics.get('pearson_r', '?')})")
            plt.legend()
            plt.tight_layout()
            plt.savefig("dqs_scatter.png", dpi=150)
            print("Scatter plot saved to dqs_scatter.png")
            plt.show()
        except ImportError:
            print("[info] matplotlib not installed, skipping plot")


if __name__ == "__main__":
    main()
