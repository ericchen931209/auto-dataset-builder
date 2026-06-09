"""
Training script for Neural DQS.

Usage:
    python -m models.dqs.train_dqs --data training_data.json

training_data.json format:
    [
        {"features": [0.8, 0.6, 0.7, 0.5, 1.0], "map": 0.72},
        ...
    ]
"""

import argparse
import json
import logging
from models.dqs.neural_dqs import train

logging.basicConfig(level=logging.INFO)


def main():
    parser = argparse.ArgumentParser(description="Train Neural DQS model")
    parser.add_argument("--data", required=True, help="Path to training_data.json")
    parser.add_argument("--save", default=None, help="Path to save model (.pkl)")
    args = parser.parse_args()

    with open(args.data) as f:
        data = json.load(f)

    features = [item["features"] for item in data]
    map_scores = [item["map"] for item in data]

    print(f"Training on {len(features)} samples...")
    kwargs = {}
    if args.save:
        kwargs["save_path"] = args.save

    metrics = train(features, map_scores, **kwargs)

    print("\n=== Training Results ===")
    print(f"  Samples:   {metrics['n_samples']}")
    print(f"  Pearson r: {metrics['pearson_r']:.4f}")
    print(f"  Train MSE: {metrics['train_mse']:.4f}")
    print(f"  Train MAE: {metrics['train_mae']:.4f}")
    print(f"  Train R²:  {metrics['train_r2']:.4f}")


if __name__ == "__main__":
    main()
