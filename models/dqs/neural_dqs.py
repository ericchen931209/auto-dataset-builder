"""
Neural DQS — MLP Regressor

g: ℝ⁵ → [0,1]
Trained with mAP@0.5 as supervision signal.
See docs/dqs-model.md for full formulation.
"""

import os
import logging
import json
from pathlib import Path

import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import joblib

logger = logging.getLogger(__name__)

MODEL_DIR = os.path.join(os.path.dirname(__file__), "checkpoints")
DEFAULT_MODEL_PATH = os.path.join(MODEL_DIR, "neural_dqs.pkl")


def build_model() -> Pipeline:
    """Build a scikit-learn pipeline: StandardScaler + MLPRegressor."""
    return Pipeline([
        ("scaler", StandardScaler()),
        ("mlp", MLPRegressor(
            hidden_layer_sizes=(32, 16),
            activation="relu",
            solver="adam",
            alpha=1e-3,           # L2 regularization
            max_iter=500,
            random_state=42,
            early_stopping=True,
            validation_fraction=0.15,
            n_iter_no_change=20,
        )),
    ])


def train(
    features: list[list[float]],    # shape (M, 5)
    map_scores: list[float],         # shape (M,)  — mAP@0.5 ground truth
    save_path: str = DEFAULT_MODEL_PATH,
) -> dict:
    """
    Train Neural DQS on a list of (feature_vector, mAP) pairs.
    Returns training metrics.
    """
    X = np.array(features, dtype=np.float32)
    y = np.array(map_scores, dtype=np.float32)

    if len(X) < 5:
        raise ValueError(f"Need at least 5 training samples, got {len(X)}")

    model = build_model()
    model.fit(X, y)

    # Training metrics
    y_pred = model.predict(X)
    residuals = y - y_pred
    metrics = {
        "n_samples": len(X),
        "train_mse": float(np.mean(residuals ** 2)),
        "train_mae": float(np.mean(np.abs(residuals))),
        "train_r2": float(1 - np.var(residuals) / np.var(y)),
    }

    # Compute Pearson r
    corr = np.corrcoef(y, y_pred)
    metrics["pearson_r"] = float(corr[0, 1])

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, save_path)
    logger.info(f"Neural DQS trained: r={metrics['pearson_r']:.3f}, saved to {save_path}")

    return metrics


def predict(
    features: list[float],          # length 5
    model_path: str = DEFAULT_MODEL_PATH,
) -> float:
    """
    Predict DQS score for a single dataset given its feature vector.
    Returns a float in [0, 1].
    """
    if not os.path.exists(model_path):
        logger.warning("Neural DQS model not found — returning heuristic score")
        return _heuristic_dqs(features)

    model = joblib.load(model_path)
    X = np.array([features], dtype=np.float32)
    score = float(model.predict(X)[0])
    return max(0.0, min(1.0, score))


def predict_with_shap(
    features: list[float],
    model_path: str = DEFAULT_MODEL_PATH,
    background_data: np.ndarray | None = None,
) -> dict:
    """
    Predict DQS and compute SHAP values for interpretability.
    Returns {"score": float, "shap_values": {"annotation_quality": float, ...}}
    """
    score = predict(features, model_path)

    try:
        import shap

        model = joblib.load(model_path)
        mlp = model.named_steps["mlp"]
        scaler = model.named_steps["scaler"]

        X = scaler.transform(np.array([features], dtype=np.float32))

        if background_data is None:
            background_data = np.zeros((1, 5), dtype=np.float32)
        else:
            background_data = scaler.transform(background_data)

        explainer = shap.KernelExplainer(mlp.predict, background_data)
        shap_vals = explainer.shap_values(X, nsamples=100)

        feature_names = [
            "annotation_quality", "diversity", "lighting_diversity",
            "pose_diversity", "class_balance"
        ]
        shap_dict = {name: float(val) for name, val in zip(feature_names, shap_vals[0])}

        return {"score": score, "shap_values": shap_dict}

    except ImportError:
        logger.warning("shap not installed — returning score only")
        return {"score": score, "shap_values": None}


def _heuristic_dqs(features: list[float]) -> float:
    """
    Geometric mean of all features as a fallback when model is not trained yet.
    This is equivalent to the balanced weighting baseline.
    """
    arr = np.array(features, dtype=np.float64)
    arr = np.clip(arr, 1e-6, 1.0)
    return float(np.exp(np.mean(np.log(arr))))
