"""
Neural DQS — Regressor (Ridge + optional MLP)

g: ℝ⁶ → [0,1]
Trained with mAP@0.5 as supervision signal.
Ridge regression is the default — better suited for small datasets (<100 samples).
MLP is used when n_samples >= 100.
See docs/dqs-model.md for full formulation.
"""

import os
import logging
from pathlib import Path

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import Pipeline
import joblib

logger = logging.getLogger(__name__)

MODEL_DIR = os.path.dirname(__file__)
DEFAULT_MODEL_PATH = os.path.join(MODEL_DIR, "neural_dqs_model.pkl")

FEATURE_NAMES = [
    "annotation_quality", "sharpness", "clip_diversity",
    "lighting_diversity", "pose_diversity", "class_balance",
]


def build_model(n_samples: int = 0) -> Pipeline:
    """Ridge (degree-2 poly) for small datasets; MLP for large."""
    if n_samples < 100:
        return Pipeline([
            ("scaler", StandardScaler()),
            ("poly", PolynomialFeatures(degree=2, include_bias=False)),
            ("ridge", Ridge(alpha=1.0)),
        ])
    return Pipeline([
        ("scaler", StandardScaler()),
        ("mlp", MLPRegressor(
            hidden_layer_sizes=(64, 32),
            activation="relu",
            solver="adam",
            alpha=1e-3,
            max_iter=1000,
            random_state=42,
            early_stopping=True,
            validation_fraction=0.15,
            n_iter_no_change=20,
        )),
    ])


def train(
    features: list[list[float]],
    map_scores: list[float],
    save_path: str = DEFAULT_MODEL_PATH,
) -> dict:
    """Train Neural DQS and return metrics dict."""
    X = np.array(features, dtype=np.float32)
    y = np.array(map_scores, dtype=np.float32)

    if len(X) < 5:
        raise ValueError(f"Need at least 5 training samples, got {len(X)}")

    model = build_model(n_samples=len(X))
    model.fit(X, y)

    y_pred = model.predict(X)
    residuals = y - y_pred
    metrics = {
        "n_samples": len(X),
        "train_mse": float(np.mean(residuals ** 2)),
        "train_mae": float(np.mean(np.abs(residuals))),
        "train_r2": float(1 - np.var(residuals) / np.var(y)),
        "pearson_r": float(np.corrcoef(y, y_pred)[0, 1]),
    }

    if len(X) >= 10:
        from sklearn.model_selection import cross_val_predict, KFold
        cv = KFold(n_splits=min(5, len(X)), shuffle=True, random_state=42)
        y_cv = cross_val_predict(build_model(n_samples=len(X)), X, y, cv=cv)
        metrics["cv_pearson_r"] = float(np.corrcoef(y, y_cv)[0, 1])
        cv_res = y - y_cv
        metrics["cv_mse"] = float(np.mean(cv_res ** 2))
        metrics["cv_r2"]  = float(1 - np.var(cv_res) / np.var(y))

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, save_path)
    logger.info(f"Neural DQS trained: r={metrics['pearson_r']:.3f}, saved to {save_path}")
    return metrics


def predict(
    features: list[float],
    model_path: str = DEFAULT_MODEL_PATH,
) -> float:
    """Predict DQS score for a single dataset. Returns float in [0,1]."""
    if not os.path.exists(model_path):
        logger.warning("Neural DQS model not found — returning heuristic score")
        return _heuristic_dqs(features)

    model = joblib.load(model_path)
    score = float(model.predict(np.array([features], dtype=np.float32))[0])
    return max(0.0, min(1.0, score))


def predict_with_shap(
    features: list[float],
    model_path: str = DEFAULT_MODEL_PATH,
    background_data: "np.ndarray | None" = None,
) -> dict:
    """
    Predict DQS and compute SHAP values in the original 6-feature space.

    Uses KernelExplainer wrapping the full pipeline so SHAP values correspond
    to [AQ, IQ, CD, LD, PD, CB] regardless of whether the model is Ridge or MLP.

    Returns {"score": float, "shap_values": {feature_name: float, ...}}
    """
    score = predict(features, model_path)

    try:
        import shap

        model = joblib.load(model_path)
        X_point = np.array([features], dtype=np.float32)

        if background_data is None:
            # Use zero-vector background as neutral baseline
            background_data = np.zeros((1, len(features)), dtype=np.float32)

        explainer = shap.KernelExplainer(
            lambda x: model.predict(x.astype(np.float32)),
            background_data,
        )
        shap_vals = explainer.shap_values(X_point, nsamples=200, silent=True)
        shap_dict = {
            name: float(val)
            for name, val in zip(FEATURE_NAMES, shap_vals[0])
        }
        return {"score": score, "shap_values": shap_dict}

    except ImportError:
        logger.warning("shap not installed — returning score only")
        return {"score": score, "shap_values": None}


def compute_shap_importance(
    X: np.ndarray,
    model_path: str = DEFAULT_MODEL_PATH,
    nsamples: int = 200,
) -> dict[str, float]:
    """
    Compute mean |SHAP| importance across all samples in X.
    Returns {feature_name: mean_abs_shap}.
    """
    try:
        import shap
    except ImportError:
        raise RuntimeError("shap not installed — run: pip install shap")

    model = joblib.load(model_path)
    X = X.astype(np.float32)

    # Use training mean as background for better attribution
    background = X.mean(axis=0, keepdims=True)

    explainer = shap.KernelExplainer(
        lambda x: model.predict(x.astype(np.float32)),
        background,
    )
    shap_vals = explainer.shap_values(X, nsamples=nsamples, silent=True)
    mean_abs = np.abs(shap_vals).mean(axis=0)
    return {name: float(v) for name, v in zip(FEATURE_NAMES, mean_abs)}


def _heuristic_dqs(features: list[float]) -> float:
    arr = np.clip(np.array(features, dtype=np.float64), 1e-6, 1.0)
    return float(np.exp(np.mean(np.log(arr))))
