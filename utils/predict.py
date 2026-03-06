"""
Prediction Utilities

Loads saved Keras models and runs inference for all CNN architectures.
Falls back gracefully to demo predictions when saved weights are not present
(useful for development / demo without a GPU).
"""

import os
import numpy as np
from typing import Any


# ──────────────────────────────────────────────
# Lazy model cache  (loaded once, reused)
# ──────────────────────────────────────────────
_MODEL_CACHE: dict[str, Any] = {}


def _load_model(model_name: str, model_dir: str):
    """
    Load a saved Keras model from disk.
    Returns None if the weight file does not exist (demo mode).
    """
    if model_name in _MODEL_CACHE:
        return _MODEL_CACHE[model_name]

    try:
        import tensorflow as tf  # noqa: F401
        from tensorflow import keras
    except ImportError:
        return None

    path = os.path.join(model_dir, f"{model_name}.keras")
    # Also accept legacy .h5 format
    if not os.path.exists(path):
        path = os.path.join(model_dir, f"{model_name}.h5")

    if not os.path.exists(path):
        return None  # → demo mode

    model = keras.models.load_model(path)
    _MODEL_CACHE[model_name] = model
    return model


# ──────────────────────────────────────────────
# Demo predictions  (used when no weights exist)
# ──────────────────────────────────────────────
_DEMO_RESULTS = {
    "VGG16": {
        "label": "Malignant",
        "confidence": 91.4,
        "precision": 89.2,
        "recall": 93.1,
        "f1": 91.1,
        "mode": "demo",
    },
    "ResNet50V2": {
        "label": "Malignant",
        "confidence": 94.7,
        "precision": 92.8,
        "recall": 95.3,
        "f1": 94.0,
        "mode": "demo",
    },
    "InceptionV3": {
        "label": "Benign",
        "confidence": 87.3,
        "precision": 85.6,
        "recall": 88.9,
        "f1": 87.2,
        "mode": "demo",
    },
}


def _demo_result(model_name: str) -> dict:
    """Return a fixed demo result for the given model."""
    return _DEMO_RESULTS.get(model_name, {
        "label": "Unknown",
        "confidence": 0.0,
        "precision": 0.0,
        "recall": 0.0,
        "f1": 0.0,
        "mode": "demo",
    })


# ──────────────────────────────────────────────
# Core inference
# ──────────────────────────────────────────────
def _run_inference(model, img_array: np.ndarray) -> dict:
    """
    Run binary classification inference.

    Args:
        model:     Loaded Keras model.
        img_array: Preprocessed image, shape (1, H, W, 3).

    Returns:
        dict with keys label, confidence, mode='live'.
    """
    prob = float(model.predict(img_array, verbose=0)[0][0])
    label = "Malignant" if prob >= 0.5 else "Benign"
    confidence = prob * 100 if label == "Malignant" else (1 - prob) * 100

    return {
        "label": label,
        "confidence": round(confidence, 2),
        "mode": "live",
    }


def predict_all_models(img_array: np.ndarray, model_dir: str) -> dict:
    """
    Run inference with VGG16, ResNet50V2, and InceptionV3.

    Each model either:
        a) Loads saved weights from `model_dir` and returns a live prediction, or
        b) Falls back to demo values if weights are absent.

    Args:
        img_array: Preprocessed image, shape (1, H, W, 3).
        model_dir: Directory containing <ModelName>.keras or <ModelName>.h5 files.

    Returns:
        dict keyed by model name, each value containing:
            label        : "Benign" | "Malignant"
            confidence   : float  (0–100)
            precision    : float  (0–100)  — from stored evaluation metrics
            recall       : float  (0–100)
            f1           : float  (0–100)
            mode         : "live" | "demo"
    """
    results = {}
    model_names = ["VGG16", "ResNet50V2", "InceptionV3"]

    # Try to load evaluation metrics saved during training
    eval_metrics = _load_eval_metrics(model_dir)

    for name in model_names:
        model = _load_model(name, model_dir)

        if model is None:
            results[name] = _demo_result(name)
        else:
            pred = _run_inference(model, img_array)
            # Attach precision/recall/f1 from stored evaluation
            metrics = eval_metrics.get(name, {})
            results[name] = {
                **pred,
                "precision": metrics.get("precision", 0.0),
                "recall": metrics.get("recall", 0.0),
                "f1": metrics.get("f1", 0.0),
            }

    return results


def _load_eval_metrics(model_dir: str) -> dict:
    """
    Load evaluation metrics (precision, recall, f1) saved after training.
    Expects a JSON file at <model_dir>/eval_metrics.json.
    Returns empty dict if not found.
    """
    import json
    path = os.path.join(model_dir, "eval_metrics.json")
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {}
