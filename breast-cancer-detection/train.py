"""
Training Script — Breast Cancer Detection CNNs

Trains VGG16, ResNet50V2, and InceptionV3 on the BreaKHis dataset,
evaluates each model, and saves weights + evaluation metrics.

Dataset expected layout (Kaggle BreaKHis):
    data/
      benign/   (any sub-folder structure is fine, Keras walks recursively)
      malignant/

Usage:
    python train.py --data_dir ./data --epochs 20 --batch_size 32

Outputs:
    models/saved/VGG16.keras
    models/saved/ResNet50V2.keras
    models/saved/InceptionV3.keras
    models/saved/eval_metrics.json
    models/saved/training_plots/  (PNG accuracy & loss curves)
"""

import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")  # non-interactive backend for servers
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from models.cnn_models import build_vgg16, build_resnet50v2, build_inceptionv3


# ─────────────────────────────────────────────────────────
# Argument Parsing
# ─────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description="Train breast cancer CNN models")
    p.add_argument("--data_dir",   default="./data",    help="Root dataset directory")
    p.add_argument("--save_dir",   default="./models/saved", help="Where to save weights")
    p.add_argument("--img_size",   type=int, default=128)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--epochs",     type=int, default=20)
    p.add_argument("--use_smote",  action="store_true", help="Apply SMOTE oversampling")
    p.add_argument("--models",     nargs="+",
                   default=["VGG16", "ResNet50V2", "InceptionV3"],
                   help="Which models to train")
    return p.parse_args()


# ─────────────────────────────────────────────────────────
# Data Loading
# ─────────────────────────────────────────────────────────
def load_data(data_dir: str, img_size: int, batch_size: int):
    """
    Use ImageDataGenerator to load images from a directory.
    Returns (X, y) numpy arrays after exhausting all batches.
    """
    datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        brightness_range=[0.8, 1.2],
        zoom_range=[0.99, 1.01],
        horizontal_flip=True,
        fill_mode="constant",
    )

    generator = datagen.flow_from_directory(
        directory=data_dir,
        target_size=(img_size, img_size),
        batch_size=batch_size,
        shuffle=False,
        class_mode="binary",
    )

    total = generator.samples
    print(f"Found {total} images in {len(generator.class_indices)} classes: {generator.class_indices}")

    # Load all at once (adjust batch_size if RAM is limited)
    all_x, all_y = [], []
    steps = int(np.ceil(total / batch_size))
    for _ in range(steps):
        x_batch, y_batch = next(generator)
        all_x.append(x_batch)
        all_y.append(y_batch)

    X = np.concatenate(all_x, axis=0)[:total]
    y = np.concatenate(all_y, axis=0)[:total]
    return X, y


# ─────────────────────────────────────────────────────────
# SMOTE Oversampling
# ─────────────────────────────────────────────────────────
def apply_smote(X: np.ndarray, y: np.ndarray, img_size: int):
    """Flatten → SMOTE → reshape back."""
    print("Applying SMOTE oversampling...")
    n_samples, h, w, c = X.shape
    X_flat = X.reshape(n_samples, h * w * c)
    sm = SMOTE(random_state=42)
    X_res, y_res = sm.fit_resample(X_flat, y)
    X_res = X_res.reshape(-1, h, w, c)
    print(f"After SMOTE: {X_res.shape}, labels: {np.bincount(y_res.astype(int))}")
    return X_res, y_res


# ─────────────────────────────────────────────────────────
# Training + Evaluation
# ─────────────────────────────────────────────────────────
def train_model(model, model_name, X_train, y_train, X_val, y_val,
                save_dir, epochs, batch_size):
    """Train a single model with callbacks; return history."""
    os.makedirs(save_dir, exist_ok=True)
    weight_path = os.path.join(save_dir, f"{model_name}.keras")

    callbacks = [
        ModelCheckpoint(
            filepath=weight_path,
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1,
        ),
        EarlyStopping(
            monitor="val_loss",
            patience=5,
            restore_best_weights=True,
            verbose=1,
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=3,
            min_lr=1e-6,
            verbose=1,
        ),
    ]

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1,
    )

    print(f"\n✓ {model_name} saved to {weight_path}")
    return history


def evaluate_model(model, model_name, X_test, y_test) -> dict:
    """Evaluate and return a metrics dict."""
    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    y_pred = np.round(model.predict(X_test, verbose=0)).astype(int).flatten()

    report = classification_report(
        y_test, y_pred,
        target_names=["Benign", "Malignant"],
        output_dict=True,
    )

    avg = report["weighted avg"]
    metrics = {
        "accuracy":  round(accuracy * 100, 2),
        "loss":      round(loss, 4),
        "precision": round(avg["precision"] * 100, 2),
        "recall":    round(avg["recall"] * 100, 2),
        "f1":        round(avg["f1-score"] * 100, 2),
    }

    print(f"\n── {model_name} Evaluation ──")
    for k, v in metrics.items():
        print(f"  {k:12s}: {v}")

    return metrics


# ─────────────────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────────────────
def save_plots(histories: dict, save_dir: str):
    """Save accuracy and loss curves for all trained models."""
    plot_dir = os.path.join(save_dir, "training_plots")
    os.makedirs(plot_dir, exist_ok=True)

    for metric, ylabel in [("accuracy", "Accuracy"), ("loss", "Loss")]:
        plt.figure(figsize=(8, 5))
        for model_name, history in histories.items():
            if metric in history.history:
                plt.plot(history.history[metric], label=f"{model_name} train")
            val_key = f"val_{metric}"
            if val_key in history.history:
                plt.plot(history.history[val_key], linestyle="--", label=f"{model_name} val")

        plt.title(f"Training {ylabel}")
        plt.xlabel("Epoch")
        plt.ylabel(ylabel)
        plt.legend()
        plt.tight_layout()
        path = os.path.join(plot_dir, f"{metric}_curve.png")
        plt.savefig(path, dpi=150)
        plt.close()
        print(f"Saved plot: {path}")


# ─────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────
MODEL_BUILDERS = {
    "VGG16":       build_vgg16,
    "ResNet50V2":  build_resnet50v2,
    "InceptionV3": build_inceptionv3,
}


def main():
    args = parse_args()

    # 1. Load data
    print(f"\nLoading data from: {args.data_dir}")
    X, y = load_data(args.data_dir, args.img_size, args.batch_size)

    # 2. SMOTE (optional)
    if args.use_smote:
        X, y = apply_smote(X, y, args.img_size)

    # 3. Train / val / test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.20, random_state=42, stratify=y_train
    )

    print(f"\nSplit → Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")

    # 4. Train each model
    all_metrics = {}
    all_histories = {}

    for model_name in args.models:
        if model_name not in MODEL_BUILDERS:
            print(f"Unknown model '{model_name}', skipping.")
            continue

        print(f"\n{'='*50}")
        print(f"  Training {model_name}")
        print(f"{'='*50}")

        img_size = 299 if model_name == "InceptionV3" else args.img_size
        builder = MODEL_BUILDERS[model_name]
        model = builder(img_size=img_size, trainable_base=False)
        model.summary()

        history = train_model(
            model, model_name,
            X_train, y_train, X_val, y_val,
            args.save_dir, args.epochs, args.batch_size,
        )

        metrics = evaluate_model(model, model_name, X_test, y_test)
        all_metrics[model_name] = metrics
        all_histories[model_name] = history

    # 5. Save metrics JSON
    metrics_path = os.path.join(args.save_dir, "eval_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(all_metrics, f, indent=2)
    print(f"\n✓ Metrics saved to {metrics_path}")

    # 6. Plots
    save_plots(all_histories, args.save_dir)

    # 7. Summary table
    print("\n" + "="*55)
    print(f"{'Model':<15} {'Accuracy':>10} {'Precision':>10} {'Recall':>8} {'F1':>8}")
    print("-"*55)
    for model_name, m in all_metrics.items():
        print(f"{model_name:<15} {m['accuracy']:>9.2f}% {m['precision']:>9.2f}% {m['recall']:>7.2f}% {m['f1']:>7.2f}%")
    print("="*55)


if __name__ == "__main__":
    main()
