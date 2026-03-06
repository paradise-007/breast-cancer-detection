"""
CNN Model Builders

Defines VGG16, ResNet50V2, and InceptionV3 transfer-learning architectures
for binary breast cancer classification (Benign vs Malignant).

Each builder:
  1. Loads ImageNet pre-trained base (frozen by default).
  2. Adds a custom classification head.
  3. Compiles with Adam + binary cross-entropy.

Usage:
    from models.cnn_models import build_vgg16, build_resnet50v2, build_inceptionv3

    model = build_vgg16(img_size=128, trainable_base=False)
    model.fit(...)
"""

from tensorflow.keras import Sequential
from tensorflow.keras.applications import VGG16, ResNet50V2, InceptionV3
from tensorflow.keras.layers import (
    BatchNormalization,
    Dense,
    Dropout,
    Flatten,
    GlobalAveragePooling2D,
)
from tensorflow.keras.optimizers import Adam


# ─────────────────────────────────────────────────────────
# Shared helper
# ─────────────────────────────────────────────────────────
def _freeze(model, trainable: bool):
    for layer in model.layers:
        layer.trainable = trainable


def _compile(model, lr: float = 1e-4):
    model.compile(
        optimizer=Adam(learning_rate=lr),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model


# ─────────────────────────────────────────────────────────
# VGG16
# ─────────────────────────────────────────────────────────
def build_vgg16(img_size: int = 128, trainable_base: bool = False):
    """
    VGG16 transfer-learning model.

    Architecture:
        VGG16 (ImageNet, no top)
        → Flatten
        → BN → Dense(512, relu) → BN → Dropout(0.5)
        → Dense(256, relu) → BN → Dropout(0.5)
        → Dense(128, relu) → BN → Dropout(0.5)
        → Dense(64,  relu) → Dropout(0.5) → BN
        → Dense(1, sigmoid)

    Args:
        img_size:       Input image dimensions (square). Default 128.
        trainable_base: If True, unfreeze all VGG16 layers for fine-tuning.

    Returns:
        Compiled Keras Sequential model.
    """
    base = VGG16(weights="imagenet", include_top=False, input_shape=(img_size, img_size, 3))
    _freeze(base, trainable_base)

    model = Sequential([
        base,
        Flatten(),
        BatchNormalization(),
        Dense(512, activation="relu"),
        BatchNormalization(),
        Dropout(0.5),
        Dense(256, activation="relu"),
        BatchNormalization(),
        Dropout(0.5),
        Dense(128, activation="relu"),
        BatchNormalization(),
        Dropout(0.5),
        Dense(64, activation="relu"),
        Dropout(0.5),
        BatchNormalization(),
        Dense(1, activation="sigmoid"),
    ], name="VGG16")

    return _compile(model)


# ─────────────────────────────────────────────────────────
# ResNet50V2
# ─────────────────────────────────────────────────────────
def build_resnet50v2(img_size: int = 128, trainable_base: bool = False):
    """
    ResNet50V2 transfer-learning model.

    Architecture:
        ResNet50V2 (ImageNet, no top)
        → Flatten
        → Dense(512, relu) → BN → Dropout(0.5)
        → Dense(256, relu) → BN → Dropout(0.5)
        → Dense(1, sigmoid)

    Args:
        img_size:       Input image dimensions (square). Default 128.
        trainable_base: If True, unfreeze all ResNet layers for fine-tuning.

    Returns:
        Compiled Keras Sequential model.
    """
    base = ResNet50V2(weights="imagenet", include_top=False, input_shape=(img_size, img_size, 3))
    _freeze(base, trainable_base)

    model = Sequential([
        base,
        Flatten(),
        Dense(512, activation="relu"),
        BatchNormalization(),
        Dropout(0.5),
        Dense(256, activation="relu"),
        BatchNormalization(),
        Dropout(0.5),
        Dense(1, activation="sigmoid"),
    ], name="ResNet50V2")

    return _compile(model)


# ─────────────────────────────────────────────────────────
# InceptionV3
# ─────────────────────────────────────────────────────────
def build_inceptionv3(img_size: int = 299, trainable_base: bool = False):
    """
    InceptionV3 transfer-learning model.

    Note: InceptionV3 requires minimum input 75×75. Default img_size=299.
    For 128×128 inputs, pass img_size=128.

    Architecture:
        InceptionV3 (ImageNet, no top)
        → GlobalAveragePooling2D
        → Dense(512, relu) → BN → Dropout(0.4)
        → Dense(256, relu) → BN → Dropout(0.4)
        → Dense(1, sigmoid)

    Args:
        img_size:       Input image dimensions (square). Default 299.
        trainable_base: If True, unfreeze all Inception layers for fine-tuning.

    Returns:
        Compiled Keras Sequential model.
    """
    base = InceptionV3(weights="imagenet", include_top=False, input_shape=(img_size, img_size, 3))
    _freeze(base, trainable_base)

    model = Sequential([
        base,
        GlobalAveragePooling2D(),
        Dense(512, activation="relu"),
        BatchNormalization(),
        Dropout(0.4),
        Dense(256, activation="relu"),
        BatchNormalization(),
        Dropout(0.4),
        Dense(1, activation="sigmoid"),
    ], name="InceptionV3")

    return _compile(model)
