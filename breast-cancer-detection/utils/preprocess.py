"""
Image Preprocessing Utilities

Handles loading, resizing, normalizing and augmenting
mammogram/histology images before feeding to CNN models.
"""

import numpy as np
from PIL import Image


def preprocess_image(image_path: str, img_size: int = 128) -> np.ndarray:
    """
    Load an image from disk and preprocess it for inference.

    Steps:
        1. Open the image with Pillow (handles PNG, JPG, TIFF, BMP)
        2. Convert to RGB (handles grayscale or RGBA inputs)
        3. Resize to (img_size, img_size) using LANCZOS resampling
        4. Convert to float32 numpy array
        5. Normalize pixel values to [0, 1]
        6. Add batch dimension → shape (1, img_size, img_size, 3)

    Args:
        image_path: Path to the image file on disk.
        img_size:   Target square size. Defaults to 128 (matches training).

    Returns:
        numpy array of shape (1, img_size, img_size, 3), dtype float32.
    """
    img = Image.open(image_path).convert("RGB")
    img = img.resize((img_size, img_size), Image.LANCZOS)
    arr = np.array(img, dtype=np.float32) / 255.0
    return np.expand_dims(arr, axis=0)  # (1, H, W, 3)


def augment_tta(image_path: str, img_size: int = 128, n: int = 5) -> list[np.ndarray]:
    """
    Test-Time Augmentation (TTA): generate n augmented versions of the image.
    Averaging predictions over augmented copies reduces variance.

    Augmentations applied randomly:
        - Horizontal flip
        - Slight brightness jitter ±0.1
        - Slight zoom (crop 95 % then resize back)

    Args:
        image_path: Path to the image file.
        img_size:   Target square size.
        n:          Number of augmented copies to produce.

    Returns:
        List of n numpy arrays, each of shape (1, img_size, img_size, 3).
    """
    rng = np.random.default_rng(seed=42)
    copies = []

    img = Image.open(image_path).convert("RGB")
    img = img.resize((img_size, img_size), Image.LANCZOS)
    arr = np.array(img, dtype=np.float32) / 255.0

    for _ in range(n):
        a = arr.copy()

        # Horizontal flip
        if rng.random() > 0.5:
            a = np.fliplr(a)

        # Brightness jitter
        delta = rng.uniform(-0.10, 0.10)
        a = np.clip(a + delta, 0.0, 1.0)

        # Zoom-in crop
        crop_frac = rng.uniform(0.90, 1.00)
        margin = int(img_size * (1 - crop_frac) / 2)
        if margin > 0:
            cropped = a[margin:-margin, margin:-margin, :]
            pil_crop = Image.fromarray((cropped * 255).astype(np.uint8))
            pil_crop = pil_crop.resize((img_size, img_size), Image.LANCZOS)
            a = np.array(pil_crop, dtype=np.float32) / 255.0

        copies.append(np.expand_dims(a, axis=0))

    return copies
