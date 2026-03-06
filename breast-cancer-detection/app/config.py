"""
Application Configuration
"""

import os


class Config:
    # Flask
    SECRET_KEY = os.environ.get("SECRET_KEY", "mammoscan-dev-secret-key-change-in-prod")
    DEBUG = os.environ.get("DEBUG", "True") == "True"

    # File Upload
    UPLOAD_FOLDER = os.path.join(os.path.dirname(os.path.dirname(__file__)), "static", "uploads")
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16 MB
    ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "tiff", "bmp"}

    # Model
    MODEL_FOLDER = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models", "saved")
    IMG_SIZE = 128
    BATCH_SIZE = 1

    # Available CNN models
    MODELS = ["VGG16", "ResNet50V2", "InceptionV3"]


class ProductionConfig(Config):
    DEBUG = False
    SECRET_KEY = os.environ.get("SECRET_KEY")  # Must be set in production
