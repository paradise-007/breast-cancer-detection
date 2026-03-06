"""
Flask Routes — handles upload and prediction endpoints
"""

import os
import uuid
from flask import Blueprint, render_template, request, jsonify, current_app
from werkzeug.utils import secure_filename

from utils.preprocess import preprocess_image
from utils.predict import predict_all_models

main = Blueprint("main", __name__)


def allowed_file(filename):
    return (
        "." in filename
        and filename.rsplit(".", 1)[1].lower()
        in current_app.config["ALLOWED_EXTENSIONS"]
    )


@main.route("/")
def index():
    return render_template("index.html")


@main.route("/predict", methods=["POST"])
def predict():
    """
    Accepts a multipart/form-data image upload.
    Returns JSON with predictions from all three CNN models.
    """
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files["file"]

    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400

    if not allowed_file(file.filename):
        return jsonify({"error": "File type not allowed. Use PNG, JPG, JPEG, TIFF or BMP"}), 400

    # Save file with unique name
    ext = file.filename.rsplit(".", 1)[1].lower()
    unique_name = f"{uuid.uuid4().hex}.{ext}"
    upload_path = os.path.join(current_app.config["UPLOAD_FOLDER"], unique_name)
    os.makedirs(current_app.config["UPLOAD_FOLDER"], exist_ok=True)
    file.save(upload_path)

    # Preprocess
    img_array = preprocess_image(upload_path, current_app.config["IMG_SIZE"])

    # Run all models
    results = predict_all_models(
        img_array,
        model_dir=current_app.config["MODEL_FOLDER"],
    )

    return jsonify({
        "filename": unique_name,
        "results": results,
    })


@main.route("/health")
def health():
    return jsonify({"status": "ok"})
