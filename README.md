# 🔬 MammoScan AI — Breast Cancer Detection

<div align="center">

![Python](https://img.shields.io/badge/Python-3.11-blue?style=for-the-badge&logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.16-orange?style=for-the-badge&logo=tensorflow&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.32-red?style=for-the-badge&logo=streamlit&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-3.0-black?style=for-the-badge&logo=flask&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

**An AI-powered web application for breast cancer detection using Convolutional Neural Networks.**

Upload a mammogram or histology image → Get instant predictions from 3 CNN models simultaneously.

[🚀 Live Demo](#-live-demo) · [📦 Installation](#-installation) · [🏋️ Training](#️-training-the-models) · [📁 Project Structure](#-project-structure)

</div>

---

## 📌 Overview

Breast cancer is one of the most common cancers among women worldwide. Early detection significantly improves survival rates. This project leverages **transfer learning** with three state-of-the-art CNN architectures to automatically analyze mammogram and histology images, classifying them as **Benign** or **Malignant**.

The project includes:
- A **Streamlit web app** (deployed online, mobile-friendly)
- A **Flask web app** (for local use)
- A complete **training pipeline** with SMOTE oversampling, callbacks, and evaluation metrics
- **Demo mode** — runs with example predictions even without trained weights

---

## 🚀 Live Demo

> 🔗 **[Open Live App](https://paradise-007-breast-cancer-detection-streamlit-app.streamlit.app)**  
> Works on mobile, tablet, and desktop — no installation needed.

---

## 🧠 Models

Three CNN architectures trained on the [BreaKHis Dataset](https://www.kaggle.com/datasets/ambarish/breakhis):

| Model | Architecture | Input Size | Highlights |
|-------|-------------|-----------|------------|
| **VGG16** | 16-layer deep CNN + custom head | 128×128 | Strong baseline, deep feature extraction |
| **ResNet50V2** | 50-layer residual network + custom head | 128×128 | Best accuracy, handles vanishing gradients |
| **InceptionV3** | Multi-scale Inception modules + GAP | 128×128 | Most efficient, learns varied features |

All models use:
- ✅ **ImageNet pre-trained weights** (transfer learning)
- ✅ **Frozen base layers** (fine-tune only the head)
- ✅ **Adam optimizer** + binary cross-entropy loss
- ✅ **EarlyStopping**, **ReduceLROnPlateau**, **ModelCheckpoint** callbacks
- ✅ **SMOTE** oversampling to handle class imbalance

---

## 🖥️ App Features

- 📤 **Drag-and-drop** image upload (PNG, JPG, TIFF, BMP)
- 🔍 **Three models analyzed simultaneously**
- 📊 **Per-model results**: confidence score, precision, recall, F1-score
- 📋 **Model comparison table** side by side
- 💡 **Demo mode** when no weights are present
- 📱 **Fully mobile responsive**
- ⚠️ Medical disclaimer built in

---

## 📁 Project Structure

```
breast-cancer-detection/
│
├── streamlit_app.py          # Streamlit app (Streamlit Cloud deployment)
├── app.py                    # Flask app entry point (local)
├── train.py                  # Training script with CLI
├── requirements.txt          # Dependencies
├── Dockerfile                # Docker config for deployment
│
├── app/
│   ├── __init__.py           # Flask app factory
│   ├── config.py             # Configuration (paths, upload limits)
│   └── routes.py             # API endpoints: GET / and POST /predict
│
├── models/
│   ├── cnn_models.py         # VGG16, ResNet50V2, InceptionV3 builders
│   └── saved/                # Place trained .keras weights here
│       └── eval_metrics.json # Auto-generated after training
│
├── utils/
│   ├── preprocess.py         # Image loading, resizing, normalization, TTA
│   └── predict.py            # Model loader + inference runner
│
├── templates/
│   └── index.html            # Flask HTML template
│
└── static/
    ├── css/style.css         # Dark medical UI stylesheet
    └── js/main.js            # Upload, API calls, results rendering
```

---

## 📦 Installation

### 1. Clone the repository

```bash
git clone https://github.com/paradise-007/breast-cancer-detection.git
cd breast-cancer-detection
```

### 2. Create a virtual environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the app

```bash
# Streamlit (recommended)
streamlit run streamlit_app.py

# OR Flask
python app.py
```

Open → `http://localhost:8501` (Streamlit) or `http://localhost:5050` (Flask)

> **Demo Mode**: The app works immediately without any trained weights — it shows example predictions so you can explore the UI.

---

## 🏋️ Training the Models

### 1. Download the BreaKHis Dataset

→ [Kaggle: BreaKHis Dataset](https://www.kaggle.com/datasets/ambarish/breakhis)

### 2. Organize your data folder

```
data/
  benign/         ← all benign images (sub-folders are fine)
  malignant/      ← all malignant images
```

### 3. Run training

```bash
# Basic training
python train.py --data_dir ./data --epochs 20

# With SMOTE oversampling (recommended for imbalanced data)
python train.py --data_dir ./data --epochs 20 --use_smote

# Train specific models only
python train.py --data_dir ./data --models VGG16 ResNet50V2

# All options
python train.py --help
```

### 4. Training outputs

After training, these files are saved to `models/saved/`:

```
models/saved/
  VGG16.keras
  ResNet50V2.keras
  InceptionV3.keras
  eval_metrics.json         ← precision / recall / F1 per model
  training_plots/
    accuracy_curve.png
    loss_curve.png
```

---

## 📊 Dataset

**BreaKHis (Breast Cancer Histopathological Database)**
- 7,909 microscopic images of breast tumor tissue
- Two classes: **Benign** and **Malignant**
- Four magnification factors: 40×, 100×, 200×, 400×
- Image size: 700×460 pixels (resized to 128×128 for training)

---

## 🔌 API Reference (Flask)

### `GET /`
Serves the web UI.

### `POST /predict`

**Request:** `multipart/form-data` with an image file

**Response:**
```json
{
  "filename": "abc123.png",
  "results": {
    "VGG16": {
      "label": "Malignant",
      "confidence": 91.4,
      "precision": 89.2,
      "recall": 93.1,
      "f1": 91.1,
      "mode": "live"
    },
    "ResNet50V2": { "..." },
    "InceptionV3": { "..." }
  }
}
```

`mode` is `"live"` when trained weights are loaded, `"demo"` otherwise.

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| ML Framework | TensorFlow / Keras |
| Web App | Streamlit + Flask |
| Image Processing | Pillow, NumPy |
| Data Balancing | imbalanced-learn (SMOTE) |
| Evaluation | scikit-learn |
| Visualization | Matplotlib |
| Deployment | Streamlit Community Cloud |

---

## ⚠️ Disclaimer

> This tool is for **research and educational purposes only**.  
> Results must be reviewed by a **qualified radiologist**.  
> This application is **not intended for clinical diagnosis** or treatment decisions.  
> Always consult a medical professional for health concerns.

---

## 📄 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgements

- [BreaKHis Dataset](https://web.inf.ufpr.br/vri/databases/breast-cancer-histopathological-database-breakhis/) — Spanhol et al., 2016
- [TensorFlow / Keras](https://www.tensorflow.org/) — Model building and training
- [Streamlit](https://streamlit.io/) — Web app framework

---

<div align="center">
  Made with ❤️ by <a href="https://github.com/paradise-007">paradise-007</a>
</div>
