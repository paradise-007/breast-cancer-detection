---
title: MammoScan AI
emoji: 🔬
colorFrom: pink
colorTo: purple
sdk: streamlit
sdk_version: 1.32.0
app_file: streamlit_app.py
pinned: false
license: mit
---

# MammoScan AI — Breast Cancer Detection

A full-stack Python web application for breast cancer detection using Convolutional Neural Networks.

Upload a mammogram or histology image — three CNN models (VGG16, ResNet50V2, InceptionV3) analyze it and return confidence scores, precision, recall, and F1.

---

## 🚀 Deployment

**Hugging Face Spaces** (free, mobile-friendly): push this repo to a Space and it auto-deploys.
**Local Streamlit**: `streamlit run streamlit_app.py`
**Local Flask**: `python app.py`

> **Demo Mode**: Without trained weights in `models/saved/`, the app shows example predictions.

---

## ⚡ Quick Start

```bash
pip install -r requirements.txt

# Streamlit (recommended)
streamlit run streamlit_app.py

# Flask
python app.py
```

## 🏋️ Train Models

```bash
# data/benign/ and data/malignant/ folders required
python train.py --data_dir ./data --epochs 20 --use_smote
```

---

## ⚠️ Disclaimer

For **research and educational purposes only**. Not for clinical diagnosis.
