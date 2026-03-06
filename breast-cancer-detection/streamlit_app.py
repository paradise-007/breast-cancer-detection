"""
MammoScan AI — Streamlit Interface
Breast Cancer Detection using VGG16, ResNet50V2, InceptionV3

Deploy to: Hugging Face Spaces (free, mobile-friendly)
"""

import io
import os
import sys

import numpy as np
import streamlit as st
from PIL import Image

# ── Page config (must be first Streamlit call) ──────────────────────────────
st.set_page_config(
    page_title="MammoScan AI",
    page_icon="🔬",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ── Inline CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600&family=DM+Mono:wght@500&display=swap');

  html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

  .block-container { padding-top: 1.5rem; padding-bottom: 2rem; }

  /* Cards */
  .card {
    background: #0d0d1a;
    border: 1px solid #2a2a3e;
    border-radius: 14px;
    padding: 1.2rem 1.4rem;
    margin-bottom: 1rem;
  }

  /* Result label */
  .malignant { color: #e84393; font-size: 2rem; font-weight: 700; }
  .benign    { color: #00c9a7; font-size: 2rem; font-weight: 700; }

  /* Metric pill */
  .pill {
    display: inline-block;
    background: #1a1a2e;
    border-radius: 8px;
    padding: 0.4rem 0.9rem;
    font-family: 'DM Mono', monospace;
    font-size: 0.85rem;
    margin: 0.2rem;
  }

  /* Demo banner */
  .demo-banner {
    background: #f59e0b18;
    border: 1px solid #f59e0b55;
    border-radius: 10px;
    padding: 0.7rem 1rem;
    font-size: 0.8rem;
    color: #f59e0b;
    margin-top: 0.5rem;
  }

  /* Disclaimer */
  .disclaimer {
    background: #1a1a2e;
    border-left: 3px solid #f59e0b;
    border-radius: 6px;
    padding: 0.7rem 1rem;
    font-size: 0.78rem;
    color: #8a8aaa;
    margin-top: 1rem;
  }

  /* Hide Streamlit default menu & footer on mobile */
  #MainMenu { visibility: hidden; }
  footer    { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# ── Model / Inference helpers ─────────────────────────────────────────────────
MODEL_META = {
    "VGG16":       {"color": "#e84393", "desc": "16-layer deep CNN"},
    "ResNet50V2":  {"color": "#00c9a7", "desc": "50-layer residual network"},
    "InceptionV3": {"color": "#f59e0b", "desc": "Multi-scale Inception modules"},
}

DEMO_RESULTS = {
    "VGG16":       {"label": "Malignant", "confidence": 91.4, "precision": 89.2, "recall": 93.1, "f1": 91.1},
    "ResNet50V2":  {"label": "Malignant", "confidence": 94.7, "precision": 92.8, "recall": 95.3, "f1": 94.0},
    "InceptionV3": {"label": "Benign",    "confidence": 87.3, "precision": 85.6, "recall": 88.9, "f1": 87.2},
}


@st.cache_resource(show_spinner="Loading models…")
def load_models(model_dir: str = "models/saved"):
    """Load all saved Keras models once and cache them."""
    loaded = {}
    try:
        from tensorflow import keras
        for name in MODEL_META:
            for ext in (".keras", ".h5"):
                path = os.path.join(model_dir, f"{name}{ext}")
                if os.path.exists(path):
                    loaded[name] = keras.models.load_model(path)
                    break
    except Exception:
        pass  # TF not available or no weights → demo mode
    return loaded


def preprocess(img: Image.Image, size: int = 128) -> np.ndarray:
    """Resize + normalize PIL image to (1, size, size, 3)."""
    img = img.convert("RGB").resize((size, size), Image.LANCZOS)
    arr = np.array(img, dtype=np.float32) / 255.0
    return np.expand_dims(arr, axis=0)


def run_inference(models: dict, img_array: np.ndarray) -> dict:
    """Run all loaded models. Fall back to demo values if absent."""
    import json
    results = {}
    eval_path = "models/saved/eval_metrics.json"
    eval_metrics = {}
    if os.path.exists(eval_path):
        with open(eval_path) as f:
            eval_metrics = json.load(f)

    for name in MODEL_META:
        if name in models:
            prob = float(models[name].predict(img_array, verbose=0)[0][0])
            label = "Malignant" if prob >= 0.5 else "Benign"
            conf  = prob * 100 if label == "Malignant" else (1 - prob) * 100
            m = eval_metrics.get(name, {})
            results[name] = {
                "label":      label,
                "confidence": round(conf, 1),
                "precision":  m.get("precision", 0.0),
                "recall":     m.get("recall", 0.0),
                "f1":         m.get("f1", 0.0),
                "mode":       "live",
            }
        else:
            results[name] = {**DEMO_RESULTS[name], "mode": "demo"}

    return results


# ── UI ────────────────────────────────────────────────────────────────────────
def render_header():
    st.markdown("""
    <div style="text-align:center; padding: 0.5rem 0 1.2rem;">
      <span style="font-size:2rem;">🔬</span>
      <h1 style="font-size:1.8rem; font-weight:700; margin:0.3rem 0 0.2rem; color:#fff;">
        MammoScan <span style="color:#e84393;">AI</span>
      </h1>
      <p style="color:#5a5a7a; font-size:0.9rem; margin:0;">
        Breast Cancer Detection · VGG16 · ResNet50V2 · InceptionV3
      </p>
    </div>
    """, unsafe_allow_html=True)


def render_result_card(name: str, r: dict):
    color = MODEL_META[name]["color"]
    desc  = MODEL_META[name]["desc"]
    is_mal = r["label"] == "Malignant"
    label_class = "malignant" if is_mal else "benign"
    verdict_bg  = "#e8439318" if is_mal else "#00c9a718"
    verdict_border = "#e8439366" if is_mal else "#00c9a766"
    verdict_text = (
        "⚠ Signs consistent with malignant tissue. Please consult a radiologist."
        if is_mal else
        "✓ No malignant patterns detected. Regular screening still recommended."
    )

    st.markdown(f"""
    <div class="card" style="border-color:{color}44;">
      <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:0.5rem;">
        <span style="font-family:'DM Mono',monospace; font-size:0.8rem; color:{color}; font-weight:600;">{name}</span>
        <span style="font-size:0.75rem; color:#5a5a7a;">{desc}</span>
      </div>

      <div style="display:flex; align-items:center; gap:1.2rem; margin:0.6rem 0;">
        <div>
          <div class="{label_class}">{r['label']}</div>
          <div style="font-size:0.8rem; color:#5a5a7a;">Confidence: <b style="color:{color}">{r['confidence']:.1f}%</b></div>
        </div>
      </div>

      <div>
        <span class="pill">Precision&nbsp;<b style="color:{color}">{r['precision']:.1f}%</b></span>
        <span class="pill">Recall&nbsp;<b style="color:{color}">{r['recall']:.1f}%</b></span>
        <span class="pill">F1&nbsp;<b style="color:{color}">{r['f1']:.1f}%</b></span>
      </div>

      <div style="background:{verdict_bg}; border:1px solid {verdict_border}; border-radius:8px; padding:0.6rem 0.9rem; margin-top:0.8rem; font-size:0.8rem; color:#c0c0d8;">
        {verdict_text}
      </div>

      {'<div class="demo-banner">⚡ DEMO MODE — no trained weights found for this model</div>' if r["mode"] == "demo" else ""}
    </div>
    """, unsafe_allow_html=True)


def render_comparison_table(results: dict):
    st.markdown("#### 📊 Model Comparison")
    cols = st.columns(4)
    cols[0].markdown("**Model**")
    cols[1].markdown("**Label**")
    cols[2].markdown("**Conf.**")
    cols[3].markdown("**F1**")
    for name, r in results.items():
        color = "#e84393" if r["label"] == "Malignant" else "#00c9a7"
        c = st.columns(4)
        c[0].markdown(f"<span style='color:{MODEL_META[name]['color']};font-family:monospace'>{name}</span>", unsafe_allow_html=True)
        c[1].markdown(f"<span style='color:{color}'>{r['label']}</span>", unsafe_allow_html=True)
        c[2].markdown(f"{r['confidence']:.1f}%")
        c[3].markdown(f"{r['f1']:.1f}%")


# ── Main App ──────────────────────────────────────────────────────────────────
def main():
    render_header()

    # Load models (cached)
    models = load_models()
    live_count = len(models)

    if live_count == 0:
        st.info("💡 Running in **demo mode** — place trained `.keras` weights in `models/saved/` for live predictions.", icon="ℹ️")

    # Upload
    st.markdown("### 📤 Upload Image")
    uploaded = st.file_uploader(
        "Drop a mammogram or histology image",
        type=["png", "jpg", "jpeg", "tiff", "bmp"],
        label_visibility="collapsed",
    )

    if uploaded is None:
        st.markdown("""
        <div class="card" style="text-align:center; padding:2rem; color:#5a5a7a;">
          <div style="font-size:2.5rem; margin-bottom:0.5rem;">🩻</div>
          <div>Upload a mammogram or histology image to begin analysis</div>
          <div style="font-size:0.78rem; margin-top:0.4rem; color:#3a3a5a;">PNG · JPG · TIFF · BMP</div>
        </div>
        """, unsafe_allow_html=True)

        # How it works
        with st.expander("ℹ️ How it works"):
            st.markdown("""
            1. **Upload** a mammogram or histology breast image
            2. **Three CNNs** analyze it: VGG16, ResNet50V2, InceptionV3
            3. **Results** show confidence score, precision, recall, and F1 per model
            4. **Compare** all three models side-by-side
            """)
        return

    # Show image
    img = Image.open(uploaded)
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.image(img, caption=uploaded.name, use_container_width=True)

    # Analyze button
    if st.button("🔍  Run CNN Analysis", type="primary", use_container_width=True):
        with st.spinner("Analyzing image with all three models…"):
            arr     = preprocess(img)
            results = run_inference(models, arr)

        st.success("Analysis complete!", icon="✅")
        st.markdown("---")
        st.markdown("### 🧬 Results")

        # Model selector tabs
        tab_names = list(MODEL_META.keys())
        tabs = st.tabs(tab_names)
        for i, name in enumerate(tab_names):
            with tabs[i]:
                render_result_card(name, results[name])

        st.markdown("---")
        render_comparison_table(results)

    # Disclaimer
    st.markdown("""
    <div class="disclaimer">
      ⚠️ <strong>For research and educational purposes only.</strong>
      Results must be reviewed by a qualified radiologist.
      Not intended for clinical diagnosis or treatment decisions.
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
