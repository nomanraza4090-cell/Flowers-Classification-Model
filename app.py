"""
AI Vision — 17-Class Flower Classifier (Grassy Theme)
A production-quality Streamlit application serving a trained CNN
(EfficientNetB2 transfer-learning model) for 17-class flower classification.

Model: models/final_model.keras (fallback: models/flower_saved_model/)
Preprocessing: RGB → Resize 260x260 → float32 (EfficientNetB2 normalizes internally)
Output: softmax over 17 classes
"""

import io
import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from PIL import Image, UnidentifiedImageError

# ----------------------------------------------------------------------------
# CONSTANTS
# ----------------------------------------------------------------------------
APP_DIR = Path(__file__).parent
MODELS_DIR = APP_DIR / "models"

MODEL_CANDIDATES = [
    MODELS_DIR / "final_model.keras",
    MODELS_DIR / "flower_saved_model",
]
LABELS_PATH = MODELS_DIR / "class_names.json"

IMG_SIZE = 260
SUPPORTED_FORMATS = ("jpg", "jpeg", "png", "webp")
CONFIDENCE_THRESHOLD = 0.45
TOP_K = 5

PAGES = ["Dashboard", "Classify Image", "Analytics", "About Model"]

# Classic professional color palette
PALETTE = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
    "#1a55a0", "#e67e22", "#27ae60", "#c0392b", "#8e44ad",
    "#2c3e50", "#f39c12",
]

# ----------------------------------------------------------------------------
# PAGE CONFIG
# ----------------------------------------------------------------------------
st.set_page_config(
    page_title="AI Vision | Flower Classifier",
    page_icon="🌸",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ----------------------------------------------------------------------------
# CUSTOM CSS — Grassy Theme with Flower Chunks
# ----------------------------------------------------------------------------
def inject_css() -> None:
    st.markdown(
        """
        <style>
        /* Import clean font */
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

        /* Reset & base */
        html, body, [class*="css"] { 
            font-family: 'Inter', sans-serif; 
        }

        /* ================================================================
           GRASSY THEME — Flower chunks as background
           ================================================================ */
        
        /* Main background with flower chunks pattern */
        .stApp {
            background: 
                /* Dark grassy gradient base */
                radial-gradient(ellipse at 20% 50%, #1a3a2a 0%, #0d1f16 70%, #0a1812 100%),
                /* Flower chunks pattern — small flower shapes */
                radial-gradient(circle 8px at 10% 15%, rgba(255,182,193,0.15) 0%, transparent 100%),
                radial-gradient(circle 6px at 25% 45%, rgba(255,105,180,0.12) 0%, transparent 100%),
                radial-gradient(circle 10px at 40% 75%, rgba(255,182,193,0.10) 0%, transparent 100%),
                radial-gradient(circle 7px at 55% 20%, rgba(255,20,147,0.08) 0%, transparent 100%),
                radial-gradient(circle 9px at 70% 60%, rgba(255,182,193,0.12) 0%, transparent 100%),
                radial-gradient(circle 5px at 85% 35%, rgba(255,105,180,0.10) 0%, transparent 100%),
                radial-gradient(circle 8px at 92% 80%, rgba(255,182,193,0.08) 0%, transparent 100%),
                /* More flower chunks scattered */
                radial-gradient(circle 6px at 15% 88%, rgba(255,20,147,0.10) 0%, transparent 100%),
                radial-gradient(circle 11px at 48% 92%, rgba(255,182,193,0.08) 0%, transparent 100%),
                radial-gradient(circle 7px at 78% 10%, rgba(255,105,180,0.12) 0%, transparent 100%),
                /* Light green grass blades */
                repeating-linear-gradient(45deg, 
                    rgba(34,139,34,0.03) 0px, 
                    rgba(34,139,34,0.03) 2px,
                    transparent 2px,
                    transparent 8px
                ),
                /* Subtle grass texture */
                repeating-linear-gradient(-45deg,
                    rgba(0,100,0,0.02) 0px,
                    rgba(0,100,0,0.02) 3px,
                    transparent 3px,
                    transparent 12px
                );
            color: #e8f0e8;
        }

        /* Sidebar - semi-transparent dark glass */
        section[data-testid="stSidebar"] {
            background: rgba(13, 31, 22, 0.92);
            backdrop-filter: blur(12px);
            border-right: 1px solid rgba(255, 255, 255, 0.06);
        }

        /* Cards - glassmorphism on grass */
        .card {
            background: rgba(255, 255, 255, 0.04);
            backdrop-filter: blur(8px);
            border-radius: 12px;
            padding: 20px 24px;
            border: 1px solid rgba(255, 255, 255, 0.06);
            box-shadow: 0 4px 20px rgba(0,0,0,0.2);
        }

        /* Metric cards */
        .metric-card {
            background: rgba(255, 255, 255, 0.05);
            backdrop-filter: blur(8px);
            border-radius: 10px;
            padding: 16px 20px;
            border: 1px solid rgba(255, 255, 255, 0.06);
            box-shadow: 0 2px 10px rgba(0,0,0,0.15);
            height: 100%;
        }
        .metric-label {
            font-size: 0.7rem;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            color: #8ab4a8;
            font-weight: 600;
        }
        .metric-value {
            font-size: 1.3rem;
            font-weight: 700;
            color: #e8f0e8;
        }
        .metric-icon { font-size: 1.2rem; margin-bottom: 6px; }

        /* Headings */
        .page-title {
            font-size: 1.8rem;
            font-weight: 700;
            color: #e8f0e8;
            margin-bottom: 4px;
            text-shadow: 0 2px 10px rgba(0,0,0,0.3);
        }
        .page-subtitle {
            color: #8ab4a8;
            font-size: 0.95rem;
            margin-bottom: 20px;
        }
        .section-title {
            font-size: 1.1rem;
            font-weight: 600;
            color: #e8f0e8;
            margin: 8px 0 4px 0;
        }
        .section-caption {
            color: #8ab4a8;
            font-size: 0.85rem;
            margin-bottom: 16px;
        }

        /* Buttons - grassy green */
        .stButton > button {
            background: linear-gradient(135deg, #2d6a4f 0%, #1a4a3a 100%);
            color: white;
            border: 1px solid rgba(255,255,255,0.08);
            border-radius: 8px;
            padding: 0.5rem 1.2rem;
            font-weight: 600;
            transition: all 0.2s ease;
            box-shadow: 0 2px 8px rgba(0,0,0,0.2);
        }
        .stButton > button:hover {
            background: linear-gradient(135deg, #3a8a6a 0%, #2a5a4a 100%);
            box-shadow: 0 4px 16px rgba(45, 106, 79, 0.4);
            transform: translateY(-1px);
        }
        .stButton > button:active {
            transform: scale(0.97);
        }

        /* Result card - highlight */
        .result-card {
            background: linear-gradient(135deg, rgba(45, 106, 79, 0.2), rgba(26, 74, 58, 0.3));
            border: 1px solid rgba(45, 106, 79, 0.3);
            border-radius: 12px;
            padding: 24px 28px;
            text-align: center;
            backdrop-filter: blur(8px);
        }
        .result-label {
            color: #8ab4a8;
            font-size: 0.75rem;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            font-weight: 600;
        }
        .result-class {
            font-size: 2rem;
            font-weight: 700;
            margin: 4px 0;
            color: #e8f0e8;
            text-transform: capitalize;
            text-shadow: 0 2px 10px rgba(0,0,0,0.2);
        }
        .result-confidence {
            font-size: 1rem;
            color: #5c9a7a;
            font-weight: 600;
        }
        .result-lowconf {
            font-size: 0.8rem;
            color: #fbbf24;
            font-weight: 600;
            margin-top: 6px;
        }

        /* Upload zone - grass themed */
        div[data-testid="stFileUploaderDropzone"] {
            background: rgba(255, 255, 255, 0.03);
            border: 1.5px dashed rgba(255, 255, 255, 0.12);
            border-radius: 12px;
            backdrop-filter: blur(4px);
        }

        /* Dataframes */
        div[data-testid="stDataFrame"] {
            border-radius: 8px;
            overflow: hidden;
            border: 1px solid rgba(255, 255, 255, 0.06);
        }

        /* Empty state */
        .empty-state {
            text-align: center;
            padding: 40px 20px;
            color: #8ab4a8;
            border: 1px dashed rgba(255, 255, 255, 0.08);
            border-radius: 12px;
            background: rgba(255, 255, 255, 0.02);
            backdrop-filter: blur(4px);
        }

        .footer-note {
            text-align: center;
            color: #5a7a6a;
            font-size: 0.75rem;
            margin-top: 30px;
            padding-top: 16px;
            border-top: 1px solid rgba(255, 255, 255, 0.05);
        }

        /* Status pills */
        .status-pill {
            display: inline-flex;
            align-items: center;
            gap: 6px;
            font-size: 0.75rem;
            font-weight: 600;
            padding: 4px 12px;
            border-radius: 999px;
        }
        .status-online {
            background: rgba(34, 197, 94, 0.15);
            border: 1px solid rgba(34, 197, 94, 0.25);
            color: #4ade80;
        }
        .status-offline {
            background: rgba(239, 68, 68, 0.15);
            border: 1px solid rgba(239, 68, 68, 0.25);
            color: #f87171;
        }
        .status-dot {
            width: 6px; height: 6px; border-radius: 50%; background: currentColor;
        }

        /* Additional flower decorations */
        .flower-deco {
            display: inline-block;
            animation: float 6s ease-in-out infinite;
        }
        @keyframes float {
            0%, 100% { transform: translateY(0px) rotate(0deg); }
            50% { transform: translateY(-8px) rotate(5deg); }
        }

        /* Override streamlit elements */
        .stSelectbox, .stTextInput, .stNumberInput {
            background: rgba(255,255,255,0.03);
            border-radius: 8px;
            border: 1px solid rgba(255,255,255,0.06);
        }

        /* Plotly charts - dark theme compatible */
        .js-plotly-plot .plotly .main-svg {
            border-radius: 8px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

# ----------------------------------------------------------------------------
# MODEL / LABEL LOADING
# ----------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_model():
    """Load the trained Keras model once."""
    import tensorflow as tf

    last_error = None
    for path in MODEL_CANDIDATES:
        if path.exists():
            try:
                model = tf.keras.models.load_model(path)
                return model, path
            except Exception as exc:
                last_error = exc
                continue
    if last_error:
        raise RuntimeError(f"Found model file(s) but failed to load: {last_error}")
    raise FileNotFoundError(
        "No model file found. Expected one of: "
        + ", ".join(str(p) for p in MODEL_CANDIDATES)
    )

@st.cache_data(show_spinner=False)
def load_class_labels():
    """Load the 17-class ordering."""
    if not LABELS_PATH.exists():
        raise FileNotFoundError(f"Missing class label file: {LABELS_PATH}")
    with open(LABELS_PATH, "r") as f:
        classes = json.load(f)
    if not isinstance(classes, list) or len(classes) < 2:
        raise ValueError("class_names.json does not contain a valid class list.")
    return classes

def get_model_metadata(model):
    """Pull real architecture facts from the loaded model."""
    try:
        input_shape = model.input_shape
    except Exception:
        input_shape = None
    try:
        output_shape = model.output_shape
    except Exception:
        output_shape = None
    try:
        num_layers = len(model.layers)
    except Exception:
        num_layers = None
    try:
        total_params = int(model.count_params())
    except Exception:
        total_params = None
    return {
        "input_shape": input_shape,
        "output_shape": output_shape,
        "num_layers": num_layers,
        "total_params": total_params,
    }

# ----------------------------------------------------------------------------
# INFERENCE PIPELINE
# ----------------------------------------------------------------------------
def preprocess_image(image: Image.Image) -> np.ndarray:
    """Reproduce the exact training-time preprocessing."""
    img = image.convert("RGB").resize((IMG_SIZE, IMG_SIZE))
    arr = np.asarray(img, dtype=np.float32)
    arr = np.expand_dims(arr, axis=0)
    return arr

def predict_image(model, arr: np.ndarray, classes: list) -> dict:
    """Run inference and interpret the softmax output correctly."""
    probs = model.predict(arr, verbose=0)[0]
    probs = np.asarray(probs).flatten()

    if len(probs) != len(classes):
        raise ValueError(
            f"Model output size ({len(probs)}) does not match number of classes "
            f"({len(classes)}) in class_names.json."
        )

    pred_idx = int(np.argmax(probs))
    pred_class = classes[pred_idx]
    confidence = float(probs[pred_idx])
    probabilities = {classes[i]: float(probs[i]) * 100.0 for i in range(len(classes))}

    ranked = sorted(probabilities.items(), key=lambda kv: kv[1], reverse=True)
    top_k = ranked[:TOP_K]

    return {
        "predicted_class": pred_class,
        "confidence": confidence * 100.0,
        "probabilities": probabilities,
        "top_k": top_k,
        "low_confidence": confidence < CONFIDENCE_THRESHOLD,
    }

# ----------------------------------------------------------------------------
# SESSION STATE
# ----------------------------------------------------------------------------
def init_session_state():
    if "history" not in st.session_state:
        st.session_state.history = []
    if "page" not in st.session_state:
        st.session_state.page = "Dashboard"
    if "last_result" not in st.session_state:
        st.session_state.last_result = None

def record_prediction(filename: str, result: dict, model_label: str):
    st.session_state.history.append(
        {
            "Time": datetime.now().strftime("%H:%M:%S"),
            "Image": filename,
            "Prediction": result["predicted_class"],
            "Confidence": f"{result['confidence']:.2f}%",
            "Low Confidence": "Yes" if result["low_confidence"] else "No",
            "Model": model_label,
        }
    )

# ----------------------------------------------------------------------------
# SHARED UI HELPERS
# ----------------------------------------------------------------------------
def metric_card(col, icon: str, label: str, value: str):
    with col:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-icon">{icon}</div>
                <div class="metric-label">{label}</div>
                <div class="metric-value">{value}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

def color_for_class(class_name: str, classes: list) -> str:
    try:
        idx = classes.index(class_name)
    except ValueError:
        idx = 0
    return PALETTE[idx % len(PALETTE)]

def top_k_bar_chart(top_k: list, classes: list):
    labels = [c.replace("_", " ").title() for c, _ in top_k]
    values = [v for _, v in top_k]
    colors = [color_for_class(c, classes) for c, _ in top_k]

    fig = go.Figure(
        go.Bar(
            x=values,
            y=labels,
            orientation="h",
            marker=dict(color=colors, line=dict(width=0)),
            text=[f"{v:.2f}%" for v in values],
            textposition="outside",
            hovertemplate="%{y}: %{x:.2f}%<extra></extra>",
        )
    )
    fig.update_layout(
        height=280,
        margin=dict(l=10, r=40, t=10, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#e8f0e8", family="Inter"),
        xaxis=dict(range=[0, 100], showgrid=True, gridcolor="rgba(255,255,255,0.06)", ticksuffix="%", color="#8ab4a8"),
        yaxis=dict(showgrid=False, color="#e8f0e8", autorange="reversed"),
        showlegend=False,
    )
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

def sidebar_nav(model_loaded: bool, model_label: str, num_classes: int):
    with st.sidebar:
        st.markdown(
            """
            <div style="font-size:1.2rem;font-weight:700;color:#e8f0e8;margin-bottom:2px;">
                🌸 AI Vision
            </div>
            <div style="color:#8ab4a8;font-size:0.75rem;margin-bottom:20px;">
                Flower Classifier
            </div>
            """,
            unsafe_allow_html=True,
        )

        icons = {"Dashboard": "🏠", "Classify Image": "🔍", "Analytics": "📊", "About Model": "ℹ️"}
        for page in PAGES:
            label = f"{icons[page]} {page}"
            btn_type = "primary" if st.session_state.page == page else "secondary"
            if st.button(label, key=f"nav_{page}", use_container_width=True, type=btn_type):
                st.session_state.page = page

        st.markdown("---")
        st.markdown("<div class='metric-label'>Model Information</div>", unsafe_allow_html=True)
        st.markdown(
            f"""
            <div style="font-size:0.8rem; color:#8ab4a8; line-height:1.8;">
            CNN (EfficientNetB2 Transfer Learning)<br>
            {num_classes}-Class Flower Classification<br>
            TensorFlow / Keras
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("<div style='margin-top:14px;'></div>", unsafe_allow_html=True)
        if model_loaded:
            st.markdown(
                "<span class='status-pill status-online'><span class='status-dot'></span>"
                "Model Loaded</span>",
                unsafe_allow_html=True,
            )
            st.markdown(
                f"<div style='font-size:0.7rem;color:#5a7a6a;margin-top:6px;'>{model_label}</div>",
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                "<span class='status-pill status-offline'><span class='status-dot'></span>"
                "Model Unavailable</span>",
                unsafe_allow_html=True,
            )

        st.markdown(
            "<div class='footer-note' style='margin-top:20px;'>AI Vision v1.0<br>"
            "Built with Streamlit &amp; TensorFlow</div>",
            unsafe_allow_html=True,
        )

# ----------------------------------------------------------------------------
# PAGE: DASHBOARD
# ----------------------------------------------------------------------------
def render_dashboard(model_loaded: bool, classes: list, model_label: str):
    st.markdown("<div class='page-title'>🌸 AI Vision</div>", unsafe_allow_html=True)
    st.markdown(
        "<div class='page-subtitle'>17-Class Flower Species Classification</div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<div style='color:#8ab4a8;font-size:0.95rem;margin-bottom:20px;'>"
        "Upload a flower photo and the AI will identify the species using an EfficientNetB2 "
        "convolutional neural network trained on a labeled flower image dataset.</div>",
        unsafe_allow_html=True,
    )

    cols = st.columns(4)
    metric_card(cols[0], "🧠", "Model Architecture", "EfficientNetB2")
    metric_card(cols[1], "🏷️", "Number of Classes", str(len(classes)))
    metric_card(cols[2], "⚡", "Prediction Mode", "Single Image")
    metric_card(
        cols[3], "✅" if model_loaded else "⚠️", "Model Status", "Loaded" if model_loaded else "Unavailable"
    )

    st.write("")
    left, right = st.columns([1.3, 1])
    with left:
        st.markdown("<div class='section-title'>🚀 Quick Start</div>", unsafe_allow_html=True)
        st.markdown(
            "<div class='section-caption'>Upload an image to get started with AI classification.</div>",
            unsafe_allow_html=True,
        )
        st.markdown(
            """
            <div class="card">
                <div style="font-weight:600;margin-bottom:8px;color:#e8f0e8;">Pipeline Overview</div>
                <div style="color:#8ab4a8;font-size:0.88rem;line-height:1.7;">
                Upload Image → Preprocess (Resize 260×260) → EfficientNetB2 Backbone → 
                Dense Head → Softmax (17 classes) → Class + Confidence Score
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if st.button("🔍 Go to Classify Image", use_container_width=True):
            st.session_state.page = "Classify Image"
            st.rerun()

    with right:
        st.markdown("<div class='section-title'>System Status</div>", unsafe_allow_html=True)
        status_text = "Model Loaded" if model_loaded else "Model Unavailable"
        status_class = "status-online" if model_loaded else "status-offline"
        class_preview = ", ".join(c.replace("_", " ").title() for c in classes[:6])
        st.markdown(
            f"""
            <div class="card">
                <div style="display:flex;justify-content:space-between;margin-bottom:8px;">
                    <span style="color:#8ab4a8;">Model Type</span>
                    <span style="font-weight:500;color:#e8f0e8;">CNN (Transfer Learning)</span>
                </div>
                <div style="display:flex;justify-content:space-between;margin-bottom:8px;">
                    <span style="color:#8ab4a8;">Task</span>
                    <span style="font-weight:500;color:#e8f0e8;">17-Class Image Classification</span>
                </div>
                <div style="display:flex;justify-content:space-between;margin-bottom:8px;">
                    <span style="color:#8ab4a8;">Sample Classes</span>
                    <span style="text-align:right;font-weight:500;color:#e8f0e8;">{class_preview}…</span>
                </div>
                <div style="display:flex;justify-content:space-between;">
                    <span style="color:#8ab4a8;">Status</span>
                    <span class="status-pill {status_class}"><span class="status-dot"></span>{status_text}</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.write("")
    st.markdown("<div class='section-title'>🌼 Recognized Species</div>", unsafe_allow_html=True)
    chip_html = "".join(
        f"<span style='display:inline-block;margin:4px;padding:5px 12px;border-radius:999px;"
        f"background:rgba(255,255,255,0.04);border:1px solid rgba(255,255,255,0.06);"
        f"color:#e8f0e8;font-size:0.8rem;'>{c.replace('_',' ').title()}</span>"
        for c in classes
    )
    st.markdown(f"<div class='card'>{chip_html}</div>", unsafe_allow_html=True)

# ----------------------------------------------------------------------------
# PAGE: CLASSIFY IMAGE
# ----------------------------------------------------------------------------
def render_classifier(model, classes: list, model_label: str):
    st.markdown("<div class='page-title'>🔍 Classify Image</div>", unsafe_allow_html=True)
    st.markdown(
        "<div class='page-subtitle'>Upload a flower photo for AI-powered species identification.</div>",
        unsafe_allow_html=True,
    )

    uploaded_file = st.file_uploader(
        "Upload an Image",
        type=list(SUPPORTED_FORMATS),
        label_visibility="collapsed",
    )

    if uploaded_file is None:
        st.markdown(
            "<div class='empty-state'>📤 Upload an image to begin AI classification.</div>",
            unsafe_allow_html=True,
        )
        return

    try:
        raw_bytes = uploaded_file.getvalue()
        image = Image.open(io.BytesIO(raw_bytes))
        image.verify()
        image = Image.open(io.BytesIO(raw_bytes))
    except (UnidentifiedImageError, OSError):
        st.error("This file could not be read as a valid image. Please upload a JPG, PNG, or WEBP file.")
        return

    img_col, _ = st.columns([1, 1.4])
    with img_col:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.image(image, width=340)
        st.markdown(
            f"<div style='color:#8ab4a8;font-size:0.8rem;margin-top:6px;text-align:center;'>"
            f"{uploaded_file.name} · {image.size[0]}×{image.size[1]}px</div>",
            unsafe_allow_html=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)
        st.write("")
        analyze_clicked = st.button("🧠 Analyze Image", use_container_width=True)

    if not analyze_clicked:
        return

    if model is None:
        st.error("The model is not currently loaded. Please check the sidebar for details.")
        return

    with st.spinner("🔍 Analyzing image..."):
        try:
            arr = preprocess_image(image)
            start = time.time()
            result = predict_image(model, arr, classes)
            elapsed_ms = (time.time() - start) * 1000
        except Exception as exc:
            st.error("Something went wrong while analyzing this image. Please try a different file.")
            with st.expander("Technical details"):
                st.code(str(exc))
            return

    record_prediction(uploaded_file.name, result, model_label)
    st.session_state.last_result = result

    st.write("")
    st.markdown("<div class='section-title'>📊 Prediction Results</div>", unsafe_allow_html=True)

    pred_class = result["predicted_class"]
    confidence = result["confidence"]
    display_name = pred_class.replace("_", " ").title()

    low_conf_html = (
        f"<div class='result-lowconf'>⚠️ Low confidence — below the "
        f"{CONFIDENCE_THRESHOLD*100:.0f}% threshold</div>"
        if result["low_confidence"]
        else ""
    )

    st.markdown(
        f"""
        <div class="result-card">
            <div class="result-label">Predicted Species</div>
            <div class="result-class">🌸 {display_name}</div>
            <div class="result-confidence">Confidence: {confidence:.2f}%</div>
            {low_conf_html}
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.write("")
    detail_cols = st.columns(4)
    metric_card(detail_cols[0], "🏷️", "Predicted Class", display_name)
    metric_card(detail_cols[1], "📈", "Confidence", f"{confidence:.2f}%")
    metric_card(detail_cols[2], "🧠", "Model", model_label)
    metric_card(detail_cols[3], "🖼️", "Image Size", f"{image.size[0]}×{image.size[1]}px")

    st.write("")
    chart_col, table_col = st.columns([1.3, 1])
    with chart_col:
        st.markdown(f"<div class='section-title'>Top {TOP_K} Predictions</div>", unsafe_allow_html=True)
        top_k_bar_chart(result["top_k"], classes)
    with table_col:
        st.markdown("<div class='section-title'>Full Probability Distribution</div>", unsafe_allow_html=True)
        prob_df = pd.DataFrame(
            [
                {"Class": k.replace("_", " ").title(), "Probability": f"{v:.2f}%"}
                for k, v in sorted(result["probabilities"].items(), key=lambda kv: kv[1], reverse=True)
            ]
        )
        st.dataframe(prob_df, hide_index=True, use_container_width=True, height=280)
        st.markdown(
            f"<div style='color:#5a7a6a;font-size:0.75rem;margin-top:4px;'>"
            f"Processed in {elapsed_ms:.0f} ms</div>",
            unsafe_allow_html=True,
        )

    st.write("")
    st.markdown("<div class='section-title'>📋 Prediction History</div>", unsafe_allow_html=True)
    hist_df = pd.DataFrame(st.session_state.history[::-1])
    st.dataframe(hist_df, hide_index=True, use_container_width=True)
    if st.button("🗑️ Clear History"):
        st.session_state.history = []
        st.rerun()

# ----------------------------------------------------------------------------
# PAGE: ANALYTICS
# ----------------------------------------------------------------------------
def render_analytics(classes: list):
    st.markdown("<div class='page-title'>📊 Analytics</div>", unsafe_allow_html=True)
    st.markdown(
        "<div class='page-subtitle'>Session statistics from all predictions made.</div>",
        unsafe_allow_html=True,
    )

    history = st.session_state.history
    if not history:
        st.markdown(
            "<div class='empty-state'>📊 No predictions yet. Classify an image to "
            "generate analytics.</div>",
            unsafe_allow_html=True,
        )
        return

    df = pd.DataFrame(history)
    df["ConfidenceValue"] = df["Confidence"].str.rstrip("%").astype(float)

    total = len(df)
    counts = df["Prediction"].value_counts()
    avg_conf = df["ConfidenceValue"].mean()
    max_conf = df["ConfidenceValue"].max()
    low_conf_rate = (df["Low Confidence"] == "Yes").mean() * 100
    most_common = counts.index[0] if len(counts) else "—"

    cols = st.columns(5)
    metric_card(cols[0], "🔢", "Total Predictions", str(total))
    metric_card(cols[1], "🌼", "Most Predicted", most_common.replace("_", " ").title())
    metric_card(cols[2], "📈", "Average Confidence", f"{avg_conf:.2f}%")
    metric_card(cols[3], "🏆", "Highest Confidence", f"{max_conf:.2f}%")
    metric_card(cols[4], "⚠️", "Low-Confidence Rate", f"{low_conf_rate:.1f}%")

    st.write("")
    chart_col1, chart_col2 = st.columns(2)

    with chart_col1:
        st.markdown("<div class='section-title'>Prediction Distribution</div>", unsafe_allow_html=True)
        labels = [c.replace("_", " ").title() for c in counts.index]
        colors = [color_for_class(c, classes) for c in counts.index]
        fig = go.Figure(
            go.Pie(
                labels=labels,
                values=counts.values,
                hole=0.55,
                marker=dict(colors=colors, line=dict(color="#0d1f16", width=2)),
                textinfo="label+percent",
            )
        )
        fig.update_layout(
            margin=dict(l=10, r=10, t=10, b=10),
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#e8f0e8", family="Inter"),
            showlegend=False,
            height=340,
        )
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    with chart_col2:
        st.markdown("<div class='section-title'>Confidence Distribution</div>", unsafe_allow_html=True)
        fig2 = go.Figure(
            go.Histogram(
                x=df["ConfidenceValue"],
                nbinsx=10,
                marker=dict(color="#2d6a4f"),
            )
        )
        fig2.update_layout(
            margin=dict(l=10, r=10, t=10, b=10),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#e8f0e8", family="Inter"),
            xaxis=dict(title="Confidence (%)", color="#8ab4a8", showgrid=True, gridcolor="rgba(255,255,255,0.06)"),
            yaxis=dict(title="Count", color="#8ab4a8", showgrid=True, gridcolor="rgba(255,255,255,0.06)"),
            height=340,
        )
        st.plotly_chart(fig2, use_container_width=True, config={"displayModeBar": False})

    st.write("")
    st.markdown("<div class='section-title'>Session History</div>", unsafe_allow_html=True)
    st.dataframe(df.drop(columns=["ConfidenceValue"])[::-1], hide_index=True, use_container_width=True)

# ----------------------------------------------------------------------------
# PAGE: ABOUT MODEL
# ----------------------------------------------------------------------------
def render_about(model, classes: list, model_label: str):
    st.markdown("<div class='page-title'>ℹ️ About Model</div>", unsafe_allow_html=True)
    st.markdown(
        "<div class='page-subtitle'>Technical details about the AI model powering this application.</div>",
        unsafe_allow_html=True,
    )

    st.markdown(
        f"""
        <div class="card">
            <div style="font-weight:600;margin-bottom:8px;color:#e8f0e8;">What is this application?</div>
            <div style="color:#8ab4a8;line-height:1.7;font-size:0.92rem;">
            This application is powered by a Convolutional Neural Network built on top of
            <b style="color:#e8f0e8;">EfficientNetB2</b>, a proven image-recognition backbone pretrained on ImageNet.
            The base network was frozen and used as a fixed feature extractor to train a new
            classification head, then partially unfrozen and fine-tuned end-to-end — a
            technique called transfer learning that lets a large, general-purpose vision model
            specialize in recognizing {len(classes)} specific flower species from relatively
            few training images.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.write("")
    st.markdown(
        f"""
        <div class="card">
            <div style="font-weight:600;margin-bottom:8px;color:#e8f0e8;">Model Pipeline</div>
            <div style="color:#8ab4a8;line-height:1.7;font-size:0.92rem;">
            Image → Preprocessing (RGB, resize {IMG_SIZE}×{IMG_SIZE}) → EfficientNetB2 Backbone
            → Global Average Pooling → Batch Normalization + Dropout → Dense(256, swish)
            → Dense({len(classes)}, softmax) → Predicted Class + Confidence
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.write("")
    st.markdown("<div class='section-title'>Architecture & Training Details</div>", unsafe_allow_html=True)

    if model is not None:
        meta = get_model_metadata(model)
        cols = st.columns(4)
        metric_card(cols[0], "📥", "Input Shape", str(meta["input_shape"]))
        metric_card(cols[1], "📤", "Output Shape", str(meta["output_shape"]))
        metric_card(cols[2], "🧱", "Number of Layers", str(meta["num_layers"]))
        total_params = meta["total_params"]
        metric_card(cols[3], "⚙️", "Total Parameters", f"{total_params:,}" if total_params else "N/A")

        with st.expander("View full model architecture (model.summary())"):
            stream = io.StringIO()
            model.summary(print_fn=lambda x: stream.write(x + "\n"))
            st.code(stream.getvalue(), language="text")
    else:
        st.info("Load the model to view live architecture details.")

    st.write("")
    st.markdown(
        f"""
        <div class="card">
            <div style="font-weight:600;margin-bottom:8px;color:#e8f0e8;">Preprocessing (matches training)</div>
            <div style="color:#8ab4a8;line-height:1.9;font-size:0.88rem;">
            • Convert image to RGB<br>
            • Resize to {IMG_SIZE}×{IMG_SIZE} pixels<br>
            • Cast to float32 (no manual division by 255 — EfficientNetB2 normalizes internally)<br>
            • Output activation: Softmax over {len(classes)} classes<br>
            • Low-confidence threshold: predictions below {CONFIDENCE_THRESHOLD*100:.0f}%
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.write("")
    st.markdown(f"<div class='section-title'>Class Labels ({len(classes)} total)</div>", unsafe_allow_html=True)
    labels_df = pd.DataFrame(
        {"Index": range(len(classes)), "Class": [c.replace("_", " ").title() for c in classes]}
    )
    st.dataframe(labels_df, hide_index=True, use_container_width=True)

# ----------------------------------------------------------------------------
# MAIN
# ----------------------------------------------------------------------------
def main():
    inject_css()
    init_session_state()

    model, model_error = None, None
    model_path = None
    try:
        model, model_path = load_model()
    except Exception as exc:
        model_error = str(exc)

    classes, labels_error = None, None
    try:
        classes = load_class_labels()
    except Exception as exc:
        labels_error = str(exc)
        classes = [f"class_{i}" for i in range(17)]

    model_loaded = model is not None and labels_error is None
    model_label = model_path.name if model_path else "Unavailable"

    sidebar_nav(model_loaded, model_label, len(classes))

    if model_error:
        st.error(f"Model could not be loaded: {model_error}")
    if labels_error:
        st.error(f"Class labels could not be loaded: {labels_error}")

    page = st.session_state.page
    if page == "Dashboard":
        render_dashboard(model_loaded, classes, model_label)
    elif page == "Classify Image":
        render_classifier(model, classes, model_label)
    elif page == "Analytics":
        render_analytics(classes)
    elif page == "About Model":
        render_about(model, classes, model_label)

    st.markdown(
        "<div class='footer-note'>AI Vision — Flower Classifier · Built with "
        "Streamlit, TensorFlow &amp; Plotly</div>",
        unsafe_allow_html=True,
    )

if __name__ == "__main__":
    main()
