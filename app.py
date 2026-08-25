"""
AI Vision — 17-Class Flower Classifier (Dark Professional)
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

# Elegant professional color palette for dark theme
PALETTE = [
    "#6c5ce7", "#0984e3", "#00b894", "#fdcb6e", "#e17055",
    "#fd79a8", "#00cec9", "#a29bfe", "#ff7675", "#74b9ff",
    "#55efc4", "#f8a5c2", "#ffd93d", "#6c5ce7", "#0984e3",
    "#00b894", "#e17055",
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
# CUSTOM CSS — Dark Professional Theme
# ----------------------------------------------------------------------------
def inject_css() -> None:
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

        html, body, [class*="css"] { 
            font-family: 'Inter', sans-serif; 
        }

        /* ================================================================
           DARK PROFESSIONAL THEME
           ================================================================ */
        
        .stApp {
            background-color: #0a0a1a;
            background-image:
                linear-gradient(145deg, #0a0a1a 0%, #14142e 40%, #1a1a3e 100%),
                url("data:image/svg+xml,%3Csvg%20xmlns%3D%22http%3A//www.w3.org/2000/svg%22%20width%3D%22130%22%20height%3D%22130%22%20viewBox%3D%220%200%20130%20130%22%3E%0A%3Cg%20fill%3D%22none%22%20stroke%3D%22%23a29bfe%22%20stroke-opacity%3D%220.07%22%20stroke-width%3D%221.6%22%3E%0A%3Cg%20transform%3D%22translate%2865%2C65%29%22%3E%0A%3Cellipse%20cx%3D%220%22%20cy%3D%22-22%22%20rx%3D%2210%22%20ry%3D%2216%22/%3E%0A%3Cellipse%20cx%3D%220%22%20cy%3D%2222%22%20rx%3D%2210%22%20ry%3D%2216%22/%3E%0A%3Cellipse%20cx%3D%22-22%22%20cy%3D%220%22%20rx%3D%2216%22%20ry%3D%2210%22/%3E%0A%3Cellipse%20cx%3D%2222%22%20cy%3D%220%22%20rx%3D%2216%22%20ry%3D%2210%22/%3E%0A%3Cellipse%20cx%3D%22-15%22%20cy%3D%22-15%22%20rx%3D%2210%22%20ry%3D%2216%22%20transform%3D%22rotate%2845%20-15%20-15%29%22/%3E%0A%3Cellipse%20cx%3D%2215%22%20cy%3D%22-15%22%20rx%3D%2210%22%20ry%3D%2216%22%20transform%3D%22rotate%28-45%2015%20-15%29%22/%3E%0A%3Cellipse%20cx%3D%22-15%22%20cy%3D%2215%22%20rx%3D%2210%22%20ry%3D%2216%22%20transform%3D%22rotate%28-45%20-15%2015%29%22/%3E%0A%3Cellipse%20cx%3D%2215%22%20cy%3D%2215%22%20rx%3D%2210%22%20ry%3D%2216%22%20transform%3D%22rotate%2845%2015%2015%29%22/%3E%0A%3Ccircle%20cx%3D%220%22%20cy%3D%220%22%20r%3D%229%22/%3E%0A%3C/g%3E%0A%3Cline%20x1%3D%2265%22%20y1%3D%2295%22%20x2%3D%2265%22%20y2%3D%22128%22%20stroke-width%3D%221.4%22/%3E%0A%3C/g%3E%0A%3C/svg%3E");
            background-repeat: no-repeat, repeat;
            background-size: cover, 190px 190px;
            background-attachment: fixed, fixed;
            color: #e8e8e8;
        }

        /* Hide menu and footer */
        #MainMenu, footer, header { visibility: hidden; }

        /* Sidebar - dark glass */
        section[data-testid="stSidebar"] {
            background: rgba(10, 10, 26, 0.92);
            backdrop-filter: blur(12px);
            border-right: 1px solid rgba(108, 92, 231, 0.15);
        }

        /* Cards - glassmorphism */
        .card {
            background: rgba(255, 255, 255, 0.04);
            backdrop-filter: blur(8px);
            border: 1px solid rgba(108, 92, 231, 0.12);
            border-radius: 14px;
            padding: 20px 24px;
            box-shadow: 0 4px 24px rgba(0,0,0,0.3);
            transition: all 0.3s ease;
        }
        .card:hover {
            border-color: rgba(108, 92, 231, 0.25);
            box-shadow: 0 6px 32px rgba(108, 92, 231, 0.08);
        }

        /* Metric cards - modern stats */
        .metric-card {
            background: rgba(255, 255, 255, 0.04);
            backdrop-filter: blur(8px);
            border: 1px solid rgba(108, 92, 231, 0.10);
            border-radius: 12px;
            padding: 18px 20px;
            height: 100%;
            transition: all 0.3s ease;
            text-align: center;
        }
        .metric-card:hover {
            background: rgba(108, 92, 231, 0.08);
            border-color: rgba(108, 92, 231, 0.25);
            transform: translateY(-2px);
            box-shadow: 0 8px 24px rgba(108, 92, 231, 0.12);
        }
        .metric-icon { font-size: 1.8rem; margin-bottom: 6px; }
        .metric-label {
            font-size: 0.7rem;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            color: #8a8ab5;
            font-weight: 600;
        }
        .metric-value {
            font-size: 1.5rem;
            font-weight: 700;
            background: linear-gradient(135deg, #6c5ce7, #0984e3);
            -webkit-background-clip: text;
            background-clip: text;
            -webkit-text-fill-color: transparent;
        }

        /* Headings */
        .page-title {
            font-size: 2.2rem;
            font-weight: 800;
            color: #e8e8e8;
            margin-bottom: 2px;
            background: linear-gradient(90deg, #e8e8e8, #a29bfe);
            -webkit-background-clip: text;
            background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        .page-subtitle {
            color: #8a8ab5;
            font-size: 0.95rem;
            margin-bottom: 24px;
            font-weight: 300;
        }
        .section-title {
            font-size: 1.1rem;
            font-weight: 600;
            color: #e8e8e8;
            margin: 8px 0 4px 0;
        }
        .section-caption {
            color: #8a8ab5;
            font-size: 0.85rem;
            margin-bottom: 16px;
        }

        /* Buttons - gradient */
        .stButton > button {
            background: linear-gradient(135deg, #6c5ce7, #0984e3);
            color: white;
            border: none;
            border-radius: 10px;
            padding: 0.6rem 1.4rem;
            font-weight: 600;
            transition: all 0.3s ease;
            box-shadow: 0 4px 16px rgba(108, 92, 231, 0.25);
        }
        .stButton > button:hover {
            box-shadow: 0 6px 24px rgba(108, 92, 231, 0.4);
            transform: translateY(-2px);
            filter: brightness(1.1);
        }
        .stButton > button:active {
            transform: scale(0.97);
        }

        /* Status pills */
        .status-pill {
            display: inline-flex;
            align-items: center;
            gap: 6px;
            font-size: 0.75rem;
            font-weight: 600;
            padding: 4px 14px;
            border-radius: 999px;
        }
        .status-online {
            background: rgba(0, 184, 148, 0.12);
            border: 1px solid rgba(0, 184, 148, 0.25);
            color: #55efc4;
        }
        .status-offline {
            background: rgba(225, 112, 85, 0.12);
            border: 1px solid rgba(225, 112, 85, 0.25);
            color: #ff7675;
        }
        .status-dot {
            width: 7px; height: 7px; border-radius: 50%; background: currentColor;
            animation: pulse 2s infinite;
        }
        @keyframes pulse {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.4; }
        }

        /* Upload zone */
        div[data-testid="stFileUploaderDropzone"] {
            background: rgba(255, 255, 255, 0.02);
            border: 1.5px dashed rgba(108, 92, 231, 0.2);
            border-radius: 14px;
        }

        /* Empty state */
        .empty-state {
            text-align: center;
            padding: 50px 20px;
            color: #8a8ab5;
            border: 1px dashed rgba(108, 92, 231, 0.12);
            border-radius: 14px;
            background: rgba(255, 255, 255, 0.02);
        }

        .footer-note {
            text-align: center;
            color: #5a5a8a;
            font-size: 0.75rem;
            margin-top: 30px;
            padding-top: 16px;
            border-top: 1px solid rgba(108, 92, 231, 0.08);
        }

        /* Scrollbar */
        ::-webkit-scrollbar { width: 6px; }
        ::-webkit-scrollbar-track { background: #0a0a1a; }
        ::-webkit-scrollbar-thumb { background: rgba(108, 92, 231, 0.3); border-radius: 10px; }
        ::-webkit-scrollbar-thumb:hover { background: rgba(108, 92, 231, 0.5); }

        /* Result card - premium glow */
        .result-card {
            background: linear-gradient(135deg, rgba(108, 92, 231, 0.10), rgba(9, 132, 227, 0.05));
            border: 1px solid rgba(108, 92, 231, 0.20);
            border-radius: 16px;
            padding: 30px 32px;
            text-align: center;
            box-shadow: 0 8px 40px rgba(108, 92, 231, 0.06);
        }
        .result-label {
            color: #8a8ab5;
            font-size: 0.75rem;
            text-transform: uppercase;
            letter-spacing: 0.1em;
            font-weight: 600;
        }
        .result-class {
            font-size: 2.4rem;
            font-weight: 700;
            margin: 6px 0;
            background: linear-gradient(135deg, #6c5ce7, #a29bfe);
            -webkit-background-clip: text;
            background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        .result-confidence {
            font-size: 1.1rem;
            color: #74b9ff;
            font-weight: 600;
        }
        .result-lowconf {
            font-size: 0.85rem;
            color: #fdcb6e;
            font-weight: 600;
            margin-top: 8px;
        }

        /* Custom scrollbar for sidebar */
        section[data-testid="stSidebar"]::-webkit-scrollbar { width: 4px; }
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
def metric_card(col, icon: str, label: str, value: str, highlight: bool = False):
    with col:
        glow = "border: 1px solid rgba(108, 92, 231, 0.25);" if highlight else ""
        st.markdown(
            f"""
            <div class="metric-card" style="{glow}">
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
        font=dict(color="#e8e8e8", family="Inter"),
        xaxis=dict(
            range=[0, 100],
            showgrid=True,
            gridcolor="rgba(255,255,255,0.04)",
            ticksuffix="%",
            color="#8a8ab5",
        ),
        yaxis=dict(showgrid=False, color="#e8e8e8", autorange="reversed"),
        showlegend=False,
    )
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

def sidebar_nav(model_loaded: bool, model_label: str, num_classes: int):
    with st.sidebar:
        st.markdown(
            """
            <div style="font-size:1.3rem;font-weight:800;margin-bottom:2px;
                 background: linear-gradient(135deg, #6c5ce7, #a29bfe);
                 -webkit-background-clip: text; background-clip: text;
                 -webkit-text-fill-color: transparent;">
                🌸 AI Vision
            </div>
            <div style="color:#8a8ab5;font-size:0.75rem;margin-bottom:20px;">
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
        st.markdown("<div style='color:#8a8ab5;font-size:0.7rem;text-transform:uppercase;letter-spacing:0.08em;margin-bottom:10px;'>Model Info</div>", unsafe_allow_html=True)
        st.markdown(
            f"""
            <div style="font-size:0.8rem; color:#a8a8d0; line-height:1.9;">
            <span style="color:#8a8ab5;">Architecture:</span> EfficientNetB2<br>
            <span style="color:#8a8ab5;">Classes:</span> {num_classes}<br>
            <span style="color:#8a8ab5;">Framework:</span> TensorFlow/Keras
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
                f"<div style='font-size:0.7rem;color:#5a5a8a;margin-top:6px;'>{model_label}</div>",
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
            "Streamlit · TensorFlow</div>",
            unsafe_allow_html=True,
        )

# ----------------------------------------------------------------------------
# PAGE: DASHBOARD (Rearranged Layout)
# ----------------------------------------------------------------------------
def render_dashboard(model_loaded: bool, classes: list, model_label: str):
    # Hero section
    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown("<div class='page-title'>🌸 AI Vision</div>", unsafe_allow_html=True)
        st.markdown(
            "<div class='page-subtitle'>17-Class Flower Species Classification</div>",
            unsafe_allow_html=True,
        )
        st.markdown(
            "<div style='color:#8a8ab5;font-size:0.95rem;margin-bottom:16px;'>"
            "Upload a flower photo and the AI will identify the species using an EfficientNetB2 "
            "convolutional neural network.</div>",
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("🚀 Go to Classify Image", use_container_width=True):
            st.session_state.page = "Classify Image"
            st.rerun()

    # Stats row - 4 columns
    cols = st.columns(4)
    metric_card(cols[0], "🧠", "Model", "EfficientNetB2", highlight=True)
    metric_card(cols[1], "🏷️", "Classes", f"{len(classes)} Species")
    metric_card(cols[2], "⚡", "Mode", "Single Image")
    metric_card(
        cols[3], "✅" if model_loaded else "⚠️", "Status", "Online" if model_loaded else "Offline",
        highlight=model_loaded
    )

    st.markdown("<br>", unsafe_allow_html=True)

    # Two column layout - left: quick stats, right: system status
    left, right = st.columns([1, 1])

    with left:
        st.markdown("<div class='section-title'>📈 Quick Stats</div>", unsafe_allow_html=True)
        # Stats in a card
        st.markdown(
            """
            <div class="card">
                <div style="display:grid;grid-template-columns:1fr 1fr;gap:12px;">
                    <div>
                        <div style="color:#8a8ab5;font-size:0.7rem;text-transform:uppercase;">Total Predictions</div>
                        <div style="font-size:1.8rem;font-weight:700;color:#e8e8e8;">{}</div>
                    </div>
                    <div>
                        <div style="color:#8a8ab5;font-size:0.7rem;text-transform:uppercase;">Most Predicted</div>
                        <div style="font-size:1.2rem;font-weight:600;color:#a29bfe;">{}</div>
                    </div>
                    <div>
                        <div style="color:#8a8ab5;font-size:0.7rem;text-transform:uppercase;">Avg Confidence</div>
                        <div style="font-size:1.2rem;font-weight:600;color:#55efc4;">{:.1f}%</div>
                    </div>
                    <div>
                        <div style="color:#8a8ab5;font-size:0.7rem;text-transform:uppercase;">Session Status</div>
                        <div style="font-size:1.2rem;font-weight:600;color:#74b9ff;">{}</div>
                    </div>
                </div>
            </div>
            """.format(
                len(st.session_state.history),
                st.session_state.history[-1]["Prediction"].replace("_", " ").title() if st.session_state.history else "—",
                float(st.session_state.history[-1]["Confidence"].rstrip("%")) if st.session_state.history else 0,
                "Active" if st.session_state.history else "Idle"
            ),
            unsafe_allow_html=True,
        )

    with right:
        st.markdown("<div class='section-title'>⚙️ System Status</div>", unsafe_allow_html=True)
        status_text = "Online" if model_loaded else "Offline"
        status_color = "#55efc4" if model_loaded else "#ff7675"
        class_preview = ", ".join(c.replace("_", " ").title() for c in classes[:6])
        st.markdown(
            f"""
            <div class="card">
                <div style="display:flex;justify-content:space-between;padding:6px 0;">
                    <span style="color:#8a8ab5;">Model Status</span>
                    <span style="color:{status_color};font-weight:600;">● {status_text}</span>
                </div>
                <div style="display:flex;justify-content:space-between;padding:6px 0;">
                    <span style="color:#8a8ab5;">Backbone</span>
                    <span style="color:#e8e8e8;">EfficientNetB2</span>
                </div>
                <div style="display:flex;justify-content:space-between;padding:6px 0;">
                    <span style="color:#8a8ab5;">Classification</span>
                    <span style="color:#e8e8e8;">17 Species</span>
                </div>
                <div style="display:flex;justify-content:space-between;padding:6px 0;">
                    <span style="color:#8a8ab5;">Image Size</span>
                    <span style="color:#e8e8e8;">{IMG_SIZE}×{IMG_SIZE}</span>
                </div>
                <div style="display:flex;justify-content:space-between;padding:6px 0;">
                    <span style="color:#8a8ab5;">Sample Classes</span>
                    <span style="color:#a29bfe;text-align:right;">{class_preview}…</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # Full width - Recognized species
    st.markdown("<div class='section-title'>🌸 Recognized Species</div>", unsafe_allow_html=True)
    chip_html = "".join(
        f"<span style='display:inline-block;margin:4px;padding:6px 14px;border-radius:999px;"
        f"background:rgba(108,92,231,0.08);border:1px solid rgba(108,92,231,0.12);"
        f"color:#a8a8d0;font-size:0.8rem;'>"
        f"{c.replace('_',' ').title()}</span>"
        for c in classes
    )
    st.markdown(f"<div class='card'>{chip_html}</div>", unsafe_allow_html=True)

# ----------------------------------------------------------------------------
# PAGE: CLASSIFY IMAGE (Rearranged Layout)
# ----------------------------------------------------------------------------
def render_classifier(model, classes: list, model_label: str):
    st.markdown("<div class='page-title'>🔍 Classify Image</div>", unsafe_allow_html=True)
    st.markdown(
        "<div class='page-subtitle'>Upload a flower photo for AI-powered species identification.</div>",
        unsafe_allow_html=True,
    )

    # Two column layout: image upload on left, instructions on right
    col1, col2 = st.columns([1.2, 1])

    with col1:
        uploaded_file = st.file_uploader(
            "Upload an Image",
            type=list(SUPPORTED_FORMATS),
            label_visibility="collapsed",
        )

    with col2:
        st.markdown(
            """
            <div class="card" style="height:100%;display:flex;flex-direction:column;justify-content:center;">
                <div style="font-weight:600;color:#e8e8e8;margin-bottom:8px;">📋 Instructions</div>
                <div style="color:#8a8ab5;font-size:0.88rem;line-height:1.8;">
                ✅ Upload a clear flower photo<br>
                ✅ Supports JPG, PNG, WEBP<br>
                ✅ Max file size: 200MB<br>
                ✅ AI analyzes in seconds
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    if uploaded_file is None:
        st.markdown(
            "<div class='empty-state' style='margin-top:20px;'>📤 Upload an image to begin AI classification.</div>",
            unsafe_allow_html=True,
        )
        return

    # Image preview and analysis section
    try:
        raw_bytes = uploaded_file.getvalue()
        image = Image.open(io.BytesIO(raw_bytes))
        image.verify()
        image = Image.open(io.BytesIO(raw_bytes))
    except (UnidentifiedImageError, OSError):
        st.error("This file could not be read as a valid image.")
        return

    # Image and analyze button
    img_col, info_col = st.columns([1, 1.2])

    with img_col:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.image(image, width=380)
        st.markdown(
            f"<div style='color:#8a8ab5;font-size:0.8rem;margin-top:8px;text-align:center;'>"
            f"📁 {uploaded_file.name} · {image.size[0]}×{image.size[1]}px</div>",
            unsafe_allow_html=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)

    with info_col:
        st.markdown(
            f"""
            <div class="card" style="height:100%;">
                <div style="font-weight:600;color:#e8e8e8;margin-bottom:12px;">📸 Image Details</div>
                <div style="color:#8a8ab5;font-size:0.88rem;line-height:2.2;">
                    <span style="color:#a8a8d0;">File Name:</span> {uploaded_file.name}<br>
                    <span style="color:#a8a8d0;">Dimensions:</span> {image.size[0]} × {image.size[1]} px<br>
                    <span style="color:#a8a8d0;">Format:</span> {image.format}<br>
                    <span style="color:#a8a8d0;">Mode:</span> {image.mode}<br>
                    <span style="color:#a8a8d0;">Status:</span> <span style="color:#55efc4;">● Ready</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.write("")
        analyze_clicked = st.button("🧠 Analyze Image", use_container_width=True)

    if not analyze_clicked:
        return

    if model is None:
        st.error("Model is not loaded. Please check the sidebar.")
        return

    with st.spinner("🔍 Analyzing image..."):
        try:
            arr = preprocess_image(image)
            start = time.time()
            result = predict_image(model, arr, classes)
            elapsed_ms = (time.time() - start) * 1000
        except Exception as exc:
            st.error("Something went wrong. Please try a different file.")
            with st.expander("Technical details"):
                st.code(str(exc))
            return

    record_prediction(uploaded_file.name, result, model_label)
    st.session_state.last_result = result

    # Results section
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<div class='section-title'>📊 Prediction Results</div>", unsafe_allow_html=True)

    pred_class = result["predicted_class"]
    confidence = result["confidence"]
    display_name = pred_class.replace("_", " ").title()

    low_conf_html = (
        f"<div class='result-lowconf'>⚠️ Low confidence — below {CONFIDENCE_THRESHOLD*100:.0f}% threshold</div>"
        if result["low_confidence"]
        else ""
    )

    # Result card
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

    # Metrics row
    st.write("")
    detail_cols = st.columns(4)
    metric_card(detail_cols[0], "🏷️", "Predicted", display_name)
    metric_card(detail_cols[1], "📈", "Confidence", f"{confidence:.2f}%")
    metric_card(detail_cols[2], "🧠", "Model", model_label)
    metric_card(detail_cols[3], "⏱️", "Time", f"{elapsed_ms:.0f} ms")

    # Charts
    st.write("")
    chart_col, table_col = st.columns([1.3, 1])
    with chart_col:
        st.markdown(f"<div class='section-title'>🎯 Top {TOP_K} Predictions</div>", unsafe_allow_html=True)
        top_k_bar_chart(result["top_k"], classes)
    with table_col:
        st.markdown("<div class='section-title'>📋 Full Distribution</div>", unsafe_allow_html=True)
        prob_df = pd.DataFrame(
            [
                {"Class": k.replace("_", " ").title(), "Probability": f"{v:.2f}%"}
                for k, v in sorted(result["probabilities"].items(), key=lambda kv: kv[1], reverse=True)
            ]
        )
        st.dataframe(prob_df, hide_index=True, use_container_width=True, height=280)

    # History
    st.write("")
    st.markdown("<div class='section-title'>📋 Prediction History</div>", unsafe_allow_html=True)
    hist_df = pd.DataFrame(st.session_state.history[::-1])
    st.dataframe(hist_df, hide_index=True, use_container_width=True)
    if st.button("🗑️ Clear History"):
        st.session_state.history = []
        st.rerun()

# ----------------------------------------------------------------------------
# PAGE: ANALYTICS (Rearranged Layout)
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

    # Stats row - 5 metrics
    cols = st.columns(5)
    metric_card(cols[0], "🔢", "Total Predictions", str(total), highlight=True)
    metric_card(cols[1], "🌼", "Most Predicted", most_common.replace("_", " ").title())
    metric_card(cols[2], "📈", "Avg Confidence", f"{avg_conf:.2f}%")
    metric_card(cols[3], "🏆", "Highest", f"{max_conf:.2f}%")
    metric_card(cols[4], "⚠️", "Low-Conf %", f"{low_conf_rate:.1f}%")

    st.write("")
    # Charts side by side
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
                marker=dict(colors=colors, line=dict(color="#0a0a1a", width=2)),
                textinfo="label+percent",
            )
        )
        fig.update_layout(
            margin=dict(l=10, r=10, t=10, b=10),
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#e8e8e8", family="Inter"),
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
                marker=dict(color="#6c5ce7", line=dict(color="#0a0a1a", width=1)),
            )
        )
        fig2.update_layout(
            margin=dict(l=10, r=10, t=10, b=10),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#e8e8e8", family="Inter"),
            xaxis=dict(title="Confidence (%)", color="#8a8ab5", showgrid=True, gridcolor="rgba(255,255,255,0.04)"),
            yaxis=dict(title="Count", color="#8a8ab5", showgrid=True, gridcolor="rgba(255,255,255,0.04)"),
            height=340,
        )
        st.plotly_chart(fig2, use_container_width=True, config={"displayModeBar": False})

    # History table
    st.write("")
    st.markdown("<div class='section-title'>📋 Session History</div>", unsafe_allow_html=True)
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

    col1, col2 = st.columns([1.5, 1])

    with col1:
        st.markdown(
            f"""
            <div class="card">
                <div style="font-weight:600;margin-bottom:10px;color:#e8e8e8;">🧠 Model Architecture</div>
                <div style="color:#8a8ab5;line-height:1.8;font-size:0.92rem;">
                This application is powered by a <b style="color:#a29bfe;">Convolutional Neural Network</b>
                built on <b style="color:#a29bfe;">EfficientNetB2</b>, a proven image-recognition backbone
                pretrained on ImageNet. The base network was frozen and used as a fixed feature extractor
                to train a new classification head, then partially unfrozen and fine-tuned end-to-end.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.write("")
        st.markdown(
            f"""
            <div class="card">
                <div style="font-weight:600;margin-bottom:10px;color:#e8e8e8;">⚙️ Model Pipeline</div>
                <div style="color:#8a8ab5;line-height:1.8;font-size:0.92rem;">
                Image → Preprocessing (RGB, {IMG_SIZE}×{IMG_SIZE}) → EfficientNetB2 Backbone
                → Global Average Pooling → BN + Dropout → Dense(256, swish) → Dense({len(classes)}, softmax)
                → Predicted Class + Confidence
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col2:
        if model is not None:
            meta = get_model_metadata(model)
            st.markdown(
                f"""
                <div class="card">
                    <div style="font-weight:600;margin-bottom:12px;color:#e8e8e8;">📊 Model Specs</div>
                    <div style="color:#8a8ab5;font-size:0.88rem;line-height:2.4;">
                        <span style="color:#a8a8d0;">Input Shape:</span> {meta['input_shape']}<br>
                        <span style="color:#a8a8d0;">Output Shape:</span> {meta['output_shape']}<br>
                        <span style="color:#a8a8d0;">Layers:</span> {meta['num_layers']}<br>
                        <span style="color:#a8a8d0;">Parameters:</span> {meta['total_params']:,}<br>
                        <span style="color:#a8a8d0;">Classes:</span> {len(classes)}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.info("Load the model to view live architecture details.")

    st.write("")
    st.markdown(
        f"""
        <div class="card">
            <div style="font-weight:600;margin-bottom:10px;color:#e8e8e8;">🔧 Preprocessing Pipeline</div>
            <div style="color:#8a8ab5;line-height:1.9;font-size:0.88rem;">
            • Convert image to RGB<br>
            • Resize to {IMG_SIZE}×{IMG_SIZE} pixels<br>
            • Cast to float32 (no manual division by 255 — EfficientNetB2 normalizes internally)<br>
            • Output activation: Softmax over {len(classes)} classes<br>
            • Low-confidence threshold: {CONFIDENCE_THRESHOLD*100:.0f}%
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.write("")
    st.markdown(f"<div class='section-title'>🏷️ Class Labels ({len(classes)} total)</div>", unsafe_allow_html=True)
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
