"""
AI Vision — 17-Class Flower Classifier
A production-quality Streamlit application serving a trained CNN
(EfficientNetB2 transfer-learning model) for 17-class flower classification.

Model: models/final_model.keras (fallback: models/flower_saved_model/)
Preprocessing (must match training exactly — see notebooks/Flower_Classifier_17Class_HighAccuracy.ipynb):
    - Convert to RGB
    - Resize to 260x260
    - Cast to float32 (NO manual rescale/division by 255 — EfficientNetB2's
      Keras implementation has normalization built into the base model)
Output: softmax over 17 classes, class order taken from models/class_names.json
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
# CONSTANTS — derived from inspection of Flower_Classifier_17Class_HighAccuracy.ipynb
# ----------------------------------------------------------------------------
APP_DIR = Path(__file__).parent
MODELS_DIR = APP_DIR / "models"

MODEL_CANDIDATES = [
    MODELS_DIR / "final_model.keras",
    MODELS_DIR / "flower_saved_model",
]
LABELS_PATH = MODELS_DIR / "class_names.json"

IMG_SIZE = 260           # IMG_SIZE = (260, 260) — EfficientNetB2 native input resolution
SUPPORTED_FORMATS = ("jpg", "jpeg", "png", "webp")
CONFIDENCE_THRESHOLD = 0.45  # matches predict_flower()'s low-confidence flag in the notebook
TOP_K = 5                 # matches the notebook's top5_predictions() diagnostic

PAGES = ["Dashboard", "Classify Image", "Analytics", "About Model"]

# Distinct colors so the top-5 chart / pie stay readable across 17 possible classes
PALETTE = [
    "#6366f1", "#22d3ee", "#f59e0b", "#ec4899", "#22c55e",
    "#a855f7", "#ef4444", "#14b8a6", "#eab308", "#3b82f6",
    "#f97316", "#84cc16", "#e879f9", "#06b6d4", "#f43f5e",
    "#8b5cf6", "#10b981",
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
# CUSTOM CSS — premium dark dashboard (shared visual language with the
# Cat & Dog classifier, re-themed around a floral accent palette)
# ----------------------------------------------------------------------------
def inject_css() -> None:
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

        html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

        :root {
            --bg-primary: #0b0f17;
            --card-bg: rgba(255, 255, 255, 0.035);
            --card-border: rgba(255, 255, 255, 0.08);
            --accent: #ec4899;
            --accent-2: #a855f7;
            --text-primary: #f1f5f9;
            --text-secondary: #94a3b8;
            --success: #22c55e;
            --danger: #ef4444;
            --warning: #f59e0b;
        }

        .stApp {
            background: radial-gradient(circle at 15% 0%, #1c1226 0%, #0b0f17 45%, #090c12 100%);
            color: var(--text-primary);
        }

        section[data-testid="stSidebar"] {
            background: #120d1c;
            border-right: 1px solid var(--card-border);
        }

        #MainMenu, footer, header { visibility: hidden; }

        .block-container { padding-top: 2rem; padding-bottom: 3rem; max-width: 1200px; }

        /* Hero */
        .hero-eyebrow {
            display: inline-block;
            font-size: 0.75rem;
            font-weight: 600;
            letter-spacing: 0.12em;
            text-transform: uppercase;
            color: var(--accent-2);
            background: rgba(168, 85, 247, 0.08);
            border: 1px solid rgba(168, 85, 247, 0.25);
            padding: 4px 12px;
            border-radius: 999px;
            margin-bottom: 14px;
        }
        .hero-title {
            font-size: 2.6rem;
            font-weight: 800;
            line-height: 1.1;
            margin: 0 0 6px 0;
            background: linear-gradient(90deg, #f8fafc, #cbd5e1 60%, var(--accent));
            -webkit-background-clip: text;
            background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        .hero-subtitle {
            font-size: 1.15rem;
            font-weight: 600;
            color: var(--text-primary);
            margin-bottom: 10px;
        }
        .hero-desc {
            color: var(--text-secondary);
            font-size: 0.98rem;
            max-width: 640px;
            line-height: 1.5;
        }

        /* Generic glass card */
        .glass-card {
            background: var(--card-bg);
            border: 1px solid var(--card-border);
            border-radius: 16px;
            padding: 22px 24px;
            backdrop-filter: blur(6px);
        }

        /* Metric cards */
        .metric-card {
            background: var(--card-bg);
            border: 1px solid var(--card-border);
            border-radius: 14px;
            padding: 18px 20px;
            height: 100%;
        }
        .metric-label {
            font-size: 0.75rem;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            color: var(--text-secondary);
            font-weight: 600;
            margin-bottom: 6px;
        }
        .metric-value {
            font-size: 1.35rem;
            font-weight: 700;
            color: var(--text-primary);
        }
        .metric-icon { font-size: 1.4rem; margin-bottom: 8px; opacity: 0.9; }

        .status-pill {
            display: inline-flex;
            align-items: center;
            gap: 6px;
            font-size: 0.8rem;
            font-weight: 600;
            padding: 5px 12px;
            border-radius: 999px;
        }
        .status-online {
            background: rgba(34, 197, 94, 0.1);
            border: 1px solid rgba(34, 197, 94, 0.35);
            color: #4ade80;
        }
        .status-offline {
            background: rgba(239, 68, 68, 0.1);
            border: 1px solid rgba(239, 68, 68, 0.35);
            color: #f87171;
        }
        .status-warning {
            background: rgba(245, 158, 11, 0.1);
            border: 1px solid rgba(245, 158, 11, 0.35);
            color: #fbbf24;
        }
        .status-dot {
            width: 7px; height: 7px; border-radius: 50%; background: currentColor;
        }

        /* Section headings */
        .section-title {
            font-size: 1.25rem;
            font-weight: 700;
            color: var(--text-primary);
            margin: 8px 0 4px 0;
        }
        .section-caption { color: var(--text-secondary); font-size: 0.9rem; margin-bottom: 18px; }

        /* Upload zone */
        div[data-testid="stFileUploaderDropzone"] {
            background: rgba(255, 255, 255, 0.02);
            border: 1.5px dashed rgba(255, 255, 255, 0.18);
            border-radius: 16px;
        }

        /* Buttons */
        .stButton > button {
            background: linear-gradient(90deg, var(--accent), #be185d);
            color: white;
            border: none;
            border-radius: 10px;
            padding: 0.6rem 1.4rem;
            font-weight: 600;
            transition: filter 0.15s ease, transform 0.15s ease;
        }
        .stButton > button:hover { filter: brightness(1.12); transform: translateY(-1px); }

        /* Prediction result card */
        .result-card {
            background: linear-gradient(145deg, rgba(236,72,153,0.10), rgba(168,85,247,0.05));
            border: 1px solid rgba(236, 72, 153, 0.28);
            border-radius: 20px;
            padding: 30px 32px;
            text-align: center;
        }
        .result-label { color: var(--text-secondary); font-size: 0.85rem; text-transform: uppercase; letter-spacing: 0.1em; font-weight: 600; }
        .result-class { font-size: 2.4rem; font-weight: 800; margin: 6px 0; color: var(--text-primary); text-transform: capitalize; }
        .result-confidence { font-size: 1.1rem; color: var(--accent-2); font-weight: 700; }
        .result-lowconf { font-size: 0.85rem; color: #fbbf24; font-weight: 600; margin-top: 8px; }

        .empty-state {
            text-align: center;
            padding: 50px 20px;
            color: var(--text-secondary);
            border: 1px dashed var(--card-border);
            border-radius: 16px;
            background: rgba(255,255,255,0.015);
        }

        .footer-note {
            text-align: center;
            color: var(--text-secondary);
            font-size: 0.78rem;
            margin-top: 40px;
            padding-top: 18px;
            border-top: 1px solid var(--card-border);
        }

        div[data-testid="stDataFrame"] { border-radius: 12px; overflow: hidden; }
        </style>
        """,
        unsafe_allow_html=True,
    )


# ----------------------------------------------------------------------------
# MODEL / LABEL LOADING
# ----------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_model():
    """Load the trained Keras model once, preferring the .keras export."""
    import tensorflow as tf  # imported lazily so the page can render a clean error if missing

    last_error = None
    for path in MODEL_CANDIDATES:
        if path.exists():
            try:
                model = tf.keras.models.load_model(path)
                return model, path
            except Exception as exc:  # noqa: BLE001
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
    """Load the exact 17-class ordering (never guessed)."""
    if not LABELS_PATH.exists():
        raise FileNotFoundError(f"Missing class label file: {LABELS_PATH}")
    with open(LABELS_PATH, "r") as f:
        classes = json.load(f)
    if not isinstance(classes, list) or len(classes) < 2:
        raise ValueError("class_names.json does not contain a valid class list.")
    return classes


def get_model_metadata(model):
    """Pull real architecture facts from the loaded model (no invented values)."""
    try:
        input_shape = model.input_shape
    except Exception:  # noqa: BLE001
        input_shape = None
    try:
        output_shape = model.output_shape
    except Exception:  # noqa: BLE001
        output_shape = None
    try:
        num_layers = len(model.layers)
    except Exception:  # noqa: BLE001
        num_layers = None
    try:
        total_params = int(model.count_params())
    except Exception:  # noqa: BLE001
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
    """Reproduce the exact training-time preprocessing.

    The training notebook resizes to 260x260 and casts to float32 WITHOUT
    dividing by 255 — EfficientNetB2's Keras implementation normalizes
    pixel values internally, so raw 0-255 floats are the correct input.
    """
    img = image.convert("RGB").resize((IMG_SIZE, IMG_SIZE))
    arr = np.asarray(img, dtype=np.float32)
    arr = np.expand_dims(arr, axis=0)  # batch dimension
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
        st.session_state.history = []  # list of dicts
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
        font=dict(color="#f1f5f9", family="Inter"),
        xaxis=dict(range=[0, 100], showgrid=False, ticksuffix="%", color="#94a3b8"),
        yaxis=dict(showgrid=False, color="#f1f5f9", autorange="reversed"),
        showlegend=False,
    )
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


def sidebar_nav(model_loaded: bool, model_label: str, num_classes: int):
    with st.sidebar:
        st.markdown(
            "<div style='font-size:1.3rem;font-weight:800;color:#f1f5f9;"
            "letter-spacing:-0.02em;margin-bottom:0px;'>AI Vision</div>"
            "<div style='color:#94a3b8;font-size:0.8rem;margin-bottom:20px;'>"
            "Computer Vision Suite</div>",
            unsafe_allow_html=True,
        )

        icons = {"Dashboard": "🏠", "Classify Image": "🔍", "Analytics": "📊", "About Model": "ℹ️"}
        for page in PAGES:
            label = f"{icons[page]}  {page}"
            btn_type = "primary" if st.session_state.page == page else "secondary"
            if st.button(label, key=f"nav_{page}", use_container_width=True, type=btn_type):
                st.session_state.page = page

        st.markdown("---")
        st.markdown("<div class='metric-label'>Model Information</div>", unsafe_allow_html=True)
        st.markdown(
            f"""
            <div style="font-size:0.85rem; color:#cbd5e1; line-height:1.9;">
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
                f"<div style='font-size:0.72rem;color:#64748b;margin-top:6px;'>{model_label}</div>",
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                "<span class='status-pill status-offline'><span class='status-dot'></span>"
                "Model Unavailable</span>",
                unsafe_allow_html=True,
            )

        st.markdown(
            "<div class='footer-note'>AI Vision v1.0<br>Built with Streamlit &amp; TensorFlow</div>",
            unsafe_allow_html=True,
        )


# ----------------------------------------------------------------------------
# PAGE: DASHBOARD
# ----------------------------------------------------------------------------
def render_dashboard(model_loaded: bool, classes: list, model_label: str):
    st.markdown("<div class='hero-eyebrow'>Computer Vision · 17-Class Classification</div>", unsafe_allow_html=True)
    st.markdown("<div class='hero-title'>AI Vision</div>", unsafe_allow_html=True)
    st.markdown("<div class='hero-subtitle'>Flower Species Classification</div>", unsafe_allow_html=True)
    st.markdown(
        "<div class='hero-desc'>An AI-powered computer vision system that analyzes uploaded "
        "photos and identifies which of 17 flower species is shown, using an EfficientNetB2 "
        "transfer-learning CNN fine-tuned on a labeled flower image dataset.</div>",
        unsafe_allow_html=True,
    )
    st.write("")

    cols = st.columns(4)
    metric_card(cols[0], "🧠", "Model Type", "CNN (EfficientNetB2)")
    metric_card(cols[1], "🏷️", "Number of Classes", str(len(classes)))
    metric_card(cols[2], "⚡", "Prediction Mode", "Single Image")
    metric_card(
        cols[3], "✅" if model_loaded else "⚠️", "Model Status", "Loaded" if model_loaded else "Unavailable"
    )

    st.write("")
    left, right = st.columns([1.3, 1])
    with left:
        st.markdown("<div class='section-title'>Get Started</div>", unsafe_allow_html=True)
        st.markdown(
            "<div class='section-caption'>Head to the Classify Image page to upload a "
            "photo and run the model.</div>",
            unsafe_allow_html=True,
        )
        st.markdown(
            f"""
            <div class="glass-card">
                <div style="font-weight:700;margin-bottom:10px;">Pipeline</div>
                <div style="color:#94a3b8;font-size:0.92rem;line-height:1.7;">
                Upload Image → Preprocess (resize {IMG_SIZE}×{IMG_SIZE}) →
                EfficientNetB2 Backbone → Dense Head → Softmax (17-way) →
                Class + Confidence
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
            <div class="glass-card">
                <div style="display:flex;justify-content:space-between;margin-bottom:10px;">
                    <span style="color:#94a3b8;">Model</span><span>CNN (Transfer Learning)</span>
                </div>
                <div style="display:flex;justify-content:space-between;margin-bottom:10px;">
                    <span style="color:#94a3b8;">Task</span><span>17-Class Image Classification</span>
                </div>
                <div style="display:flex;justify-content:space-between;margin-bottom:10px;">
                    <span style="color:#94a3b8;">Sample Classes</span><span style="text-align:right;">{class_preview}…</span>
                </div>
                <div style="display:flex;justify-content:space-between;">
                    <span style="color:#94a3b8;">Status</span>
                    <span class="status-pill {status_class}"><span class="status-dot"></span>{status_text}</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.write("")
    st.markdown("<div class='section-title' style='font-size:1.05rem;'>Recognized Species</div>", unsafe_allow_html=True)
    chip_html = "".join(
        f"<span style='display:inline-block;margin:4px;padding:6px 14px;border-radius:999px;"
        f"background:rgba(255,255,255,0.04);border:1px solid {color_for_class(c, classes)}55;"
        f"color:#f1f5f9;font-size:0.82rem;'>{c.replace('_',' ').title()}</span>"
        for c in classes
    )
    st.markdown(f"<div class='glass-card'>{chip_html}</div>", unsafe_allow_html=True)


# ----------------------------------------------------------------------------
# PAGE: CLASSIFY IMAGE
# ----------------------------------------------------------------------------
def render_classifier(model, classes: list, model_label: str):
    st.markdown("<div class='section-title'>Classify Image</div>", unsafe_allow_html=True)
    st.markdown(
        "<div class='section-caption'>Drag and drop a flower photo here or browse your computer.</div>",
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

    # Validate + open the image safely
    try:
        raw_bytes = uploaded_file.getvalue()
        image = Image.open(io.BytesIO(raw_bytes))
        image.verify()
        image = Image.open(io.BytesIO(raw_bytes))  # reopen after verify()
    except (UnidentifiedImageError, OSError):
        st.error("This file could not be read as a valid image. Please upload a JPG, PNG, or WEBP file.")
        return

    img_col, _ = st.columns([1, 1.4])
    with img_col:
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        st.image(image, width=340)
        st.markdown(
            f"<div style='color:#94a3b8;font-size:0.82rem;margin-top:6px;text-align:center;'>"
            f"{uploaded_file.name} · {image.size[0]}×{image.size[1]}px</div>",
            unsafe_allow_html=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)
        st.write("")
        analyze_clicked = st.button("🧠 Analyze Image", use_container_width=True)

    if not analyze_clicked:
        return

    if model is None:
        st.error("The model is not currently loaded, so this image cannot be analyzed. See the sidebar for details.")
        return

    with st.spinner("Analyzing image..."):
        try:
            arr = preprocess_image(image)
            start = time.time()
            result = predict_image(model, arr, classes)
            elapsed_ms = (time.time() - start) * 1000
        except Exception as exc:  # noqa: BLE001
            st.error("Something went wrong while analyzing this image. Please try a different file.")
            with st.expander("Technical details"):
                st.code(str(exc))
            return

    record_prediction(uploaded_file.name, result, model_label)
    st.session_state.last_result = result

    st.write("")
    st.markdown("<div class='section-title'>Prediction</div>", unsafe_allow_html=True)

    pred_class = result["predicted_class"]
    confidence = result["confidence"]
    display_name = pred_class.replace("_", " ").title()

    low_conf_html = (
        f"<div class='result-lowconf'>⚠️ Low confidence — below the "
        f"{CONFIDENCE_THRESHOLD*100:.0f}% threshold, prediction may be unreliable</div>"
        if result["low_confidence"]
        else ""
    )

    st.markdown(
        f"""
        <div class="result-card">
            <div class="result-label">Prediction</div>
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
    metric_card(detail_cols[2], "🧠", "Model Used", model_label)
    metric_card(detail_cols[3], "🖼️", "Image Size", f"{image.size[0]}×{image.size[1]}px")

    st.write("")
    chart_col, table_col = st.columns([1.3, 1])
    with chart_col:
        st.markdown(f"<div class='section-title' style='font-size:1.05rem;'>Top {TOP_K} Predictions</div>", unsafe_allow_html=True)
        top_k_bar_chart(result["top_k"], classes)
    with table_col:
        st.markdown("<div class='section-title' style='font-size:1.05rem;'>Full Class Probabilities</div>", unsafe_allow_html=True)
        prob_df = pd.DataFrame(
            [
                {"Class": k.replace("_", " ").title(), "Probability": f"{v:.2f}%"}
                for k, v in sorted(result["probabilities"].items(), key=lambda kv: kv[1], reverse=True)
            ]
        )
        st.dataframe(prob_df, hide_index=True, use_container_width=True, height=280)
        st.markdown(
            f"<div style='color:#64748b;font-size:0.78rem;margin-top:4px;'>"
            f"Processed in {elapsed_ms:.0f} ms · Status: Completed</div>",
            unsafe_allow_html=True,
        )

    # History
    st.write("")
    st.markdown("<div class='section-title'>Prediction History</div>", unsafe_allow_html=True)
    hist_df = pd.DataFrame(st.session_state.history[::-1])
    st.dataframe(hist_df, hide_index=True, use_container_width=True)
    if st.button("🗑️ Clear History"):
        st.session_state.history = []
        st.rerun()


# ----------------------------------------------------------------------------
# PAGE: ANALYTICS
# ----------------------------------------------------------------------------
def render_analytics(classes: list):
    st.markdown("<div class='section-title'>Analytics</div>", unsafe_allow_html=True)
    st.markdown(
        "<div class='section-caption'>Statistics based only on predictions made during "
        "this session.</div>",
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
        st.markdown("<div class='section-title' style='font-size:1.05rem;'>Prediction Distribution</div>", unsafe_allow_html=True)
        labels = [c.replace("_", " ").title() for c in counts.index]
        colors = [color_for_class(c, classes) for c in counts.index]
        fig = go.Figure(
            go.Pie(
                labels=labels,
                values=counts.values,
                hole=0.55,
                marker=dict(colors=colors, line=dict(color="#0b0f17", width=2)),
                textinfo="label+percent",
            )
        )
        fig.update_layout(
            margin=dict(l=10, r=10, t=10, b=10),
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#f1f5f9", family="Inter"),
            showlegend=False,
            height=340,
        )
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    with chart_col2:
        st.markdown("<div class='section-title' style='font-size:1.05rem;'>Confidence Distribution</div>", unsafe_allow_html=True)
        fig2 = go.Figure(
            go.Histogram(
                x=df["ConfidenceValue"],
                nbinsx=10,
                marker=dict(color="#ec4899"),
            )
        )
        fig2.update_layout(
            margin=dict(l=10, r=10, t=10, b=10),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#f1f5f9", family="Inter"),
            xaxis=dict(title="Confidence (%)", color="#94a3b8", showgrid=False),
            yaxis=dict(title="Count", color="#94a3b8", showgrid=False),
            height=340,
        )
        st.plotly_chart(fig2, use_container_width=True, config={"displayModeBar": False})

    st.write("")
    st.markdown("<div class='section-title' style='font-size:1.05rem;'>Session History</div>", unsafe_allow_html=True)
    st.dataframe(df.drop(columns=["ConfidenceValue"])[::-1], hide_index=True, use_container_width=True)


# ----------------------------------------------------------------------------
# PAGE: ABOUT MODEL
# ----------------------------------------------------------------------------
def render_about(model, classes: list, model_label: str):
    st.markdown("<div class='section-title'>About Model</div>", unsafe_allow_html=True)
    st.markdown(
        "<div class='section-caption'>What is this application, and how does it work?</div>",
        unsafe_allow_html=True,
    )

    st.markdown(
        f"""
        <div class="glass-card">
            <div style="font-weight:700;margin-bottom:8px;">What is this application?</div>
            <div style="color:#94a3b8;line-height:1.7;font-size:0.94rem;">
            This application is powered by a Convolutional Neural Network built on top of
            <b>EfficientNetB2</b>, a proven image-recognition backbone pretrained on ImageNet.
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
        <div class="glass-card">
            <div style="font-weight:700;margin-bottom:8px;">Model Pipeline</div>
            <div style="color:#94a3b8;line-height:1.7;font-size:0.94rem;">
            Image → Preprocessing (RGB conversion, resize to {IMG_SIZE}×{IMG_SIZE})
            → EfficientNetB2 Backbone (pretrained + fine-tuned) → Global Average Pooling
            → Batch Normalization + Dropout → Dense(256, swish) → Dense({len(classes)}, softmax)
            → Predicted Class + Confidence
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.write("")
    st.markdown("<div class='section-title' style='font-size:1.1rem;'>Architecture &amp; Training Details</div>", unsafe_allow_html=True)

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
        <div class="glass-card">
            <div style="font-weight:700;margin-bottom:8px;">Preprocessing (exactly as used in training)</div>
            <div style="color:#94a3b8;line-height:1.9;font-size:0.9rem;">
            • Convert image to RGB<br>
            • Resize to {IMG_SIZE}×{IMG_SIZE} pixels<br>
            • Cast pixel values to float32 (no manual division by 255 — EfficientNetB2's
            Keras implementation normalizes internally)<br>
            • Output activation: Softmax over {len(classes)} classes<br>
            • Low-confidence threshold: predictions below {CONFIDENCE_THRESHOLD*100:.0f}%
            are flagged as low-confidence, matching the notebook's production predictor
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.write("")
    st.markdown("<div class='section-title' style='font-size:1.05rem;'>Class Labels ({} total)</div>".format(len(classes)), unsafe_allow_html=True)
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

    # Load model + labels with robust error handling
    model, model_error = None, None
    model_path = None
    try:
        model, model_path = load_model()
    except Exception as exc:  # noqa: BLE001
        model_error = str(exc)

    classes, labels_error = None, None
    try:
        classes = load_class_labels()
    except Exception as exc:  # noqa: BLE001
        labels_error = str(exc)
        classes = [f"class_{i}" for i in range(17)]  # minimal fallback so the UI can still render

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
