import streamlit as st
from PIL import Image
import sys
from pathlib import Path
import plotly.graph_objects as go
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from app.predictor import predict_image, CLASS_NAMES

# ============================================================
# Page Config
# ============================================================
st.set_page_config(
    page_title="Flower AI Classifier",
    page_icon="🌸",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# Global Styling
# ============================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@500;600;700;800&family=Inter:wght@400;500;600&display=swap');

html, body, [class*="css"], .stMarkdown, p, span, div {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background: radial-gradient(circle at 10% 0%, #f3e8ff 0%, #f7f7fc 35%, #fdf2f8 100%);
}

.block-container { max-width: 1180px; padding-top: 1.6rem; padding-bottom: 3rem; }

/* Hero */
.hero {
    padding: 2.6rem 2.4rem;
    border-radius: 26px;
    background: linear-gradient(135deg,#6d28d9 0%,#8b5cf6 45%,#ec4899 100%);
    color: white;
    margin-bottom: 30px;
    box-shadow: 0 25px 55px rgba(109,40,217,.30);
    position: relative;
    overflow: hidden;
}
.hero::after {
    content:"";
    position:absolute; right:-60px; top:-60px;
    width:220px; height:220px; border-radius:50%;
    background:rgba(255,255,255,.12);
}
.hero-badge {
    display:inline-block; padding:6px 14px; border-radius:999px;
    background:rgba(255,255,255,.18); font-size:12px; font-weight:700;
    letter-spacing:.4px; margin-bottom:12px;
}
.hero h1 {
    font-family:'Poppins',sans-serif; font-weight:800; font-size:38px;
    margin:0 0 8px 0;
}
.hero p { opacity:.94; font-size:15.5px; margin:0; max-width:640px; line-height:1.55;}

/* Glass cards */
.glass-card {
    padding: 1.7rem;
    border-radius: 22px;
    background: rgba(255,255,255,.75);
    backdrop-filter: blur(14px);
    border: 1px solid rgba(255,255,255,.6);
    box-shadow: 0 10px 34px rgba(76,29,149,.10);
}

.section-label {
    font-size:11.5px; font-weight:800; letter-spacing:.6px;
    color:#7c3aed; text-transform:uppercase; margin-bottom:6px;
}

/* Result title */
.result-title {
    font-family:'Poppins',sans-serif; font-weight:800; font-size:28px;
    text-transform:capitalize; color:#1f2333; margin:4px 0 2px 0;
}

/* Confidence chip */
.chip {
    display:inline-block; padding:5px 14px; border-radius:999px;
    background:linear-gradient(90deg,#ede9fe,#fce7f3); color:#5b21b6;
    font-weight:700; font-size:13px;
}

/* History thumbnails */
.hist-card {
    border-radius:16px; overflow:hidden; border:1px solid #eee;
    background:#fff; box-shadow:0 6px 18px rgba(0,0,0,.06);
    text-align:center; padding-bottom:8px;
}
.hist-card img { width:100%; height:110px; object-fit:cover; }
.hist-label { font-size:12px; font-weight:700; text-transform:capitalize; margin-top:6px; }
.hist-pct { font-size:11px; color:#6b7280; }

/* Uploader styling */
[data-testid="stFileUploaderDropzone"] {
    border-radius:18px !important;
    border:2px dashed #c4b5fd !important;
    background:#faf9ff !important;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg,#ffffff,#faf7ff);
}

footer {visibility:hidden;}
</style>
""", unsafe_allow_html=True)

# ============================================================
# Hero Section
# ============================================================
st.markdown(
    '<div class="hero">'
    '<div class="hero-badge">✨ CNN • 17-CLASS FLOWER RECOGNITION</div>'
    '<h1>🌸 Flower AI Classifier</h1>'
    '<p>Upload a flower photo and our deep learning model instantly identifies '
    'it from 17 trained species — complete with confidence scoring and the '
    'top 3 closest matches.</p>'
    '</div>',
    unsafe_allow_html=True
)

# ============================================================
# Session State (history)
# ============================================================
if "history" not in st.session_state:
    st.session_state.history = []

# ============================================================
# Sidebar
# ============================================================
with st.sidebar:
    st.markdown("### 🌼 About")
    st.write(
        "This app uses a Convolutional Neural Network trained on 17 flower "
        "categories to classify uploaded images in real time."
    )
    st.markdown("---")
    st.markdown("### 📦 Model Info")
    st.code("models/final_model.keras", language="text")
    st.markdown(f"**Total classes:** {len(CLASS_NAMES)}")
    st.markdown("---")
    st.markdown("### 🏷️ Classes")
    st.write(", ".join(sorted(CLASS_NAMES)))
    st.markdown("---")
    if st.session_state.history:
        if st.button("🗑️ Clear session history", use_container_width=True):
            st.session_state.history = []
            st.rerun()
    st.caption("Built with TensorFlow / Keras + Streamlit")

# ============================================================
# Upload + Predict
# ============================================================
col_upload, col_result = st.columns([1, 1.15], gap="large")

with col_upload:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-label">STEP 1 — UPLOAD</div>', unsafe_allow_html=True)

    uploaded = st.file_uploader(
        "Drag & drop a flower image, or click to browse",
        type=["jpg", "jpeg", "png", "webp", "bmp"],
        label_visibility="collapsed"
    )

    if uploaded:
        image = Image.open(uploaded).convert("RGB")
        st.image(image, caption="Uploaded Image", use_container_width=True)
        predict_clicked = st.button(
            "🔍 Predict Flower", type="primary", use_container_width=True
        )
    else:
        image = None
        predict_clicked = False
        st.info("No image selected yet.")

    st.markdown('</div>', unsafe_allow_html=True)

with col_result:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-label">STEP 2 — RESULT</div>', unsafe_allow_html=True)

    if uploaded and predict_clicked:
        with st.spinner("Analysing image..."):
            try:
                t0 = time.time()
                label, confidence, top_predictions = predict_image(image)
                elapsed = time.time() - t0

                # Header row
                c1, c2 = st.columns([2, 1])
                with c1:
                    st.markdown(f'<div class="result-title">{label}</div>', unsafe_allow_html=True)
                    st.markdown(f'<span class="chip">🌸 Predicted Class</span>', unsafe_allow_html=True)
                with c2:
                    st.metric("Inference time", f"{elapsed:.2f}s")

                st.write("")

                # Gauge chart
                gauge = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=confidence * 100,
                    number={"suffix": "%", "font": {"size": 42}},
                    gauge={
                        "axis": {"range": [0, 100], "tickcolor": "#c4b5fd"},
                        "bar": {"color": "#7c3aed"},
                        "bgcolor": "white",
                        "borderwidth": 0,
                        "steps": [
                            {"range": [0, 50], "color": "#f3e8ff"},
                            {"range": [50, 80], "color": "#e9d5ff"},
                            {"range": [80, 100], "color": "#ddd6fe"},
                        ],
                    },
                    domain={"x": [0, 1], "y": [0, 1]}
                ))
                gauge.update_layout(
                    height=230, margin=dict(t=10, b=0, l=20, r=20),
                    paper_bgcolor="rgba(0,0,0,0)"
                )
                st.plotly_chart(gauge, use_container_width=True)

                # Top-3 bar chart
                st.markdown('<div class="section-label" style="margin-top:6px;">TOP 3 MATCHES</div>', unsafe_allow_html=True)
                names = [p["class_name"] for p in top_predictions][::-1]
                values = [p["confidence_percent"] for p in top_predictions][::-1]

                bar = go.Figure(go.Bar(
                    x=values, y=names, orientation="h",
                    marker=dict(
                        color=values,
                        colorscale=[[0, "#ddd6fe"], [1, "#7c3aed"]],
                    ),
                    text=[f"{v:.1f}%" for v in values],
                    textposition="outside"
                ))
                bar.update_layout(
                    height=180, margin=dict(t=10, b=10, l=10, r=30),
                    xaxis=dict(range=[0, 100], showgrid=False, visible=False),
                    yaxis=dict(showgrid=False),
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                )
                st.plotly_chart(bar, use_container_width=True)

                # Save to session history
                st.session_state.history.insert(0, {
                    "image": image.copy(),
                    "label": label,
                    "confidence": confidence * 100
                })
                st.session_state.history = st.session_state.history[:8]

            except Exception as e:
                st.error(f"Prediction failed: {e}")

    elif uploaded and not predict_clicked:
        st.info("Click **Predict Flower** to analyse this image.")
    else:
        st.markdown(
            '<div style="text-align:center; padding:40px 10px; color:#6b7280;">'
            '<div style="font-size:40px;">🌼</div>'
            '<p>Your prediction result will appear here.</p>'
            '</div>',
            unsafe_allow_html=True
        )

    st.markdown('</div>', unsafe_allow_html=True)

# ============================================================
# Session History
# ============================================================
if st.session_state.history:
    st.write("")
    st.markdown('<div class="section-label">RECENT PREDICTIONS (THIS SESSION)</div>', unsafe_allow_html=True)
    cols = st.columns(len(st.session_state.history))
    for col, item in zip(cols, st.session_state.history):
        with col:
            st.image(item["image"], use_container_width=True)
            st.markdown(
                f'<div class="hist-label">{item["label"]}</div>'
                f'<div class="hist-pct">{item["confidence"]:.1f}% confidence</div>',
                unsafe_allow_html=True
            )

st.write("")
st.markdown(
    '<p style="text-align:center; color:#9ca3af; font-size:12.5px;">'
    'Powered by TensorFlow / Keras CNN • 17 Flower Classes</p>',
    unsafe_allow_html=True
)