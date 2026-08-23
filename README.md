# Flower AI Classifier

A production-quality Streamlit application that serves a trained Convolutional
Neural Network (EfficientNetB2 transfer-learning model) for **17-class flower
species classification**.

![Status](https://img.shields.io/badge/status-active-success)
![Python](https://img.shields.io/badge/python-3.11-blue)
![Streamlit](https://img.shields.io/badge/streamlit-1.32%2B-red)

---

## Overview

This project takes an already-trained CNN model (built and trained in
`notebooks/Flower_Classifier_17Class_HighAccuracy.ipynb`) and wraps it in a
polished, dashboard-style web application. The app does **not** retrain,
modify, or replace the model — it loads the existing trained weights and
reproduces the exact preprocessing pipeline used during training to serve
real, live predictions.

## Features

- **Dashboard** — model status, live metrics, and the full list of recognized
  flower species
- **Classify Image** — drag-and-drop upload, clean image preview, and a
  one-click "Analyze Image" workflow
- **Prediction results** — predicted species, confidence score, a
  low-confidence warning (mirroring the notebook's production predictor),
  and a top-5 probability breakdown driven entirely by real model output
- **Confidence visualization** — interactive horizontal probability bars for
  the top 5 predicted classes (Plotly), plus a full 17-class probability
  table
- **Prediction history** — session-based log of every prediction with a
  one-click clear action
- **Analytics** — live session statistics (totals, most-predicted species,
  average/highest confidence, low-confidence rate) with distribution and
  confidence charts; no fabricated or historical data
- **About Model** — plain-language explanation of the transfer-learning
  architecture, the inference pipeline, and a live, dynamically-generated
  model architecture summary
- **Robust error handling** — missing model/labels, corrupted images, and
  unexpected output are all handled gracefully without exposing stack
  traces to end users

## Machine Learning Model

- **Architecture:** EfficientNetB2 (ImageNet-pretrained backbone, frozen
  then partially fine-tuned) → Global Average Pooling → Batch Normalization
  → Dropout → Dense(256, swish) → Dropout → Dense(17, softmax)
- **Input:** RGB images resized to **260 × 260**
- **Preprocessing:** pixel values are cast to `float32` **without** manual
  division by 255 — EfficientNetB2's Keras implementation normalizes pixel
  values internally, so this matches the training notebook's
  `make_dataset()` and `predict_flower()` functions exactly
- **Output:** Softmax probabilities over 17 classes
- **Class order:** taken directly from `models/class_names.json` — never
  hard-coded or guessed
- **Low-confidence flag:** predictions below **45%** confidence are flagged,
  matching the `confidence_threshold=0.45` used in the notebook's
  `predict_flower()` function
- **Classes:** bluebell, buttercup, colts_foot, cowslip, crocus, daffodil,
  daisy, dandelion, fritillary, iris, lily_valley, pansy, snowdrop,
  sunflower, tigerlily, tulip, windflower

## Project Structure

```text
Flower-Classifier/
│
├── app.py                              # Streamlit application (single entry point)
│
├── models/
│   ├── final_model.keras               # Trained model (preferred, used by the app)
│   ├── class_names.json                # Class order (source of truth)
│   └── flower_saved_model/             # TensorFlow SavedModel export (fallback)
│
├── notebooks/
│   └── Flower_Classifier_17Class_HighAccuracy.ipynb
│
├── assets/                             # Optional static assets
├── requirements.txt
├── README.md
├── .gitignore
├── .gitattributes                      # Git LFS config for the large model file
├── .python-version
└── runtime.txt
```

## Installation

```bash
git clone <your-repo-url>
cd Flower-Classifier
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Running the Application

```bash
streamlit run app.py
```

The app will open automatically at `http://localhost:8501`.

### A note on the model file size (important for GitHub)

`models/final_model.keras` is **~141 MB**, which exceeds GitHub's 100 MB
hard per-file limit for regular pushes. Before pushing this repo to GitHub:

```bash
git lfs install
git lfs track "models/final_model.keras"
git lfs track "models/flower_saved_model/variables/variables.data-00000-of-00001"
git add .gitattributes
git add .
git commit -m "Add flower classifier with Git LFS for large model files"
git push
```

The `.gitattributes` file in this repo is already configured for this. If
you'd rather avoid Git LFS entirely, you can remove `final_model.keras` and
let the app fall back to `models/flower_saved_model/` instead (63 MB,
under GitHub's limit) — the app automatically tries the `.keras` file first
and falls back to the SavedModel directory if it's missing.

### Deploying to Streamlit Community Cloud

TensorFlow does not yet publish wheels for the newest Python releases, and
Streamlit Community Cloud has at times defaulted to a newer, incompatible
Python version regardless of `runtime.txt` / `.python-version`. When
deploying:

1. Open **Advanced settings** before clicking Deploy.
2. Explicitly select **Python 3.11** (or 3.12) from the dropdown.
3. Set the main file path to `app.py`.
4. Confirm the build log's first lines show the correct Python version
   before assuming anything else is wrong.

If your account doesn't offer a Python version selector, or the platform
keeps ignoring it, consider deploying via a `Dockerfile` on a platform like
Hugging Face Spaces or Render instead, which lets you pin the Python
version reliably.

## How It Works

1. An image is uploaded through the drag-and-drop uploader (JPG, JPEG,
   PNG, or WEBP).
2. On clicking **Analyze Image**, the image is converted to RGB and resized
   to 260×260 — identical to the training-time pipeline (no manual pixel
   rescaling, since EfficientNetB2 normalizes internally).
3. The preprocessed array is passed through the cached CNN model
   (`st.cache_resource` ensures the model loads only once per server
   session).
4. The softmax output is mapped back to species names using
   `class_names.json`, and the predicted class + confidence are displayed
   in a dedicated result card, with a low-confidence warning if the top
   prediction falls below 45%.
5. Every prediction is appended to an in-memory session history, and the
   Analytics page aggregates that same session data into live charts.

## Model Information

Real, live details are pulled directly from the loaded Keras model at
runtime (not hard-coded), including input/output shape, layer count,
total parameter count, and the full `model.summary()` output — available
in the **About Model** page.

## Screenshots

_Add screenshots of the Dashboard, Classify Image, and Analytics pages
here once the app has been run locally._

## Future Improvements

- Add Grad-CAM visual explanations for predictions
- Support batch/multi-image classification
- Persist prediction history to a database for cross-session analytics
- Add authentication for multi-user deployments
- Containerize with Docker for simplified, Python-version-safe deployment

## License

This project is provided as-is for educational and portfolio purposes.
Add a license of your choice (e.g. MIT) before publishing publicly.
