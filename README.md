# FlowerVision AI — CNN Flower Classifier

End-to-end 17-class flower image classifier: EfficientNetB2 transfer-learning
CNN trained in Google Colab, served through a production-ready **Streamlit**
web app (and, independently, a Flask JSON API).

## Description

Upload a photo of a flower and FlowerVision AI predicts its species using a
trained deep learning model, showing the top prediction with a confidence
score plus the top-3 most likely classes.

## Features

- 🌸 Flower image classification across 17 species
- 🧠 Deep learning CNN (EfficientNetB2 backbone)
- 📊 Confidence score with a low-confidence warning
- 🥉 Top-3 predictions with probability bars
- 🖥️ Clean, modern Streamlit interface
- 💻 Runs locally or on Streamlit Community Cloud
- 🔌 Independent Flask JSON API (`/api/predict`) for programmatic use

## Model

| | |
|---|---|
| Architecture | EfficientNetB2 (transfer learning) + dense classification head |
| Framework | TensorFlow / Keras |
| Input size | 260 × 260 × 3 |
| Classes | 17 (see `models/class_names.json`) |
| Trained model | `models/final_model.keras` (used by the Streamlit app) |
| Trained model | `models/flower_saved_model/` (TF SavedModel, used by the Flask API) |

The trained model is already included in this repository. Preprocessing
(resize to 260×260, RGB, raw float32 pixel values) matches the exact
pipeline used during training — see `notebooks/Flower_Classifier_17Class_HighAccuracy.ipynb`.

> Large model files (`models/final_model.keras` and the SavedModel
> `variables/` folder) are tracked with **Git LFS** — see `.gitattributes`.
> Make sure Git LFS is installed (`git lfs install`) before cloning/pushing,
> and enable Git LFS on Streamlit Community Cloud if prompted.

## Installation

```bash
git clone <repository>
cd CNN-Flower-Classifier-Clean
python -m venv .venv
```

Windows activation:

```powershell
.venv\Scripts\activate
```

macOS/Linux activation:

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Run the Streamlit app (local)

```bash
streamlit run app/streamlit_app.py
```

Then open the local URL Streamlit prints (usually `http://localhost:8501`).

## Run the Flask API (optional, independent app)

```bash
python app/flask_api.py
```

Open `http://127.0.0.1:5000`.

API:
- `GET /health`
- `POST /predict` — multipart field `image`
- `POST /api/predict` — multipart field `image`

## Deployment to Streamlit Community Cloud

1. Push this repository to GitHub (with Git LFS enabled for the model files).
2. Go to [share.streamlit.io](https://share.streamlit.io) and sign in.
3. Click **New app** → select this repository and branch.
4. Set the **main file path** to:
   ```text
   app/streamlit_app.py
   ```
5. Click **Deploy**.

```text
GitHub Repository
       ↓
Streamlit Community Cloud
       ↓
Select repository → Select branch
       ↓
Main file: app/streamlit_app.py
       ↓
Deploy
```

## Project Structure

```text
CNN-Flower-Classifier-Clean/
│
├── app/
│   ├── __init__.py
│   ├── streamlit_app.py     # Streamlit UI (entry point)
│   ├── model.py              # Cached model + class-label loading
│   ├── predictor.py          # Preprocessing + inference pipeline
│   ├── utils.py               # Shared helpers (validation, formatting)
│   ├── styles.py              # Custom CSS for the Streamlit UI
│   └── flask_api.py           # Independent Flask JSON API
│
├── models/
│   ├── final_model.keras          # Used by the Streamlit app
│   ├── flower_saved_model/        # Used by the Flask API
│   └── class_names.json           # Shared 17-class label mapping
│
├── static/ , templates/       # Flask front-end assets
├── notebooks/                 # Training notebook (Colab, EfficientNetB2)
├── dataset/                    # Training data (ignored by git)
├── uploads/ , predictions/ , results/
│
├── .streamlit/
│   └── config.toml
├── requirements.txt
├── .gitattributes              # Git LFS rules for the model files
├── .gitignore
└── README.md
```

## Training

1. Open `notebooks/Flower_Classifier_17Class_HighAccuracy.ipynb` in Google Colab.
2. Select **Runtime → Change runtime type → T4 GPU** (or another available GPU).
3. Run cells from top to bottom.
4. Upload the dataset ZIP when requested.
5. The notebook saves the final model and `class_names.json` into `models/`.

Re-training is **not** required to run the app — a trained model is already
included in `models/`.
