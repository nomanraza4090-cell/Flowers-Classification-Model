import json
from pathlib import Path

import numpy as np
from PIL import Image
import tensorflow as tf


# =========================
# Paths
# =========================
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "final_model.keras"
CLASS_PATH = BASE_DIR / "models" / "class_names.json"

IMG_SIZE = (260, 260)


# =========================
# Load model and classes
# =========================
print("Loading flower classification model...")

model = tf.keras.models.load_model(
    MODEL_PATH,
    compile=False
)

with open(CLASS_PATH, "r", encoding="utf-8") as f:
    CLASS_NAMES = json.load(f)

print("Model loaded successfully.")
print("Input shape:", model.input_shape)
print("Output shape:", model.output_shape)
print("Classes:", len(CLASS_NAMES))


# =========================
# Core prediction logic (accepts a PIL Image object)
# =========================
def _predict_from_pil(image: Image.Image):

    image = image.convert("RGB")
    image = image.resize(IMG_SIZE)

    image_array = np.asarray(image, dtype=np.float32)
    image_array = np.expand_dims(image_array, axis=0)

    predictions = model.predict(image_array, verbose=0)[0]

    predicted_index = int(np.argmax(predictions))
    predicted_class = CLASS_NAMES[predicted_index]
    confidence = float(predictions[predicted_index])

    top_indices = np.argsort(predictions)[::-1][:3]
    top_predictions = [
        {
            "class_name": CLASS_NAMES[int(i)],
            "confidence_percent": round(float(predictions[i]) * 100, 2)
        }
        for i in top_indices
    ]

    return {
        "class_name": predicted_class,
        "confidence": confidence,
        "confidence_percent": round(confidence * 100, 2),
        "top_predictions": top_predictions
    }


# =========================
# Used by Flask (accepts a file path)
# =========================
def predict_flower(image_path):
    """
    Predict exactly one flower class from the 17 trained classes.
    Accepts a file path.
    """
    image = Image.open(image_path)
    return _predict_from_pil(image)


# =========================
# Used by Streamlit (accepts an in-memory PIL Image)
# Returns (label, confidence_float_0_to_1, top_predictions)
# =========================
def predict_image(image: Image.Image):
    """
    Predict flower class from an already-opened PIL Image object.
    """
    result = _predict_from_pil(image)
    return result["class_name"], result["confidence"], result["top_predictions"]