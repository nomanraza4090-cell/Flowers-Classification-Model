from flask import Flask, request, jsonify, render_template
from pathlib import Path
import json

import numpy as np
from PIL import Image
import tensorflow as tf


# ============================================================
# Flask App
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

app = Flask(
    __name__,
    template_folder=str(BASE_DIR / "templates"),
    static_folder=str(BASE_DIR / "static")
)

MODEL_DIR = BASE_DIR / "models" / "flower_saved_model"
CLASS_FILE = BASE_DIR / "models" / "class_names.json"
UPLOAD_DIR = BASE_DIR / "uploads"

UPLOAD_DIR.mkdir(exist_ok=True)


# ============================================================
# Load Classes
# ============================================================

with open(CLASS_FILE, "r", encoding="utf-8") as f:
    CLASS_NAMES = json.load(f)


# ============================================================
# Load SavedModel
# ============================================================

print("Loading SavedModel...")

saved_model = tf.saved_model.load(
    str(MODEL_DIR)
)

# Use the exact signature detected earlier
infer = saved_model.signatures["serve"]

print("✅ SavedModel loaded successfully")
print("✅ Classes:", len(CLASS_NAMES))
print("✅ Signature: serve")


# ============================================================
# Prediction Function
# ============================================================

def predict_flower(image_path):

    # Open image
    image = Image.open(image_path).convert("RGB")

    # SAME SIZE AS MODEL
    image = image.resize((260, 260))

    # Convert to NumPy
    image_array = np.asarray(
        image,
        dtype=np.float32
    )

    # Add batch dimension
    image_array = np.expand_dims(
        image_array,
        axis=0
    )

    # TensorFlow tensor
    input_tensor = tf.convert_to_tensor(
        image_array,
        dtype=tf.float32
    )

    # Run SavedModel
    outputs = infer(
        input_tensor
    )

    # Get model output
    output_tensor = list(outputs.values())[0]

    predictions = output_tensor.numpy()[0]

    # Best prediction
    predicted_index = int(
        np.argmax(predictions)
    )

    predicted_class = CLASS_NAMES[predicted_index]

    confidence = float(
        predictions[predicted_index]
    )

    # ------------------------------------------------------
    # Top-3 predictions (for the pro UI confidence bars)
    # ------------------------------------------------------
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
        "confidence_percent": round(
            confidence * 100,
            2
        ),
        "top_predictions": top_predictions
    }


# ============================================================
# Home Page  (now served from templates/index.html)
# ============================================================

@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")


# ============================================================
# Health Check
# ============================================================

@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "classes": len(CLASS_NAMES)
    })


# ============================================================
# JSON API  (used by the drag & drop frontend)
# ============================================================

@app.route("/api/predict", methods=["POST"])
def api_predict():

    try:

        if "image" not in request.files:

            return jsonify({
                "success": False,
                "error": "No image uploaded"
            }), 400

        file = request.files["image"]

        if file.filename == "":

            return jsonify({
                "success": False,
                "error": "No image selected"
            }), 400

        file_path = UPLOAD_DIR / file.filename

        file.save(file_path)

        result = predict_flower(
            file_path
        )

        return jsonify({
            "success": True,
            "prediction": result["class_name"],
            "confidence": result["confidence_percent"],
            "top_predictions": result["top_predictions"]
        })

    except Exception as e:

        print("❌ Prediction Error:", repr(e))

        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


# ============================================================
# Optional: Full-page result route (kept for compatibility)
# ============================================================

@app.route("/predict", methods=["POST"])
def predict():

    try:

        if "image" not in request.files:

            return jsonify({
                "success": False,
                "error": "No image uploaded"
            }), 400

        file = request.files["image"]

        if file.filename == "":

            return jsonify({
                "success": False,
                "error": "No image selected"
            }), 400

        # Save upload
        file_path = UPLOAD_DIR / file.filename

        file.save(file_path)

        # Predict
        result = predict_flower(
            file_path
        )

        return jsonify({
            "success": True,
            "prediction": result["class_name"],
            "confidence": result["confidence_percent"],
            "top_predictions": result["top_predictions"]
        })

    except Exception as e:

        print("❌ Prediction Error:", repr(e))

        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


# ============================================================
# Start Flask
# ============================================================

if __name__ == "__main__":

    app.run(
        host="127.0.0.1",
        port=5000,
        debug=False,
        use_reloader=False
    )