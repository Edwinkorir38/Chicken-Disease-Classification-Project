from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import os
import numpy as np
from PIL import Image
import io
import base64
from cnnClassifier import logger

app = Flask(__name__)
CORS(app)

# Load your trained model
import tensorflow as tf
MODEL_PATH = "artifacts/training/model.h5"  # adjust to your actual model path
model = tf.keras.models.load_model(MODEL_PATH)

CLASSES = ["Coccidiosis", "Healthy"]  # adjust to your classes

def preprocess_image(image_bytes):
    img = Image.open(io.BytesIO(image_bytes)).resize((224, 224))
    img_array = np.array(img) / 255.0
    return np.expand_dims(img_array, axis=0)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        file = request.files["file"]
        img_array = preprocess_image(file.read())
        predictions = model.predict(img_array)
        predicted_class = CLASSES[np.argmax(predictions)]
        confidence = float(np.max(predictions))
        return jsonify({
            "prediction": predicted_class,
            "confidence": round(confidence * 100, 2)
        })
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)