import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0" 

from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
import cv2

app = Flask(__name__)

MODEL_PATH = os.path.join("E:\\", "Plant Disease Detection", "model", "plant_disease_detection_model.h5")
model = tf.keras.models.load_model(MODEL_PATH)

# Warmup
dummy = np.zeros((1, 128, 128, 3), dtype=np.float32)
model.predict(dummy, verbose=0)
print("✅ Model loaded and warmed up!")

DATASET_PATH = os.path.join("E:\\", "Plant Disease Detection", "dataset", "Plant_Village", "PlantVillage")
class_names = sorted(os.listdir(DATASET_PATH))

recommendations = {
    "Pepper__bell___Bacterial_spot": "Use copper-based bactericide and remove infected leaves.",
    "Pepper__bell___healthy":        "Plant is healthy. Continue normal care.",
    "Potato___Early_blight":         "Use Mancozeb fungicide and remove infected leaves.",
    "Potato___Late_blight":          "Avoid overhead watering and use copper-based fungicide.",
    "Potato___healthy":              "Plant is healthy. Maintain regular watering.",
    "Tomato_Bacterial_spot":         "Use copper spray and avoid overhead irrigation.",
    "Tomato_Early_blight":           "Use fungicide spray and maintain proper spacing.",
    "Tomato_Late_blight":            "Remove affected leaves and use chlorothalonil spray.",
    "Tomato_Leaf_Mold":              "Improve ventilation and use fungicide.",
    "Tomato_Septoria_leaf_spot":     "Remove infected leaves and apply fungicide.",
    "Tomato_Spider_mites_Two_spotted_spider_mite": "Use miticide or neem oil spray.",
    "Tomato__Target_Spot":           "Apply fungicide and remove soil debris.",
    "Tomato__Tomato_YellowLeaf__Curl_Virus": "Control whiteflies and remove infected plants.",
    "Tomato__Tomato_mosaic_virus":   "Remove infected plants and disinfect tools.",
    "Tomato_healthy":                "Plant is healthy. Maintain regular watering."
}

IMG_SIZE = (128, 128)

def preprocess(file_bytes):
    np_arr = np.frombuffer(file_bytes, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (IMG_SIZE[1], IMG_SIZE[0]))
    return np.expand_dims(img, axis=0).astype(np.float32)  # no /255 !

@app.route("/", methods=["GET"])
def health():
    return jsonify({"status": "Plant Disease Detection API is running!"})

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    try:
        img_array = preprocess(file.read())
        preds      = model.predict(img_array, verbose=0)
        pred_index = int(np.argmax(preds))
        pred_class = class_names[pred_index]
        confidence = float(np.max(preds)) * 100
        advice     = recommendations.get(pred_class, "Consult an agricultural expert.")

        top3_idx = np.argsort(preds[0])[::-1][:3]
        top3 = [
            {"class": class_names[i], "confidence": round(float(preds[0][i]) * 100, 2)}
            for i in top3_idx
        ]

        return jsonify({
            "predicted_class": pred_class,
            "confidence":      round(confidence, 2),
            "recommendation":  advice,
            "top3":            top3
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True, port=5000, use_reloader=False)
