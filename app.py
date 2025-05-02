import os
import numpy as np
import tensorflow as tf
from flask import Flask, request, render_template, redirect, url_for
from tensorflow.keras.preprocessing import image

# Initialize Flask app
app = Flask(__name__)

# Define upload folder and allowed extensions
UPLOAD_FOLDER = "static/uploads"
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Load trained models
models = {
    "CNN Model": tf.keras.models.load_model("models/cnn_model.keras"),
    "VGG19 Model": tf.keras.models.load_model("models/vgg19_model.keras"),
    "ResNet50 Model": tf.keras.models.load_model("models/resnet50_model.keras"),
}

# Define class labels
CLASS_LABELS = ["Healthy Potato", "Blackspot Bruising Disease", "Soft Rot Disease", "Brown Rot Disease", "Dry Rot Disease"]

# Function to check allowed file types
def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

# Function to preprocess uploaded image
def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img) / 255.0  # Normalize pixel values
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# Home route (upload page)
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        if "file" not in request.files:
            return redirect(request.url)

        file = request.files["file"]
        if file.filename == "" or not allowed_file(file.filename):
            return redirect(request.url)

        if not os.path.exists(UPLOAD_FOLDER):
            os.makedirs(UPLOAD_FOLDER)

        file_path = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(file_path)

        return redirect(url_for("result", filename=file.filename))

    return render_template("index.html")

# Prediction route
@app.route("/result/<filename>")
def result(filename):
    file_path = os.path.join(UPLOAD_FOLDER, filename)

    if not os.path.exists(file_path):
        return "File not found!", 404

    img_array = preprocess_image(file_path)
    predictions = {}

    for model_name, model in models.items():
        pred = model.predict(img_array)
        predicted_class = np.argmax(pred, axis=1)[0]
        predictions[model_name] = CLASS_LABELS[predicted_class]

    # Get the most common prediction (majority voting)
    final_prediction = max(predictions.values(), key=list(predictions.values()).count)

    return render_template("result.html", filename=filename, predictions=predictions, final_prediction=final_prediction)

# Run the Flask app
if __name__ == "__main__":
    app.run(debug=True)
