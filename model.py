import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.vgg19 import preprocess_input as vgg19_preprocess
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet_preprocess
from tensorflow.keras.preprocessing import image

# Define class labels
CLASS_LABELS = ["Healthy Potato", "Blackspot Bruising Disease", "Soft Rot Disease", "Brown Rot Disease", "Dry Rot Disease"]

# Load trained models
models = {
    "CNN Model": tf.keras.models.load_model("models/cnn_model.keras"),
    "VGG19 Model": tf.keras.models.load_model("models/vgg19_model.keras"),
    "ResNet50 Model": tf.keras.models.load_model("models/resnet50_model.keras"),
}

def preprocess_image(img_path, model_name):
    """
    Preprocess the input image according to the model type.
    """
    img = image.load_img(img_path, target_size=(224, 224))  # Resize image
    img_array = image.img_to_array(img)

    # Apply specific preprocessing for each model
    if model_name == "VGG19 Model":
        img_array = vgg19_preprocess(img_array)  # VGG19 preprocessing
    elif model_name == "ResNet50 Model":
        img_array = resnet_preprocess(img_array)  # ResNet50 preprocessing
    else:  # Custom CNN Model
        img_array = img_array / 255.0  # Normalize for CNN

    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array

def predict_disease(img_path):
    """
    Run predictions on the input image using all models.
    """
    predictions = {}

    for model_name, model in models.items():
        processed_img = preprocess_image(img_path, model_name)
        pred = model.predict(processed_img)
        predicted_class = np.argmax(pred, axis=1)[0]  # Get class with highest probability
        predictions[model_name] = CLASS_LABELS[predicted_class]  # Map index to class label

    return predictions
