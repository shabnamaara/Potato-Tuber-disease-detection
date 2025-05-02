import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix

# Define dataset path
DATASET_DIR = "PotatoTuber_Split"
TEST_DIR = os.path.join(DATASET_DIR, "test")

# Image size and batch size
IMG_SIZE = (224, 224)
BATCH_SIZE = 32

# Load class labels (folders in test directory)
CLASS_LABELS = sorted(os.listdir(TEST_DIR))
NUM_CLASSES = len(CLASS_LABELS)

# ImageDataGenerator for test dataset (no augmentation)
test_datagen = ImageDataGenerator(rescale=1.0 / 255.0)

test_generator = test_datagen.flow_from_directory(
    TEST_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=False  # Keep order for confusion matrix
)

# Model paths
MODEL_PATHS = {
    "CNN Model": "models/cnn_model.keras",
    "VGG19 Model": "models/vgg19_model.keras",
    "ResNet50 Model": "models/resnet50_model.keras"
}


# Function to evaluate a model
def evaluate_model(model_path, model_name):
    print(f"\n📊 Evaluating {model_name}...")

    # Load trained model
    model = tf.keras.models.load_model(model_path)

    # Predict on test data
    y_pred_probs = model.predict(test_generator)
    y_pred = np.argmax(y_pred_probs, axis=1)  # Convert probabilities to class index
    y_true = test_generator.classes  # True labels

    # Classification Report
    print("\n🔍 Classification Report:")
    print(classification_report(y_true, y_pred, target_names=CLASS_LABELS))

    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=CLASS_LABELS, yticklabels=CLASS_LABELS)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title(f"Confusion Matrix - {model_name}")
    plt.show()

    # Calculate accuracy
    test_loss, test_acc = model.evaluate(test_generator)
    print(f"\n✅ Test Accuracy ({model_name}): {test_acc * 100:.2f}%\n")


# Evaluate all models
for model_name, model_path in MODEL_PATHS.items():
    if os.path.exists(model_path):
        evaluate_model(model_path, model_name)
    else:
        print(f"⚠️ Model file not found: {model_path}")

print("\n✅ Evaluation Completed!")
