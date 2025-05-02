import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG19, ResNet50
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# Define dataset paths
DATASET_DIR = "PotatoTuber_Split"
TRAIN_DIR = os.path.join(DATASET_DIR, "train")
VALID_DIR = os.path.join(DATASET_DIR, "val")
TEST_DIR = os.path.join(DATASET_DIR, "test")

# Image size and batch size
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 20  # You can adjust based on performance

# Define class labels (auto-detected from dataset folder structure)
CLASS_LABELS = sorted(os.listdir(TRAIN_DIR))
NUM_CLASSES = len(CLASS_LABELS)

# Image Data Augmentation
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255.0,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode="nearest"
)

valid_datagen = ImageDataGenerator(rescale=1.0 / 255.0)
test_datagen = ImageDataGenerator(rescale=1.0 / 255.0)

# Load images from directories
train_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical"
)

valid_generator = valid_datagen.flow_from_directory(
    VALID_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical"
)

test_generator = test_datagen.flow_from_directory(
    TEST_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=False
)


# ------------------------- CNN Model -------------------------
def build_cnn_model():
    model = Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation="relu", input_shape=(224, 224, 3)),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(64, (3, 3), activation="relu"),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(128, (3, 3), activation="relu"),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(NUM_CLASSES, activation="softmax")
    ])

    model.compile(optimizer=Adam(learning_rate=0.0001),
                  loss="categorical_crossentropy",
                  metrics=["accuracy"])

    return model


# ------------------------- VGG19 Model -------------------------
def build_vgg19_model():
    base_model = VGG19(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
    base_model.trainable = False  # Freeze pretrained layers

    model = Sequential([
        base_model,
        Flatten(),
        Dense(256, activation="relu"),
        Dropout(0.5),
        Dense(NUM_CLASSES, activation="softmax")
    ])

    model.compile(optimizer=Adam(learning_rate=0.0001),
                  loss="categorical_crossentropy",
                  metrics=["accuracy"])

    return model


# ------------------------- ResNet50 Model -------------------------
def build_resnet50_model():
    base_model = ResNet50(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
    base_model.trainable = False  # Freeze pretrained layers

    model = Sequential([
        base_model,
        Flatten(),
        Dense(256, activation="relu"),
        Dropout(0.5),
        Dense(NUM_CLASSES, activation="softmax")
    ])

    model.compile(optimizer=Adam(learning_rate=0.0001),
                  loss="categorical_crossentropy",
                  metrics=["accuracy"])

    return model


# ------------------------- Training Function -------------------------
def train_and_save_model(model, model_name):
    print(f"\n🚀 Training {model_name}...\n")

    early_stopping = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor="val_loss", factor=0.2, patience=3, min_lr=1e-6)

    history = model.fit(
        train_generator,
        validation_data=valid_generator,
        epochs=EPOCHS,
        callbacks=[early_stopping, reduce_lr]
    )

    # Save trained model
    model_path = f"models/{model_name.lower().replace(' ', '_')}.keras"
    model.save(model_path)
    print(f"\n✅ Model saved: {model_path}\n")


# Train all models
os.makedirs("models", exist_ok=True)

cnn_model = build_cnn_model()
train_and_save_model(cnn_model, "CNN Model")

vgg19_model = build_vgg19_model()
train_and_save_model(vgg19_model, "VGG19 Model")

resnet50_model = build_resnet50_model()
train_and_save_model(resnet50_model, "ResNet50 Model")


# ------------------------- Evaluate Models on Test Set -------------------------
def evaluate_model(model, model_name):
    print(f"\n📊 Evaluating {model_name} on test set...")
    test_loss, test_acc = model.evaluate(test_generator)
    print(f"✅ Test Accuracy ({model_name}): {test_acc * 100:.2f}%\n")


evaluate_model(cnn_model, "CNN Model")
evaluate_model(vgg19_model, "VGG19 Model")
evaluate_model(resnet50_model, "ResNet50 Model")
