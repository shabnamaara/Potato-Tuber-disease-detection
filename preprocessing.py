import cv2
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg19 import preprocess_input as vgg19_preprocess
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet_preprocess

def preprocess_image(img_path, model_name):
    """
    Preprocess the input image based on the model type.
    Ensures uniform resizing, normalization, and preprocessing.
    """

    # Load image using OpenCV (fixes blue tint issue)
    img = cv2.imread(img_path)

    if img is None:
        raise ValueError(f"❌ Error: Image not found at {img_path}")

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert from BGR to RGB

    if model_name in ["VGG19 Model", "ResNet50 Model"]:
        # Convert to PIL format for TensorFlow models
        img = Image.fromarray(img)
        img = img.resize((224, 224))  # Resize for TensorFlow models
        img_array = image.img_to_array(img)

        # Model-specific preprocessing
        if model_name == "VGG19 Model":
            img_array = vgg19_preprocess(img_array)  # VGG19-specific normalization
        elif model_name == "ResNet50 Model":
            img_array = resnet_preprocess(img_array)  # ResNet50-specific normalization

    else:  # Custom CNN Model (PyTorch)
        img = Image.fromarray(img)  # Convert for PyTorch transformations

        transform = transforms.Compose([
            transforms.Resize((224, 224)),  # Resize for PyTorch models
            transforms.ToTensor(),  # Convert to tensor
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Pretrained model normalization
        ])

        img_array = transform(img).numpy()  # Convert to NumPy array

    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

    print(f"✅ Processed Image for {model_name}: Shape {img_array.shape}")
    return img_array
