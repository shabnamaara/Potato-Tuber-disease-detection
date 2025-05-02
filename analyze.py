import os
import random
from PIL import Image
import matplotlib.pyplot as plt

DATASET_PATH = r"C:\Users\shaba\Downloads\PotatoTuber\Original Images"

def analyze_images():
    image_formats = set()
    image_sizes = {}
    total_images = 0

    for category in os.listdir(DATASET_PATH):
        category_path = os.path.join(DATASET_PATH, category)
        if os.path.isdir(category_path):
            images = os.listdir(category_path)
            total_images += len(images)

            for img_name in images:
                img_path = os.path.join(category_path, img_name)
                with Image.open(img_path) as img:
                    image_formats.add(img.format)
                    size = img.size
                    if size in image_sizes:
                        image_sizes[size] += 1
                    else:
                        image_sizes[size] = 1

    print(f"Total Images: {total_images}")
    print(f"Image Formats: {image_formats}")
    print(f"Image Sizes (with count): {image_sizes}")

# Run the analysis
analyze_images()
def display_random_images():

    categories = os.listdir(DATASET_PATH)
    selected_images = []

    for _ in range(5):
        category = random.choice(categories)
        category_path = os.path.join(DATASET_PATH, category)
        image_name = random.choice(os.listdir(category_path))
        img_path = os.path.join(category_path, image_name)
        selected_images.append((img_path, category))

    fig, axes = plt.subplots(1, 5, figsize=(15, 5))
    for ax, (img_path, category) in zip(axes, selected_images):
        img = Image.open(img_path)
        ax.imshow(img)
        ax.set_title(category)
        ax.axis("off")

    plt.show()

display_random_images()
