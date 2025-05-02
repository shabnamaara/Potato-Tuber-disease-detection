import os
import shutil
import splitfolders

DATASET_PATH = r"C:\Users\shaba\Downloads\PotatoTuber\Original Images"
OUTPUT_PATH = "PotatoTuber_Split"

# Splitting into train (70%), validation (15%), test (15%)
splitfolders.ratio(DATASET_PATH, output=OUTPUT_PATH, seed=42, ratio=(0.7, 0.15, 0.15))

print("Data split completed.")
