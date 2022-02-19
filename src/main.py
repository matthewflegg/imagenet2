"""
Creates, compiles and trains a CNN model to classify images using the
ImageNet dataset.

Matthew Flegg
matthewflegg@outlook.com
19/02/2022
"""

from loader import get_image_generators
from model import get_model, start_training_model


# Specify the path containing our dataset
TRAINING_DATA_PATH = "src/dataset/train"
VALIDATION_DATA_PATH = "src/dataset/val"

# Get the image generators for the training dataset
training_images, validation_images = get_image_generators(
    TRAINING_DATA_PATH,
    VALIDATION_DATA_PATH,
    class_mode="categorical",
    shuffle=True,
    batch_size=8,
    target_size=(320, 320),
    )

# Build the CNN model
model = get_model()

# Train the CNN model
start_training_model(
    training_images,
    validation_images,
    model,
    epochs=30
    )