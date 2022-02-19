"""
Trains a CNN model to classify images using the ImageNet
dataset.

Matthew Flegg
matthewflegg@outlook.com
19/02/2022
"""

from create_model import model
from loader import get_image_generators

# Specify the paths containing our dataset
TRAINING_DATA_PATH = "src/dataset/train"
VALIDATION_DATA_PATH = "src/dataset/val"

# Get the training and validation image generators
training_data, validation_data = get_image_generators(
    TRAINING_DATA_PATH,
    VALIDATION_DATA_PATH,
    shuffle=True,
    batch_size=8
    )

# Fitting our network to the image dataset
model.fit(

    # Use training image generator defined in images.py.
    # Train for 30 iterations
    training_data,
    epochs=30,

    # Use validation image generator defined in images.py
    validation_data=validation_data
    )