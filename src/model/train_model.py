"""
Trains a CNN model to classify images using the ImageNet
dataset.

Matthew Flegg
matthewflegg@outlook.com
19/02/2022
"""

from keras.models import Sequential


def start_training_model(training_data, validation_data, model, epochs=30):
    """Trains a CNN model on the specified image dataset.

    Trains the CNN model on the specified images. You can adjust the number of training
    iterations.

    Args:
        training_data (_type_): _description_
        validation_data (_type_): _description_
        epochs (int, optional): _description_. Defaults to 30.
    """

    # Fitting our network to the image dataset
    model.fit(

        # Use training image generator defined in images.py.
        # Train for 30 iterations
        training_data,
        epochs=epochs,

        # Use validation image generator defined in images.py
        validation_data=validation_data
        )