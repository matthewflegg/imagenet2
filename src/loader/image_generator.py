"""
Contains a function that returns training and validation image generators for a CNN model
using the ImageNet dataset.

Matthew Flegg
matthewflegg@outlook.com
19/02/2022
"""

from keras.preprocessing.image import ImageDataGenerator


def get_image_generators(training_path, validation_path, class_mode="categorical", shuffle=True,
                         batch_size=16, target_size=(320, 320)):
    """Gets the generators for the training images and validation images.

    Uses keras.preprocessing.image.ImageDataGenerator and keras.preprocessing.image.ImageDataGenerator.flow_from_directory
    in order to yield batches of training and validation images with adjustable parameters.

    Args:
        training_path (str): The path of the training images.
        validation_path (str): The path of the validation images.
        class_mode (str, optional): The classification mode of the images. Defaults to "categorical".
        shuffle (bool, optional): Whether to shuffle the images before training. Defaults to True.
        batch_size (int, optional): The size of the batches used for training. Defaults to 16.
        target_size (tuple, optional): The target dimensions of the images. Defaults to (320, 320).

    Returns:
        tuple: The training image generator and the validation
        image generator.
    """

    # Use ImageDataGenerator to load images and infer classes from
    # the number of classes and folder names
    image_generator = ImageDataGenerator()

    # training_image_generator and validation_image_generator yield batches indefinitely
    # Should be 9429 image from 10 classes
    training_image_generator = image_generator.flow_from_directory(
        training_path,
        class_mode=class_mode,
        shuffle=shuffle,
        batch_size=batch_size,
        target_size=target_size,
        )


    # Should be 3925 images from 10 classes
    validation_image_generator = image_generator.flow_from_directory(
        validation_path,
        class_mode=class_mode,
        shuffle=shuffle,
        batch_size=batch_size,
        target_size=target_size,
        )

    return training_image_generator, validation_image_generator