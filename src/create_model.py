"""
Creates a CNN model to classify images using the ImageNet
dataset.

Matthew Flegg
matthewflegg@outlook.com
19/02/2022
"""

from keras.models import Sequential
from keras.layers import (
    Conv2D,
    MaxPooling2D,
    Flatten,
    Dense,
    InputLayer,
    BatchNormalization,
    Dropout,
    )

model = Sequential([

    # The input layer has a shape of (320, 320, 3) because the images are
    # 320 x 320 px, with 3 colour channels for RGB
    InputLayer(input_shape=(320, 320, 3)),

    # Building the three convolution blocks. The second and third convolution
    # blocks normalise the output of the previous layers
    Conv2D(32, (9, 9), activation="relu", strides=(1, 1), padding="same"),
    MaxPooling2D(pool_size=(2, 2), padding="same"),

    Conv2D(64, (5, 5), activation="relu", strides=(2, 2), padding="same", use_bias=True),
    MaxPooling2D(pool_size=(2, 2), padding="same"),
    BatchNormalization(),

    # The third max pooling layer uses valid padding to prevent the filter from
    # slipping outside the input map
    Conv2D(128, (3, 3), activation="relu", strides=(2, 2), padding="same", use_bias=True),
    MaxPooling2D(pool_size=(2, 2), padding="valid"),
    BatchNormalization(),

    # Building the fully connected network. The dropout layer helps prevent
    # overfitting
    Flatten(),
    Dense(units=256, activation="relu"),
    Dense(units=256, activation="relu"),
    Dropout(0.25),

    # Building the output layer with softmax activation
    Dense(units=10, activation="softmax")
    ])

model.compile(

    # Using RMSprop optimiser
    optimizer="rmsprop",

    # Using categorical cross entropy loss because we're dealing with
    # one-hot vector labels
    loss="categorical_crossentropy",
    metrics=["accuracy"]
    )