from typing import Optional, Tuple

from tensorflow import keras
from tensorflow.keras import Model, layers

initializer = "he_normal"


def get_segmentation_model(n_classes: int, img_shape: Tuple[int, int, int]):

    inputs = keras.Input(shape=img_shape)

    ### [First half of the network: downsampling inputs] ###

    # Entry block
    x = layers.Conv2D(
        32,
        3,
        strides=2,
        padding="same",
        kernel_regularizer=keras.regularizers.l2(l2=1e-4),
    )(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    previous_block_activation = x  # Set aside residual

    # Blocks 1, 2, 3 are identical apart from the feature depth.
    for filters in [64, 128, 256]:
        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(
            filters,
            3,
            padding="same",
            kernel_regularizer=keras.regularizers.l2(l2=1e-4),
        )(x)
        x = layers.BatchNormalization()(x)

        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(
            filters,
            3,
            padding="same",
            kernel_regularizer=keras.regularizers.l2(l2=1e-4),
        )(x)
        x = layers.BatchNormalization()(x)

        x = layers.MaxPooling2D(3, strides=2, padding="same")(x)

        # Project residual
        residual = layers.Conv2D(
            filters,
            1,
            strides=2,
            padding="same",
            kernel_regularizer=keras.regularizers.l2(l2=1e-4),
        )(previous_block_activation)
        x = layers.add([x, residual])  # Add back residual
        previous_block_activation = x  # Set aside next residual

    ### [Second half of the network: upsampling inputs] ###

    for filters in [256, 128, 64, 32]:
        x = layers.Activation("relu")(x)
        x = layers.Conv2DTranspose(
            filters,
            3,
            padding="same",
            kernel_regularizer=keras.regularizers.l2(l2=1e-4),
        )(x)
        x = layers.BatchNormalization()(x)

        x = layers.Activation("relu")(x)
        x = layers.Conv2DTranspose(
            filters,
            3,
            padding="same",
            kernel_regularizer=keras.regularizers.l2(l2=1e-4),
        )(x)
        x = layers.BatchNormalization()(x)

        x = layers.UpSampling2D(2)(x)

        # Project residual
        residual = layers.UpSampling2D(2)(previous_block_activation)
        residual = layers.Conv2D(
            filters,
            1,
            padding="same",
            kernel_regularizer=keras.regularizers.l2(l2=1e-4),
        )(residual)
        x = layers.add([x, residual])  # Add back residual
        previous_block_activation = x  # Set aside next residual

    # Add a per-pixel classification layer
    outputs = layers.Conv2D(
        n_classes,
        3,
        activation="softmax",
        padding="same",
        kernel_regularizer=keras.regularizers.l2(l2=1e-4),
    )(x)

    # Define the model
    return Model(inputs, outputs)
