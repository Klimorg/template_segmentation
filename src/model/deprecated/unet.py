from typing import Optional, Tuple

import tensorflow as tf
from loguru import logger
from tensorflow.keras import Model
from tensorflow.keras.layers import (
    Concatenate,
    Conv2D,
    Conv2DTranspose,
    Input,
    MaxPool2D,
    ReLU,
)

initializer = "he_normal"


def conv_relu(tensor: tf.Tensor, filters: int) -> tf.Tensor:
    fmap = Conv2D(
        filters,
        kernel_size=(3, 3),
        strides=(1, 1),
        padding="same",
        kernel_initializer=initializer,
    )(tensor)
    fmap = ReLU()(fmap)
    fmap = Conv2D(
        filters,
        kernel_size=(3, 3),
        strides=(1, 1),
        padding="same",
        kernel_initializer=initializer,
    )(fmap)

    return ReLU()(fmap)


def downsample(
    tensor: tf.Tensor,
    sampling_method: Optional[str] = "maxpool",
) -> tf.Tensor:

    if sampling_method == "maxpool":
        fmap = MaxPool2D(pool_size=(2, 2), strides=(2, 2))(tensor)
    elif sampling_method == "conv":
        fmap = Conv2D(
            tensor,
            kernel_size=(3, 3),
            strides=(2, 2),
            padding="same",
            kernel_initializer=initializer,
        )(tensor)
    else:
        logger.error(
            "sampling_method argument not recognized. Please use either 'maxpool' or 'conv'.",
        )

    return fmap


def upsample(tensor: tf.Tensor, filters: int) -> tf.Tensor:
    return Conv2DTranspose(
        filters=filters,
        kernel_size=2,
        strides=(2, 2),
        padding="same",
        kernel_initializer=initializer,
    )(tensor)


def get_segmentation_model(
    filters: int,  # filters = 64 in article
    n_classes: int,
    img_shape: Tuple[int, int, int],
    sampling_method: Optional[str] = "maxpool",
) -> tf.keras.Model:

    growth_rate = [
        filters,
        2 * filters,
        4 * filters,
        8 * filters,
        16 * filters,  # bottleneck part
        8 * filters,
        4 * filters,
        2 * filters,
        filters,
    ]

    # input block
    img_input = Input(img_shape)

    # encoder path
    conv_enc1 = conv_relu(img_input, filters=growth_rate[0])
    down_enc1 = downsample(conv_enc1, sampling_method=sampling_method)

    conv_enc2 = conv_relu(down_enc1, filters=growth_rate[1])
    down_enc2 = downsample(conv_enc2, sampling_method=sampling_method)

    conv_enc3 = conv_relu(down_enc2, filters=growth_rate[2])
    down_enc3 = downsample(conv_enc3, sampling_method=sampling_method)

    conv_enc4 = conv_relu(down_enc3, filters=growth_rate[3])
    down_enc4 = downsample(conv_enc4, sampling_method=sampling_method)

    # bottleneck
    conv_bottleneck = conv_relu(down_enc4, growth_rate[4])

    # decoder path
    up_dec1 = upsample(conv_bottleneck, filters=growth_rate[5])
    merge_dec1 = Concatenate()([conv_enc4, up_dec1])
    conv_dec1 = conv_relu(merge_dec1, filters=growth_rate[5])

    up_dec2 = upsample(conv_dec1, filters=growth_rate[6])
    merge_dec2 = Concatenate()([conv_enc3, up_dec2])
    conv_dec2 = conv_relu(merge_dec2, filters=growth_rate[6])

    up_dec3 = upsample(conv_dec2, filters=growth_rate[7])
    merge_dec3 = Concatenate()([conv_enc2, up_dec3])
    conv_dec3 = conv_relu(merge_dec3, filters=growth_rate[7])

    up_dec4 = upsample(conv_dec3, filters=growth_rate[8])
    merge_dec4 = Concatenate()([conv_enc1, up_dec4])
    conv_dec4 = conv_relu(merge_dec4, filters=growth_rate[8])

    fmap = Conv2D(
        filters=n_classes,
        kernel_size=(3, 3),
        strides=(1, 1),
        padding="same",
        kernel_initializer=initializer,
        activation="relu",
    )(conv_dec4)

    outputs = Conv2D(
        n_classes,
        kernel_size=(1, 1),
        strides=(1, 1),
        kernel_initializer=initializer,
        activation="softmax",
    )(fmap)

    return Model(img_input, outputs)


if __name__ == "__main__":

    from tensorflow.keras.models import save_model

    model = get_segmentation_model(64, 2, (1024, 1024, 3))  # filters = 64 in article

    model.summary()

    model.save("model.h5")
