from typing import List

import tensorflow as tf
from loguru import logger
from tensorflow.keras.layers import Input, MaxPooling2D
from tensorflow.keras.models import Model

from src.model.layers.common_layers import bn_relu_conv, residual_bottleneck


def get_feature_extractor(
    img_shape: List[int],
    architecture: str,
    block_repetitions: List[int],
) -> tf.keras.Model:

    if architecture == "A":
        img_input = Input(img_shape)

        fmap = bn_relu_conv(
            tensor=img_input,
            filters=64,
            kernel_size=3,
            strides=2,
            name="block_2",
        )
        fmap = MaxPooling2D(name="maxpool_block2")(fmap)
    else:
        img_input = Input(img_shape)

        fmap = bn_relu_conv(
            tensor=img_input,
            filters=16,
            kernel_size=3,
            strides=1,
            name="block_1",
        )
        fmap = residual_bottleneck(
            tensor=fmap,
            filters=16,
            strides=2,
            shortcut=True,
            name="res0_block1",
        )
        fmap = residual_bottleneck(
            tensor=fmap,
            filters=32,
            strides=2,
            shortcut=True,
            name="res0_block2",
        )

    fmap = residual_bottleneck(
        tensor=fmap,
        filters=64,
        strides=2,
        shortcut=True,
        name="res0_block3",
    )
    for idx0 in range(block_repetitions[0] - 1):
        fmap = residual_bottleneck(tensor=fmap, filters=64, name=f"res{idx0+1}_block3")

    fmap = residual_bottleneck(
        tensor=fmap,
        filters=128,
        name="res0_block4",
        shortcut=True,
    )
    for idx1 in range(block_repetitions[1]):
        fmap = residual_bottleneck(tensor=fmap, filters=128, name=f"res{idx1+1}_block4")

    fmap = residual_bottleneck(
        tensor=fmap,
        filters=256,
        dilation_rate=2,
        name="res0_block5",
        shortcut=True,
    )
    for idx2 in range(block_repetitions[2] - 1):
        fmap = residual_bottleneck(
            tensor=fmap,
            filters=256,
            dilation_rate=2,
            name=f"res{idx2+1}_block5",
        )

    fmap = residual_bottleneck(
        tensor=fmap,
        filters=512,
        dilation_rate=4,
        name="res0_block6",
        shortcut=True,
    )
    for idx3 in range(block_repetitions[3] - 1):
        fmap = residual_bottleneck(
            tensor=fmap,
            filters=512,
            dilation_rate=4,
            name=f"res{idx3+1}_block6",
        )

    if architecture == "C":
        fmap = bn_relu_conv(
            tensor=fmap,
            filters=512,
            kernel_size=3,
            strides=1,
            dilation_rate=2,
            name="0_block_7",
        )
        fmap = bn_relu_conv(
            tensor=fmap,
            filters=512,
            kernel_size=3,
            strides=1,
            dilation_rate=2,
            name="1_block_7",
        )
        fmap = bn_relu_conv(
            tensor=fmap,
            filters=512,
            kernel_size=3,
            strides=1,
            name="0_block_8",
        )
        fmap = bn_relu_conv(
            tensor=fmap,
            filters=512,
            kernel_size=3,
            strides=1,
            name="1_block_8",
        )

    return Model(img_input, fmap)


def get_backbone(
    img_shape: List[int],
    architecture: str,
    block_repetitions: List[int],
    backbone_name: str,
) -> tf.keras.Model:
    """Instantiate the model and use it as a backbone (feature extractor) for a semantic segmentation task.

    Args:
        img_shape (List[int]): Input shape of the images/masks in the dataset.
        filters_conv3x3 (List[int]): List the number of filters used for the 3x3 `Conv2D`
            in each OSA block.
        filters_conv1x1 (List[int]): List the number of filters used for the 1x1 `Conv2D`
            in each OSA block.
        block_repetitions (List[int]): Determine the number of OSA modules to repeat in each block.
        backbone_name (str): Name of the backbone.

    Returns:
        A `tf.keras` model.
    """

    backbone = get_feature_extractor(
        img_shape=img_shape,
        architecture=architecture,
        block_repetitions=block_repetitions,
    )

    if architecture == "A":
        endpoint_layers = [
            "maxpool_block2",
            "add_res2_block6",
        ]
    else:
        endpoint_layers = [
            "add_res0_block2",
            "conv_1_block_8",
        ]

    os4_output, os8_output = [
        backbone.get_layer(layer_name).output for layer_name in endpoint_layers
    ]

    height = img_shape[1]
    logger.info(f"os4_output OS : {int(height/os4_output.shape.as_list()[1])}")
    logger.info(f"os8_output OS : {int(height/os8_output.shape.as_list()[1])}")

    return Model(
        inputs=[backbone.input],
        outputs=[os4_output, os8_output],
        name=backbone_name,
    )


if __name__ == "__main__":
    import numpy as np

    fmap = np.random.rand(1, 256, 256, 3)

    drn = get_backbone(
        img_shape=[256, 256, 3],
        backbone_name="DRN-C-58",
        architecture="C",
        block_repetitions=[3, 4, 6, 3],
    )
    drn.summary()
