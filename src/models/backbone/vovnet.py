from typing import List

import tensorflow as tf
from loguru import logger
from tensorflow.keras import Model
from tensorflow.keras.layers import Concatenate, Input, MaxPool2D

from src.models.layers.common_layers import conv_bn_relu


def osa_module(
    tensor: tf.Tensor,
    filters_conv3x3: int,
    filters_conv1x1: int,
    block_name: str,
) -> tf.Tensor:
    """One-Shot Aggregation module, the backbone of the VoVNet model.

    Architecture:
        ![Architecture](./images/osa_module.svg)

    Args:
        tensor (tf.Tensor): Input feature map of the module, size = $(H,W,C)$.
        filters_conv3x3 (int): Numbers of filters used in the 3x3 `Conv2D` layers.
        filters_conv1x1 (int): Numbers of filters used in the 1x1 `Conv2D` layer.
        block_name (str): Name of the module.

    Returns:
        Output feature map, size = $(H,W,\mathrm{filters\_ conv1x1})$.
    """

    fmap1 = conv_bn_relu(
        tensor=tensor,
        filters=filters_conv3x3,
        kernel_size=3,
        strides=1,
        name=f"conv3_1_block_{block_name}",
    )
    fmap2 = conv_bn_relu(
        tensor=fmap1,
        filters=filters_conv3x3,
        kernel_size=3,
        strides=1,
        name=f"conv3_2_block_{block_name}",
    )
    fmap3 = conv_bn_relu(
        tensor=fmap2,
        filters=filters_conv3x3,
        kernel_size=3,
        strides=1,
        name=f"conv3_3_block_{block_name}",
    )
    fmap4 = conv_bn_relu(
        tensor=fmap3,
        filters=filters_conv3x3,
        kernel_size=3,
        strides=1,
        name=f"conv3_4_block_{block_name}",
    )
    fmap5 = conv_bn_relu(
        tensor=fmap4,
        filters=filters_conv3x3,
        kernel_size=3,
        strides=1,
        name=f"conv3_5_block_{block_name}",
    )

    fmap = Concatenate(axis=-1, name=f"concat_{block_name}")(
        [fmap1, fmap2, fmap3, fmap4, fmap5],
    )

    return conv_bn_relu(
        tensor=fmap,
        filters=filters_conv1x1,
        kernel_size=1,
        strides=1,
        name=f"conv1_out_block_{block_name}",
    )


def get_feature_extractor(
    img_shape: List[int],
    filters_conv3x3: List[int],
    filters_conv1x1: List[int],
    block_repetitions: List[int],
) -> tf.keras.Model:
    """Instantiate a VoVNet model.

    Args:
        img_shape (List[int]): Input shape of the images/masks in the dataset.
        filters_conv3x3 (List[int]): List the number of filters used for the 3x3 `Conv2D`
            in each OSA block.
        filters_conv1x1 (List[int]): List the number of filters used for the 1x1 `Conv2D`
            in each OSA block.
        block_repetitions (List[int]): Determine the number of OSA modules to repeat in each block.

    Returns:
        A `tf.keras` model.
    """

    # input block
    img_input = Input(img_shape)

    fmap = conv_bn_relu(
        tensor=img_input,
        filters=64,
        kernel_size=3,
        strides=2,
        name="stem_stage_1",
    )
    fmap = conv_bn_relu(
        tensor=fmap,
        filters=64,
        kernel_size=3,
        strides=1,
        name="stem_stage_2",
    )
    fmap = conv_bn_relu(
        tensor=fmap,
        filters=128,
        kernel_size=3,
        strides=1,
        name="stem_stage_3",
    )

    for idx0 in range(block_repetitions[0]):
        fmap = osa_module(
            tensor=fmap,
            filters_conv3x3=filters_conv3x3[0],
            filters_conv1x1=filters_conv1x1[0],
            block_name=f"1_{idx0}",
        )
    fmap = MaxPool2D(pool_size=(2, 2), name="maxpool_block1_out")(fmap)

    for idx1 in range(block_repetitions[1]):
        fmap = osa_module(
            tensor=fmap,
            filters_conv3x3=filters_conv3x3[1],
            filters_conv1x1=filters_conv1x1[1],
            block_name=f"2_{idx1}",
        )
    fmap = MaxPool2D(pool_size=(2, 2), name="maxpool_block2_out")(fmap)

    for idx2 in range(block_repetitions[2]):
        fmap = osa_module(
            tensor=fmap,
            filters_conv3x3=filters_conv3x3[2],
            filters_conv1x1=filters_conv1x1[2],
            block_name=f"3_{idx2}",
        )
    fmap = MaxPool2D(pool_size=(2, 2), name="maxpool_block3_out")(fmap)

    for idx3 in range(block_repetitions[3]):
        fmap = osa_module(
            tensor=fmap,
            filters_conv3x3=filters_conv3x3[3],
            filters_conv1x1=filters_conv1x1[3],
            block_name=f"4_{idx3}",
        )
    fmap_out = MaxPool2D(pool_size=(2, 2), name="maxpool_block4_out")(fmap)

    return Model(img_input, fmap_out, name="VoVNet")


def get_backbone(
    img_shape: List[int],
    filters_conv3x3: List[int],
    filters_conv1x1: List[int],
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
        filters_conv3x3=filters_conv3x3,
        filters_conv1x1=filters_conv1x1,
        block_repetitions=block_repetitions,
    )

    endpoint_layers = [
        "maxpool_block1_out",
        "maxpool_block2_out",
        "maxpool_block3_out",
        "maxpool_block4_out",
    ]

    os4_output, os8_output, os16_output, os32_output = [
        backbone.get_layer(layer_name).output for layer_name in endpoint_layers
    ]

    height = img_shape[1]
    logger.info(f"os4_output OS : {int(height/os4_output.shape.as_list()[1])}")
    logger.info(f"os8_output OS : {int(height/os8_output.shape.as_list()[1])}")
    logger.info(f"os16_output OS : {int(height/os16_output.shape.as_list()[1])}")
    logger.info(f"os32_output OS : {int(height/os32_output.shape.as_list()[1])}")

    return Model(
        inputs=[backbone.input],
        outputs=[os4_output, os8_output, os16_output, os32_output],
        name=backbone_name,
    )
