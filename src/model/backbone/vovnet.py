from typing import List, Tuple

import tensorflow as tf
from loguru import logger
from tensorflow.keras import Model
from tensorflow.keras.layers import (
    BatchNormalization,
    Concatenate,
    Conv2D,
    Input,
    MaxPool2D,
    ReLU,
)


def conv_bn_relu(
    tensor: tf.Tensor,
    filters: int,
    kernel_size: Tuple[int, int],
    strides: Tuple[int, int],
    name: str,
    l2_regul: float = 1e-4,
) -> tf.Tensor:

    fmap = Conv2D(
        filters=filters,
        kernel_size=kernel_size,
        strides=strides,
        padding="same",
        use_bias=False,
        kernel_initializer="he_uniform",
        kernel_regularizer=tf.keras.regularizers.l2(l2=l2_regul),
        name=f"{name}",
    )(tensor)

    fmap = BatchNormalization(name=f"bn_{name}")(fmap)

    return ReLU(name=f"relu_{name}")(fmap)


def osa_module(
    tensor: tf.Tensor, filters_conv3x3: int, filters_conv1x1: int, block_name: str
) -> tf.Tensor:

    fmap1 = conv_bn_relu(
        tensor=tensor,
        filters=filters_conv3x3,
        kernel_size=(3, 3),
        strides=(1, 1),
        name=f"conv3_1_block_{block_name}",
    )
    fmap2 = conv_bn_relu(
        tensor=fmap1,
        filters=filters_conv3x3,
        kernel_size=(3, 3),
        strides=(1, 1),
        name=f"conv3_2_block_{block_name}",
    )
    fmap3 = conv_bn_relu(
        tensor=fmap2,
        filters=filters_conv3x3,
        kernel_size=(3, 3),
        strides=(1, 1),
        name=f"conv3_3_block_{block_name}",
    )
    fmap4 = conv_bn_relu(
        tensor=fmap3,
        filters=filters_conv3x3,
        kernel_size=(3, 3),
        strides=(1, 1),
        name=f"conv3_4_block_{block_name}",
    )
    fmap5 = conv_bn_relu(
        tensor=fmap4,
        filters=filters_conv3x3,
        kernel_size=(3, 3),
        strides=(1, 1),
        name=f"conv3_5_block_{block_name}",
    )

    fmap = Concatenate(axis=-1, name=f"concat_{block_name}")(
        [fmap1, fmap2, fmap3, fmap4, fmap5]
    )

    return conv_bn_relu(
        tensor=fmap,
        filters=filters_conv1x1,
        kernel_size=(1, 1),
        strides=(1, 1),
        name=f"conv1_out_block_{block_name}",
    )


def get_vovnet(
    img_shape: List[int],
    filters_conv3x3: List[int],
    filters_conv1x1: List[int],
    block_repetitions: List[int],
) -> tf.keras.Model:

    # input block
    img_input = Input(img_shape)

    fmap = conv_bn_relu(
        tensor=img_input,
        filters=64,
        kernel_size=(3, 3),
        strides=(2, 2),
        name="stem_stage_1",
    )
    fmap = conv_bn_relu(
        tensor=fmap,
        filters=64,
        kernel_size=(3, 3),
        strides=(1, 1),
        name="stem_stage_2",
    )
    fmap = conv_bn_relu(
        tensor=fmap,
        filters=128,
        kernel_size=(3, 3),
        strides=(1, 1),
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

    backbone = get_vovnet(
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
