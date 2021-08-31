from typing import List, Tuple

import tensorflow as tf
from loguru import logger
from tensorflow.keras import Model, backend
from tensorflow.keras.layers import (
    Add,
    BatchNormalization,
    Conv2D,
    DepthwiseConv2D,
    Input,
    ReLU,
)


def inverted_residual_bottleneck(
    tensor: tf.Tensor,
    filters: int,
    expansion_factor: int,
    strides: Tuple[int, int],
    skip_connection: bool,
    name: str,
) -> tf.Tensor:

    in_channels = backend.int_shape(tensor)[-1]

    img = Conv2D(
        filters=expansion_factor * in_channels,
        kernel_size=(1, 1),
        strides=(1, 1),
        padding="same",
        kernel_initializer="he_normal",
        use_bias=False,
        name=f"conv1_{name}",
    )(tensor)
    img = BatchNormalization(name=f"bn1_{name}")(img)
    img = ReLU(max_value=6, name=f"relu1_{name}")(img)

    img = DepthwiseConv2D(
        kernel_size=(3, 3),
        strides=strides,
        padding="same",
        depth_multiplier=1,
        depthwise_initializer="he_normal",
        use_bias=False,
        name=f"depthconv1_{name}",
    )(img)
    img = BatchNormalization(name=f"bn2_{name}")(img)
    img = ReLU(max_value=6, name=f"relu2_{name}")(img)

    img = Conv2D(
        filters=filters,
        kernel_size=(1, 1),
        strides=(1, 1),
        padding="same",
        kernel_initializer="he_normal",
        use_bias=False,
        name=f"conv2_{name}",
    )(img)
    img = BatchNormalization(name=f"bn3_{name}")(img)

    if skip_connection:
        img = Add(name=f"skip_connection_{name}")([img, tensor])

    return img


def get_mobilenetv2(
    img_shape: List[int],
) -> tf.keras.Model:
    """[summary].

    Args:
        img_shape (List[int], optional): [description]. Defaults to img_shape.
        n_classes (int, optional): [description]. Defaults to n_classes.
        repets (int): [description]. Defaults to repetitions.

    Returns:
        tf.keras.Model: [description]
    """
    channels = [32, 16, 24, 32, 64, 96, 160, 320]

    img_input = Input(img_shape)

    img = Conv2D(
        filters=channels[0],
        kernel_size=(3, 3),
        strides=(2, 2),
        padding="same",
        kernel_initializer="he_normal",
        use_bias=True,
    )(img_input)

    img = inverted_residual_bottleneck(
        img,
        filters=channels[1],
        expansion_factor=1,
        strides=(1, 1),
        skip_connection=False,
        name="inv_bottleneck_1",
    )

    img = inverted_residual_bottleneck(
        img,
        filters=channels[2],
        expansion_factor=6,
        strides=(2, 2),
        skip_connection=False,
        name="inv_bottleneck_2_1",
    )
    img = inverted_residual_bottleneck(
        img,
        filters=channels[2],
        expansion_factor=6,
        strides=(1, 1),
        skip_connection=True,
        name="inv_bottleneck_2_2",
    )

    img = inverted_residual_bottleneck(
        img,
        filters=channels[3],
        expansion_factor=6,
        strides=(2, 2),
        skip_connection=False,
        name="inv_bottleneck_3_1",
    )
    img = inverted_residual_bottleneck(
        img,
        filters=channels[3],
        expansion_factor=6,
        strides=(1, 1),
        skip_connection=True,
        name="inv_bottleneck_3_2",
    )
    img = inverted_residual_bottleneck(
        img,
        filters=channels[3],
        expansion_factor=6,
        strides=(1, 1),
        skip_connection=True,
        name="inv_bottleneck_3_3",
    )

    img = inverted_residual_bottleneck(
        img,
        filters=channels[4],
        expansion_factor=6,
        strides=(2, 2),
        skip_connection=False,
        name="inv_bottleneck_4_1",
    )
    img = inverted_residual_bottleneck(
        img,
        filters=channels[4],
        expansion_factor=6,
        strides=(1, 1),
        skip_connection=True,
        name="inv_bottleneck_4_2",
    )
    img = inverted_residual_bottleneck(
        img,
        filters=channels[4],
        expansion_factor=6,
        strides=(1, 1),
        skip_connection=True,
        name="inv_bottleneck_4_3",
    )
    img = inverted_residual_bottleneck(
        img,
        filters=channels[4],
        expansion_factor=6,
        strides=(1, 1),
        skip_connection=True,
        name="inv_bottleneck_4_4",
    )

    img = inverted_residual_bottleneck(
        img,
        filters=channels[5],
        expansion_factor=6,
        strides=(1, 1),
        skip_connection=False,
        name="inv_bottleneck_5_1",
    )
    img = inverted_residual_bottleneck(
        img,
        filters=channels[5],
        expansion_factor=6,
        strides=(1, 1),
        skip_connection=True,
        name="inv_bottleneck_5_2",
    )
    img = inverted_residual_bottleneck(
        img,
        filters=channels[5],
        expansion_factor=6,
        strides=(1, 1),
        skip_connection=True,
        name="inv_bottleneck_5_3",
    )

    img = inverted_residual_bottleneck(
        img,
        filters=channels[6],
        expansion_factor=6,
        strides=(2, 2),
        skip_connection=False,
        name="inv_bottleneck_6_1",
    )
    img = inverted_residual_bottleneck(
        img,
        filters=channels[6],
        expansion_factor=6,
        strides=(1, 1),
        skip_connection=True,
        name="inv_bottleneck_6_2",
    )
    img = inverted_residual_bottleneck(
        img,
        filters=channels[6],
        expansion_factor=6,
        strides=(1, 1),
        skip_connection=True,
        name="inv_bottleneck_6_3",
    )

    return Model(img_input, img)


def get_backbone(
    img_shape: List[int],
) -> tf.keras.Model:

    backbone = get_mobilenetv2(
        img_shape=img_shape,
    )

    endpoint_layers = [
        "skip_connection_inv_bottleneck_2_2",
        "skip_connection_inv_bottleneck_3_3",
        "skip_connection_inv_bottleneck_5_3",
        "skip_connection_inv_bottleneck_6_3",
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
        name="MobileNetv2",
    )


if __name__ == "__main__":

    model = get_backbone(img_shape=[256, 256, 3])
