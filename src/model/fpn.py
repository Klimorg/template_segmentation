from typing import List

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Activation, Concatenate, Conv2D, UpSampling2D

from src.model.layers.common_layers import conv_gn_relu


def FPN(
    c2_output: tf.Tensor,
    c3_output: tf.Tensor,
    c4_output: tf.Tensor,
    c5_output: tf.Tensor,
    filters: int = 256,
    kernel_size=(1, 1),
    strides=(1, 1),
    padding="same",
    kernel_initializer="he_uniform",
    l2_regul: float = 1e-4,
) -> List[tf.Tensor]:
    """The Feature Pyramid Networks head.

    Architecture:
        ![screen](./images/fpn_head.svg)

    Args:
        c2_output (tf.Tensor): Feature map coming from the backbone, output stride 4.
        c3_output (tf.Tensor): Feature map coming from the backbone, output stride 8.
        c4_output (tf.Tensor): Feature map coming from the backbone, output stride 16.
        c5_output (tf.Tensor): Feature map coming from the backbone, output stride 32.
        filters (int, optional): Number of filters in each `Conv2D` layers. Defaults to 256.
        kernel_size (tuple, optional): Size of the convolution kernels in each `Conv2D` layers.
            Defaults to (1, 1).
        strides (tuple, optional): Stride parameter in each `Conv2D` layers. Defaults to (1, 1).
        padding (str, optional): Paddinf parameter in each `Conv2D` layers. Defaults to "same".
        kernel_initializer (str, optional): Kernel initialization method used in each `Conv2D` layers.
            Defaults to "he_uniform".
        l2_regul (float, optional): Value of the constraint used for the
            $L_2$ regularization. Defaults to 1e-4.

    Returns:
        A list of feature maps, of dimensions $[(OS4, 256), (OS8, 256), (OS16, 256), (OS32, 256)]$.
    """

    # rescale filters and go down through pyramid network
    p5_output = Conv2D(
        filters=filters,
        kernel_size=kernel_size,
        strides=strides,
        padding=padding,
        kernel_initializer=kernel_initializer,
        kernel_regularizer=tf.keras.regularizers.l2(l2=l2_regul),
    )(c5_output)

    p4_output = Conv2D(
        filters=filters,
        kernel_size=kernel_size,
        strides=strides,
        padding=padding,
        kernel_initializer=kernel_initializer,
        kernel_regularizer=tf.keras.regularizers.l2(l2=l2_regul),
    )(c4_output)

    p3_output = Conv2D(
        filters=filters,
        kernel_size=kernel_size,
        strides=strides,
        padding=padding,
        kernel_initializer=kernel_initializer,
        kernel_regularizer=tf.keras.regularizers.l2(l2=l2_regul),
    )(c3_output)

    p2_output = Conv2D(
        filters=filters,
        kernel_size=kernel_size,
        strides=strides,
        padding=padding,
        kernel_initializer=kernel_initializer,
        kernel_regularizer=tf.keras.regularizers.l2(l2=l2_regul),
    )(c2_output)

    p4_output = p4_output + UpSampling2D(size=(2, 2))(p5_output)

    p3_output = p3_output + UpSampling2D(size=(2, 2))(p4_output)

    p2_output = p2_output + UpSampling2D(size=(2, 2))(p3_output)

    return [p2_output, p3_output, p4_output, p5_output]


def semantic_head_fpn(
    p2_output: tf.Tensor,
    p3_output: tf.Tensor,
    p4_output: tf.Tensor,
    p5_output: tf.Tensor,
) -> tf.Tensor:
    """The segmentation head added to the FPN.

    Architecture:
        ![screenshot](./images/fpn_segmentation_head.svg)

    Args:
        p2_output (tf.Tensor): Feature map coming from the `p2_output` FPN head, output stride 4.
        p3_output (tf.Tensor): Feature map coming from the `p3_output` FPN head, output stride 8.
        p4_output (tf.Tensor): Feature map coming from the `p4_output` FPN head, output stride 16.
        p5_output (tf.Tensor): Feature map coming from the `p5_output` FPN head, output stride 32.

    Returns:
        An output feature map of size $OS4$, with 512 filters.
    """

    p5_output = UpSampling2D(size=(2, 2), interpolation="bilinear")(
        conv_gn_relu(p5_output, filters=128, kernel_size=3)
    )
    p5_output = UpSampling2D(size=(2, 2), interpolation="bilinear")(
        conv_gn_relu(p5_output, filters=128, kernel_size=3)
    )
    fmap5 = UpSampling2D(size=(2, 2), interpolation="bilinear")(
        conv_gn_relu(p5_output, filters=128, kernel_size=3)
    )

    p4_output = UpSampling2D(size=(2, 2), interpolation="bilinear")(
        conv_gn_relu(p4_output, filters=128, kernel_size=3)
    )
    fmap4 = UpSampling2D(size=(2, 2), interpolation="bilinear")(
        conv_gn_relu(p4_output, filters=128, kernel_size=3)
    )

    fmap3 = UpSampling2D(size=(2, 2), interpolation="bilinear")(
        conv_gn_relu(p3_output, filters=128, kernel_size=3)
    )

    fmap2 = conv_gn_relu(p2_output, filters=128, kernel_size=3)

    return Concatenate(axis=-1)([fmap5, fmap4, fmap3, fmap2])


def get_segmentation_module(
    n_classes: int, backbone: tf.keras.Model, name: str
) -> tf.keras.Model:
    """Instantiate the segmentation head module for the segmentation task.

    Args:
        n_classes (int): Number of classes in the segmentation task.
        backbone (tf.keras.Model): CNN used as backbone/feature extractor.
        name (str): Name of the segmentation head module.

    Returns:
        A semantic segmentation model.
    """

    c_outputs = backbone.outputs

    p_outputs = FPN(*c_outputs)

    fmap = semantic_head_fpn(*p_outputs)

    fmap = Conv2D(
        filters=n_classes,
        kernel_size=(1, 1),
        strides=(1, 1),
        padding="same",
        kernel_initializer="he_uniform",
    )(fmap)

    fmap = UpSampling2D(size=(4, 4), interpolation="bilinear")(fmap)
    out = Activation("softmax")(fmap)

    return keras.Model(inputs=[backbone.inputs], outputs=[out], name=name)
