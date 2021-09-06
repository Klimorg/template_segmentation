import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow import keras
from tensorflow.keras.layers import Activation, Concatenate, Conv2D, ReLU, UpSampling2D


def conv_gn_relu(
    images: tf.Tensor, filters: int = 128, l2_regul: float = 1e-4
) -> tf.Tensor:

    fmap = Conv2D(
        filters=filters,
        kernel_size=(3, 3),
        strides=(1, 1),
        padding="same",
        kernel_initializer="he_uniform",
        kernel_regularizer=tf.keras.regularizers.l2(l2=l2_regul),
    )(images)

    fmap = tfa.layers.GroupNormalization()(fmap)

    return ReLU()(fmap)


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
) -> tf.Tensor:
    """[summary]

    Args:
        c2_output (tf.Tensor): [description]
        c3_output (tf.Tensor): [description]
        c4_output (tf.Tensor): [description]
        c5_output (tf.Tensor): [description]
        filters (int, optional): [description]. Defaults to 256.
        kernel_size (tuple, optional): [description]. Defaults to (1, 1).
        strides (tuple, optional): [description]. Defaults to (1, 1).
        padding (str, optional): [description]. Defaults to "same".
        kernel_initializer (str, optional): [description]. Defaults to "he_uniform".
        l2_regul (float, optional): [description]. Defaults to 1e-4.

    Returns:
        tf.Tensor: [description]
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
    """[summary]

    Args:
        p2_output (tf.Tensor): [description]
        p3_output (tf.Tensor): [description]
        p4_output (tf.Tensor): [description]
        p5_output (tf.Tensor): [description]

    Returns:
        tf.Tensor: [description]
    """

    p5_output = UpSampling2D(size=(2, 2), interpolation="bilinear")(
        conv_gn_relu(p5_output)
    )
    p5_output = UpSampling2D(size=(2, 2), interpolation="bilinear")(
        conv_gn_relu(p5_output)
    )
    fmap5 = UpSampling2D(size=(2, 2), interpolation="bilinear")(conv_gn_relu(p5_output))

    p4_output = UpSampling2D(size=(2, 2), interpolation="bilinear")(
        conv_gn_relu(p4_output)
    )
    fmap4 = UpSampling2D(size=(2, 2), interpolation="bilinear")(conv_gn_relu(p4_output))

    fmap3 = UpSampling2D(size=(2, 2), interpolation="bilinear")(conv_gn_relu(p3_output))

    fmap2 = conv_gn_relu(p2_output)

    return Concatenate(axis=-1)([fmap5, fmap4, fmap3, fmap2])


def get_segmentation_module(
    n_classes: int, backbone: tf.keras.Model, name: str
) -> tf.keras.Model:
    """[summary]

    Args:
        n_classes (int): [description]
        backbone (tf.keras.Model): [description]
        name (str): [description]

    Returns:
        tf.keras.Model: [description]
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
