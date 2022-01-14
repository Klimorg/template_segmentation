import tensorflow as tf
from tensorflow.keras.layers import Activation, Concatenate, Conv2D, UpSampling2D

from src.models.layers.common_layers import conv_bn_relu
from src.models.layers.shared_kernels import KSAConv2D


def decoder(fmap1, fmap2, filters):
    """[summary]

    Args:
        fmap1 ([type]): [description]
        fmap2 ([type]): [description]
        filters ([type]): [description]

    Returns:
        [type]: [description]
    """

    fmap1 = UpSampling2D(size=(4, 4), interpolation="bilinear")(fmap1)

    fmap2 = conv_bn_relu(tensor=fmap2, filters=filters, kernel_size=1, name="decoder1")

    fmap = Concatenate(axis=-1)([fmap1, fmap2])

    return conv_bn_relu(tensor=fmap, filters=filters, kernel_size=3, name="decoder2")


def get_segmentation_module(
    n_classes: int,
    backbone: tf.keras.Model,
    name: str,
    ksac_filters: int,
    decoder_filters: int,
) -> tf.keras.Model:
    """Instantiate the segmentation head module for the segmentation task.

    Args:
        n_classes (int): Number of classes in the segmentation task.
        backbone (tf.keras.Model): CNN used as backbone/feature extractor.
        name (str): Name of the segmentation head module.

    Returns:
        A semantic segmentation model.
    """

    c2_output, _, c4_output, _ = backbone.outputs

    fm = KSAConv2D(filters=ksac_filters)(c4_output)

    fmap = decoder(fm, c2_output, filters=decoder_filters)

    fmap = UpSampling2D(size=(4, 4), interpolation="bilinear")(fmap)

    fmap = Conv2D(
        filters=n_classes,
        kernel_size=(3, 3),
        strides=(1, 1),
        padding="same",
        kernel_initializer="he_uniform",
    )(fmap)

    out = Activation("softmax")(fmap)

    return tf.keras.Model(inputs=[backbone.inputs], outputs=[out], name=name)
