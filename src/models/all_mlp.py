import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Activation, Concatenate, Dense, UpSampling2D


def get_segmentation_module(
    units: int,
    n_classes: int,
    backbone: Model,
    name: str,
) -> Model:
    """Instantiate the segmentation head module for the segmentation task.

    Args:
        n_classes (int): Number of classes in the segmentation task.
        backbone (Model): CNN used as backbone/feature extractor.
        name (str): Name of the segmentation head module.

    Returns:
        A semantic segmentation model.
    """

    l2_regul = 1e-4
    bil = "bilinear"
    he = "he_uniform"

    os4_output, os8_output, os16_output, os32_output = backbone.outputs

    fmap1 = Dense(
        units,
        kernel_initializer=he,
        kernel_regularizer=tf.keras.regularizers.l2(l2=l2_regul),
    )(os4_output)

    fmap2 = Dense(
        units,
        kernel_initializer=he,
        kernel_regularizer=tf.keras.regularizers.l2(l2=l2_regul),
    )(os8_output)
    fmap2 = UpSampling2D(size=(2, 2), interpolation=bil)(fmap2)

    fmap3 = Dense(
        units,
        kernel_initializer=he,
        kernel_regularizer=tf.keras.regularizers.l2(l2=l2_regul),
    )(os16_output)
    fmap3 = UpSampling2D(size=(4, 4), interpolation=bil)(fmap3)

    fmap4 = Dense(
        units,
        kernel_initializer=he,
        kernel_regularizer=tf.keras.regularizers.l2(l2=l2_regul),
    )(os32_output)
    fmap4 = UpSampling2D(size=(8, 8), interpolation=bil)(fmap4)

    fmap = Concatenate(axis=-1)([fmap1, fmap2, fmap3, fmap4])

    fmap = Dense(
        units,
        kernel_initializer=he,
        kernel_regularizer=tf.keras.regularizers.l2(l2=l2_regul),
    )(fmap)

    fmap = Dense(
        n_classes,
        kernel_initializer=he,
        kernel_regularizer=tf.keras.regularizers.l2(l2=l2_regul),
    )(fmap)

    fmap = UpSampling2D(size=(4, 4), interpolation=bil)(fmap)
    out = Activation("softmax")(fmap)

    return Model(inputs=[backbone.inputs], outputs=[out], name=name)
