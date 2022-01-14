import tensorflow as tf
from tensorflow.keras.layers import Activation, Conv2D, UpSampling2D

from src.models.layers.common_layers import conv_bn_relu
from src.models.layers.object_context import ASPP_OC, BaseOC


def get_segmentation_module(
    n_classes: int,
    backbone: tf.keras.Model,
    architecture: str,
    filters: int,
    name: str,
) -> tf.keras.Model:
    """Instantiate the segmentation head module for the segmentation task.

    Args:
        n_classes (int): Number of classes in the segmentation task.
        backbone (tf.keras.Model): CNN used as backbone/feature extractor.
        architecture (str): Choice of architecture for the segmentation head : `base_oc` or `aspp_ocnet`.
        filters (int): Numbers of filters used in the segmentation head.
        name (str): Name of the segmentation head module.

    Returns:
        A semantic segmentation model.
    """

    fmap = backbone.outputs[1]

    if architecture == "base_oc":
        fmap = conv_bn_relu(fmap, filters=1024, kernel_size=3, name="pre_OCP_conv")
        fmap = BaseOC(filters=filters)(fmap)
    elif architecture == "aspp_ocnet":
        fmap = ASPP_OC(filters=filters)(fmap)

    fmap = Conv2D(
        filters=n_classes,
        kernel_size=(3, 3),
        padding="same",
        kernel_regularizer=tf.keras.regularizers.l2(l2=1e-4),
        kernel_initializer="he_uniform",
        use_bias=False,
    )(fmap)

    fmap = Activation("softmax")(fmap)

    out = UpSampling2D(size=(8, 8), interpolation="bilinear")(fmap)

    return tf.keras.Model(inputs=[backbone.inputs], outputs=[out], name=name)
