from tensorflow.keras import Model
from tensorflow.keras.layers import Activation, Conv2D, UpSampling2D

from src.models.layers.feature_pyramids import FeaturePyramidNetwork, SemanticHeadFPN


def get_segmentation_module(
    n_classes: int,
    backbone: Model,
    name: str,
) -> Model:
    """Instantiate the segmentation head module for the segmentation task.

    Args:
        n_classes (int): Number of classes in the segmentation task.
        backbone (tf.keras.Model): CNN used as backbone/feature extractor.
        name (str): Name of the segmentation head module.

    Returns:
        A semantic segmentation model.
    """

    c_outputs = backbone.outputs

    p_outputs = FeaturePyramidNetwork()(c_outputs)

    fmap = SemanticHeadFPN()(p_outputs)

    fmap = Conv2D(
        filters=n_classes,
        kernel_size=(1, 1),
        strides=(1, 1),
        padding="same",
        kernel_initializer="he_uniform",
    )(fmap)

    fmap = UpSampling2D(size=(4, 4), interpolation="bilinear")(fmap)
    out = Activation("softmax")(fmap)

    return Model(inputs=[backbone.inputs], outputs=[out], name=name)
