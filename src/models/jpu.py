from typing import List, Union

import tensorflow as tf
from tensorflow.keras.layers import Concatenate, Conv2D, UpSampling2D

from src.models.layers.aspp import ASPP
from src.models.layers.common_layers import conv_gn_relu
from src.models.layers.joint_pyramid_upsampling import JointPyramidUpsampling


def upsampling(
    fmap: tf.Tensor,
    height: Union[int, float],
    width: Union[int, float],
) -> tf.Tensor:
    """Upsampling module.

    Upsample features maps to the original height, width of the images in the dataset.

    Get the height, width of the input feature map, and the height width of the original
    images in the dataset to compute the scale to upsample the feature map.

    Args:
        fmap (tf.Tensor): Input feature map of the module.
        height (Union[int, float]): Height of the images in the dataset.
        width (Union[int, float]): Width of the images in the dataset.

    Returns:
        Output feature map, size $(H,W,C)$.
    """

    h_fmap, w_fmap = fmap.shape.as_list()[1:3]
    scale = (int(height // h_fmap), int(width // w_fmap))

    return UpSampling2D(size=scale, interpolation="bilinear")(fmap)


def decoder(
    fmap_aspp: tf.Tensor,
    endpoint: tf.Tensor,
    img_height: int,
    img_width: int,
    filters: int,
) -> tf.Tensor:
    """Decoder part of the segmentation model.

    Args:
        fmap_aspp (tf.Tensor): Input feature map coming from the ASPP module.
        endpoint (tf.Tensor): Input feature map coming from the backbone model, OS4.
        img_height (int): Height of the images in the dataset.
        img_width (int): Width of the images in the dataset.
        filters (int): Number of filters used in each `conv_gn_relu` layers.

    Returns:
        Output feature map.
    """

    fmap_a = upsampling(fmap_aspp, img_height / 4, img_width / 4)

    fmap_b = conv_gn_relu(endpoint, filters=filters, kernel_size=1)

    fmap = Concatenate(axis=-1)([fmap_a, fmap_b])

    return conv_gn_relu(fmap, filters=filters, kernel_size=3)


def get_segmentation_module(
    n_classes: int,
    img_shape: List[int],
    backbone: tf.keras.Model,
    name: str,
) -> tf.keras.Model:
    """Instantiate the segmentation head module for the segmentation task.

    Args:
        n_classes (int): Number of classes in the segmentation task.
        img_shape (List[int]): Input shape of the images/masks in the dataset.
        backbone (tf.keras.Model): CNN used as backbone/feature extractor.
        name (str): Name of the segmentation head module.

    Returns:
        A semantic segmentation model.
    """

    img_height, img_width = img_shape[:2]

    endpoints = backbone.outputs

    # JPU Module
    fmap = JointPyramidUpsampling()(endpoints[1:])

    # ASPP Head
    fmap = ASPP(filters=128)(fmap)

    fmap = decoder(fmap, endpoints[0], img_height, img_width, 128)

    fmap = Conv2D(
        n_classes,
        (3, 3),
        activation="softmax",
        padding="same",
        name="output_layer",
    )(fmap)
    out = upsampling(fmap, img_height, img_width)

    return tf.keras.Model(inputs=backbone.input, outputs=out, name=name)
