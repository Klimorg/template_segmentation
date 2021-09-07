from typing import List, Union

import tensorflow as tf
from tensorflow.keras.layers import AveragePooling2D, Concatenate, Conv2D, UpSampling2D

from src.model.layers.common_layers import conv_gn_relu, sepconv_bn_relu


def upsampling(
    fmap: tf.Tensor, height: Union[int, float], width: Union[int, float]
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


def JPU(endpoints: List[tf.Tensor], filters: int = 256) -> tf.Tensor:
    """Joint Pyramid Upsampling module.

    Architecture:
        ![screenshot](./images/jpu_details.svg)

    Args:
        endpoints (List[tf.Tensor]): OS8, OS16, and OS32 endpoint feature maps of the backbone.
        filters (int, optional): Number of filters used in each `conv_gn_relu` and `sepconv_bn_relu` layers. Defaults to 256.

    Returns:
        Output feature map, $(H,W,C)$.
    """

    _, c3_output, c4_output, c5_output = endpoints

    height, width = c3_output.shape.as_list()[1:3]

    fmap3 = conv_gn_relu(c3_output, filters, 3)

    fmap4 = conv_gn_relu(c4_output, filters, 3)
    fmap4 = upsampling(fmap4, height, width)

    fmap5 = conv_gn_relu(c5_output, filters, 3)
    fmap5 = upsampling(fmap5, height, width)

    fmap = Concatenate(axis=-1)([fmap3, fmap4, fmap5])

    sep_fmap1 = sepconv_bn_relu(fmap, filters=filters, kernel_size=3, dilation_rate=1)
    sep_fmap2 = sepconv_bn_relu(fmap, filters=filters, kernel_size=3, dilation_rate=2)
    sep_fmap4 = sepconv_bn_relu(fmap, filters=filters, kernel_size=3, dilation_rate=4)
    sep_fmap8 = sepconv_bn_relu(fmap, filters=filters, kernel_size=3, dilation_rate=8)

    fmap = Concatenate(axis=-1)([sep_fmap1, sep_fmap2, sep_fmap4, sep_fmap8])

    return conv_gn_relu(fmap, filters=filters, kernel_size=1)


def ASPP(tensor: tf.Tensor, filters: int = 128) -> tf.Tensor:
    """Atrous Spatial Pyramid Pooling module.

    Args:
        tensor (tf.Tensor): Input feature map.
        filters (int, optional):  Number of filters used in each `conv_gn_relu` layers. Defaults to 128.

    Returns:
        Output feature map.
    """

    height, width = tensor.shape.as_list()[1:3]

    fmap_pool = AveragePooling2D(pool_size=(height, width), name="average_pooling")(
        tensor
    )
    fmap_pool = conv_gn_relu(fmap_pool, filters=filters, kernel_size=1)
    fmap_pool = upsampling(fmap_pool, height, width)

    fmap1 = conv_gn_relu(tensor, filters=filters, kernel_size=1, dilation_rate=1)
    fmap6 = conv_gn_relu(tensor, filters=filters, kernel_size=3, dilation_rate=6)
    fmap12 = conv_gn_relu(tensor, filters=filters, kernel_size=3, dilation_rate=12)
    fmap18 = conv_gn_relu(tensor, filters=filters, kernel_size=3, dilation_rate=18)

    fmap = Concatenate(axis=-1)([fmap_pool, fmap1, fmap6, fmap12, fmap18])

    return conv_gn_relu(fmap, filters=filters, kernel_size=1)


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
    n_classes: int, img_shape: List[int], backbone: tf.keras.Model, name: str
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
    fmap = JPU(endpoints)

    # ASPP Head
    fmap = ASPP(fmap)

    fmap = decoder(fmap, endpoints[0], img_height, img_width, 128)

    fmap = Conv2D(
        n_classes, (3, 3), activation="softmax", padding="same", name="output_layer"
    )(fmap)
    out = upsampling(fmap, img_height, img_width)

    return tf.keras.Model(inputs=backbone.input, outputs=out, name=name)
