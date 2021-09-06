from typing import List, Union

import tensorflow as tf
from tensorflow.keras.layers import AveragePooling2D, Concatenate, Conv2D, UpSampling2D

from src.model.layers.common_layers import conv_gn_relu, sepconv_bn_relu


def upsampling(
    fmap: tf.Tensor, height: Union[int, float], width: Union[int, float]
) -> tf.Tensor:
    """[summary]

    Args:
        fmap (tf.Tensor): [description]
        height (Union[int, float]): [description]
        width (Union[int, float]): [description]

    Returns:
        tf.Tensor: [description]
    """

    h_fmap, w_fmap = fmap.shape.as_list()[1:3]
    scale = (int(height // h_fmap), int(width // w_fmap))

    return UpSampling2D(size=scale, interpolation="bilinear")(fmap)


def JPU(endpoints: List[tf.Tensor], filters: int = 256) -> tf.Tensor:
    """[summary]

    Args:
        endpoints (List[tf.Tensor]): [description]
        filters (int, optional): [description]. Defaults to 256.

    Returns:
        tf.Tensor: [description]
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
    """[summary]

    Args:
        tensor (tf.Tensor): [description]
        filters (int, optional): [description]. Defaults to 128.

    Returns:
        tf.Tensor: [description]
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
    """[summary]

    Args:
        fmap_aspp (tf.Tensor): [description]
        endpoint (tf.Tensor): [description]
        img_height (int): [description]
        img_width (int): [description]
        filters (int): [description]

    Returns:
        tf.Tensor: [description]
    """

    fmap_a = upsampling(fmap_aspp, img_height / 4, img_width / 4)

    fmap_b = conv_gn_relu(endpoint, filters=filters, kernel_size=1)

    fmap = Concatenate(axis=-1)([fmap_a, fmap_b])

    return conv_gn_relu(fmap, filters=filters, kernel_size=3)


def get_segmentation_module(
    n_classes: int, img_shape: List[int], backbone: tf.keras.Model, name: str
) -> tf.keras.Model:
    """[summary]

    Args:
        n_classes (int): [description]
        img_shape (List[int]): [description]
        backbone (tf.keras.Model): [description]
        name (str): [description]

    Returns:
        tf.keras.Model: [description]
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
