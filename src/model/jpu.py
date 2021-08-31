from typing import List, Union

import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow import keras
from tensorflow.keras.layers import (
    AveragePooling2D,
    BatchNormalization,
    Concatenate,
    Conv2D,
    ReLU,
    SeparableConv2D,
    UpSampling2D,
)

# def get_backbone2(img_shape: List[int]) -> tf.keras.Model:
#     """Builds ResNet50 with pre-trained imagenet weights"""

#     backbone = keras.applications.ResNet50(include_top=False, input_shape=img_shape)

#     endpoint_names = [
#         "conv2_block3_out",
#         "conv3_block4_out",
#         "conv4_block6_out",
#         "conv5_block3_out",
#     ]

#     c2_output, c3_output, c4_output, c5_output = [
#         backbone.get_layer(layer_name).output for layer_name in endpoint_names
#     ]

#     return keras.Model(
#         inputs=[backbone.inputs], outputs=[c2_output, c3_output, c4_output, c5_output]
#     )


# def get_backbone(img_shape: List[int]) -> tf.keras.Model:
#     """Builds ResNet50 with pre-trained imagenet weights"""

#     backbone = keras.applications.ResNet101(
#         include_top=False, input_shape=img_shape, weights="imagenet"
#     )
#     endpoint_names = [
#         "conv2_block3_out",
#         "conv3_block4_out",
#         "conv4_block23_out",
#         "conv5_block3_out",
#     ]

#     c2_output, c3_output, c4_output, c5_output = [
#         backbone.get_layer(layer_name).output for layer_name in endpoint_names
#     ]

#     return keras.Model(
#         inputs=[backbone.inputs], outputs=[c2_output, c3_output, c4_output, c5_output]
#     )


def conv_bn_relu(
    tensor: tf.Tensor,
    num_filters: int,
    kernel_size: int,
    padding: str = "same",
    strides: int = 1,
    dilation_rate: int = 1,
    w_init: str = "he_normal",
) -> tf.Tensor:

    fmap = Conv2D(
        filters=num_filters,
        kernel_size=kernel_size,
        padding=padding,
        strides=strides,
        dilation_rate=dilation_rate,
        kernel_regularizer=tf.keras.regularizers.l2(l2=1e-4),
        kernel_initializer=w_init,
        use_bias=False,
    )(tensor)

    fmap = tfa.layers.GroupNormalization()(fmap)

    # fmap = BatchNormalization()(fmap)

    return ReLU()(fmap)


def sepconv_bn_relu(
    tensor: tf.Tensor,
    num_filters: int,
    kernel_size: int,
    padding: str = "same",
    strides: int = 1,
    dilation_rate: int = 1,
    w_init: str = "he_normal",
) -> tf.Tensor:

    fmap = SeparableConv2D(
        filters=num_filters,
        depth_multiplier=1,
        kernel_size=kernel_size,
        padding=padding,
        strides=strides,
        dilation_rate=dilation_rate,
        depthwise_initializer=w_init,
        kernel_regularizer=tf.keras.regularizers.l2(l2=1e-4),
        use_bias=False,
    )(tensor)

    fmap = BatchNormalization()(fmap)

    return ReLU()(fmap)


def upsampling(
    fmap: tf.Tensor, height: Union[int, float], width: Union[int, float]
) -> tf.Tensor:

    h_fmap, w_fmap = fmap.shape.as_list()[1:3]
    scale = (int(height // h_fmap), int(width // w_fmap))

    return UpSampling2D(size=scale, interpolation="bilinear")(fmap)


def JPU(endpoints: List[tf.Tensor], filters: int = 256) -> tf.Tensor:

    _, c3_output, c4_output, c5_output = endpoints

    height, width = c3_output.shape.as_list()[1:3]

    fmap3 = conv_bn_relu(c3_output, filters, 3)

    fmap4 = conv_bn_relu(c4_output, filters, 3)
    fmap4 = upsampling(fmap4, height, width)

    fmap5 = conv_bn_relu(c5_output, filters, 3)
    fmap5 = upsampling(fmap5, height, width)

    fmap = Concatenate(axis=-1)([fmap3, fmap4, fmap5])

    sep_fmap1 = sepconv_bn_relu(fmap, filters, 3, dilation_rate=1)
    sep_fmap2 = sepconv_bn_relu(fmap, filters, 3, dilation_rate=2)
    sep_fmap4 = sepconv_bn_relu(fmap, filters, 3, dilation_rate=4)
    sep_fmap8 = sepconv_bn_relu(fmap, filters, 3, dilation_rate=8)

    fmap = Concatenate(axis=-1)([sep_fmap1, sep_fmap2, sep_fmap4, sep_fmap8])

    return conv_bn_relu(fmap, num_filters=filters, kernel_size=1)


def ASPP(tensor: tf.Tensor, filters: int = 128) -> tf.Tensor:

    height, width = tensor.shape.as_list()[1:3]

    fmap_pool = AveragePooling2D(pool_size=(height, width), name="average_pooling")(
        tensor
    )
    fmap_pool = conv_bn_relu(fmap_pool, num_filters=filters, kernel_size=1)
    fmap_pool = upsampling(fmap_pool, height, width)

    fmap1 = conv_bn_relu(tensor, num_filters=filters, kernel_size=1, dilation_rate=1)
    fmap6 = conv_bn_relu(tensor, num_filters=filters, kernel_size=3, dilation_rate=6)
    fmap12 = conv_bn_relu(tensor, num_filters=filters, kernel_size=3, dilation_rate=12)
    fmap18 = conv_bn_relu(tensor, num_filters=filters, kernel_size=3, dilation_rate=18)

    fmap = Concatenate(axis=-1)([fmap_pool, fmap1, fmap6, fmap12, fmap18])

    return conv_bn_relu(fmap, num_filters=filters, kernel_size=1)


def decoder(
    fmap_aspp: tf.Tensor,
    endpoint: tf.Tensor,
    img_height: int,
    img_width: int,
    filters: int,
) -> tf.Tensor:

    fmap_a = upsampling(fmap_aspp, img_height / 4, img_width / 4)

    fmap_b = conv_bn_relu(endpoint, num_filters=filters, kernel_size=1)

    fmap = Concatenate(axis=-1)([fmap_a, fmap_b])

    return conv_bn_relu(fmap, num_filters=filters, kernel_size=3)


def get_jpu(
    n_classes: int, img_shape: List[int], backbone: tf.keras.Model
) -> tf.keras.Model:

    img_height, img_width = img_shape[:2]

    # backbone = get_backbone2(img_shape=img_shape)
    endpoints = backbone.outputs

    # JPU Module
    fmap = JPU(endpoints)

    # ASPP Head
    fmap = ASPP(fmap)

    fmap = decoder(fmap, endpoints[0], img_height, img_width, 128)

    fmap = upsampling(fmap, img_height, img_width)
    fmap = Conv2D(n_classes, (1, 1), activation="softmax", name="output_layer")(fmap)

    return tf.keras.Model(inputs=backbone.input, outputs=fmap, name="JPU")


if __name__ == "__main__":
    model = get_jpu(n_classes=4, img_shape=[256, 256, 3])

    model.summary()
