from typing import List, Tuple

import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import (
    Activation,
    BatchNormalization,
    Concatenate,
    Conv2D,
    Conv2DTranspose,
    Dropout,
    Input,
    MaxPool2D,
    ReLU,
)

initializer = "he_normal"
padding = "same"


def bn_relu_conv(
    tensor: tf.Tensor,
    filters: int,
    kernel_size: Tuple[int, int],
    dropout_rate: float = 0.2,
) -> tf.Tensor:
    fmap = BatchNormalization()(tensor)
    fmap = ReLU()(fmap)
    fmap = Conv2D(
        filters,
        kernel_size=kernel_size,
        strides=1,
        padding=padding,
        kernel_initializer=initializer,
        kernel_regularizer=tf.keras.regularizers.l2(l2=1e-4),
    )(fmap)

    return Dropout(dropout_rate)(fmap)


def dense_layer(tensor: tf.Tensor, repetitions: int, filters: int) -> tf.Tensor:

    skip_connection = []

    for _ in range(repetitions - 1):
        out_conv = bn_relu_conv(tensor, filters, (3, 3))
        tensor = Concatenate()([tensor, out_conv])
        skip_connection.append(out_conv)

    out_conv = bn_relu_conv(tensor, filters, (3, 3))
    skip_connection.append(out_conv)
    # pour les avoir dans le bon ordre [x_{l-1}, x_{l-2}, ..., x_{0}]
    skip_connection = skip_connection[::-1]

    return Concatenate()(skip_connection)


def transition_down(tensor: tf.Tensor, filters: int):
    fmap = bn_relu_conv(tensor, filters, (1, 1))

    return MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding=padding)(fmap)


def transition_up(tensor: tf.Tensor, filters: int) -> tf.Tensor:
    return Conv2DTranspose(
        filters=filters,
        kernel_size=(3, 3),
        strides=(2, 2),
        padding=padding,
        kernel_initializer=initializer,
        kernel_regularizer=tf.keras.regularizers.l2(l2=1e-4),
    )(tensor)


def get_segmentation_model(
    filters: int,
    nb_layers: List[int],
    img_shape: Tuple[int, int, int],
    n_classes: int,
    nb_blocks: int = 5,
    nb_filters: int = 48,
) -> tf.keras.Model:
    """Architecture for the FCDenseNEt "tiramisu" segmentation model.

    Note:
        [The One Hundred Layers Tiramisu: Fully Convolutional DenseNets for Semantic Segmentation](https://arxiv.org/abs/1611.09326)

    Args:
        filters (int): The growth-rate of the convolutionnals layers in the transition_down and transition_up layers.
        nb_layers (List[int]): The list of the number of layers in each dense block.
        img_shape (Tuple[int, int, int]): The shape of the image
        n_classes (int): The number of classes in the segmentation problem, usually $n+1$ : $n$ classes plus the background.
        nb_blocks (int, optional): The number of dense blocks in the encoder part and the decoder part of the architecture. Defaults to 5.
        nb_filters (int, optional): The number of filters in the first convolution, also used to initiate the growth of the number of filters in the transition_down and transition_up layers.. Defaults to 48.

    Returns:
        tf.keras.Model: A semantic segmentation neural network model.
    """

    upsampling_fm = []

    # input block
    input_img = Input(shape=img_shape)
    fmap = Conv2D(
        nb_filters,
        kernel_size=3,
        strides=1,
        padding=padding,
        kernel_initializer=initializer,
    )(input_img)

    # downsampling blocks
    for idx_down in range(nb_blocks):
        out_conv = dense_layer(fmap, nb_layers[idx_down], filters)
        out_conv = Concatenate()([out_conv, fmap])
        nb_filters += filters * nb_layers[idx_down]

        upsampling_fm.append(out_conv)
        upsampling = upsampling_fm[::-1]
        fmap = transition_down(out_conv, nb_filters)

    # Bottleneck
    fmap = dense_layer(fmap, nb_layers[5], filters)
    nb_filters = filters * nb_layers[5]

    # upsampling path
    for idx_up in range(nb_blocks):

        fmap = transition_up(fmap, nb_filters)
        fmap = Concatenate()([fmap, upsampling[idx_up]])

        if nb_layers[6 + idx_up] == nb_layers[-1]:
            out_conv = dense_layer(fmap, nb_layers[6 + idx_up], filters)
            fmap = Concatenate()([out_conv, fmap])
        else:
            fmap = dense_layer(fmap, nb_layers[6 + idx_up], filters)
            nb_filters = filters * nb_layers[6 + idx_up]

    # output block
    fmap = Conv2D(
        n_classes,
        kernel_size=1,
        strides=(1, 1),
        padding=padding,
        kernel_initializer=initializer,
    )(fmap)
    output = Activation("softmax")(fmap)

    return Model(input_img, output)
