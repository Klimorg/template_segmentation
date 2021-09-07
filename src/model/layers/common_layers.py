import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.keras.layers import BatchNormalization, Conv2D, ReLU, SeparableConv2D


def conv_gn_relu(
    tensor: tf.Tensor,
    filters: int,
    kernel_size: int,
    padding: str = "same",
    strides: int = 1,
    dilation_rate: int = 1,
    w_init: str = "he_normal",
    l2_regul: float = 1e-4,
) -> tf.Tensor:
    """Conv2D - GroupNormalization - ReLU module.

    Args:
        tensor (tf.Tensor): Input feature map of the module, size = $(H,W,C)$.
        filters (int): Number of filters used in the `Conv2D` layer.
        kernel_size (int): Size of the convolution kernels used in the `Conv2D` layer.
        padding (str, optional): Padding parameter of the `Conv2D` layer.. Defaults to "same".
        strides (int, optional): Strides parameter of the `Conv2D` layer. Defaults to 1.
        dilation_rate (int, optional): Dilation rate of the `Conv2D` layer. Defaults to 1.
        w_init (str, optional): Kernel initialization method used in th `Conv2D` layer. Defaults to "he_normal".
        l2_regul (float, optional): Value of the constraint used for the
            $L_2$ regularization. Defaults to 1e-4.

    Returns:
        Output feature map, size = $(H,W,C)$.
    """

    fmap = Conv2D(
        filters=filters,
        kernel_size=kernel_size,
        padding=padding,
        strides=strides,
        dilation_rate=dilation_rate,
        kernel_regularizer=tf.keras.regularizers.l2(l2=l2_regul),
        kernel_initializer=w_init,
        use_bias=False,
    )(tensor)

    fmap = tfa.layers.GroupNormalization()(fmap)

    return ReLU()(fmap)


def conv_bn_relu(
    tensor: tf.Tensor,
    filters: int,
    kernel_size: int,
    name: str,
    padding: str = "same",
    strides: int = 1,
    dilation_rate: int = 1,
    w_init: str = "he_normal",
    l2_regul: float = 1e-4,
) -> tf.Tensor:
    """Conv2D - BatchNormalization - ReLU module.

    Args:
        tensor (tf.Tensor): Input feature map of the module, size = $(H,W,C)$.
        filters (int): Number of filters used in the `Conv2D` layer.
        kernel_size (int): Size of the convolution kernels used in the `Conv2D` layer.
        name (str): Name of the module.
        padding (str, optional): Padding parameter of the `Conv2D` layer. Defaults to "same".
        strides (int, optional): Strides parameter of the `Conv2D` layer. Defaults to 1.
        dilation_rate (int, optional): Dilation rate of the `Conv2D` layer. Defaults to 1.
        w_init (str, optional): Kernel initialization method used in th `Conv2D` layer. Defaults to "he_normal".
        l2_regul (float, optional): Value of the constraint used for the
            $L_2$ regularization. Defaults to 1e-4.

    Returns:
        Output feature map, size = $(H,W,C)$.
    """

    fmap = Conv2D(
        filters=filters,
        kernel_size=kernel_size,
        padding=padding,
        strides=strides,
        dilation_rate=dilation_rate,
        kernel_regularizer=tf.keras.regularizers.l2(l2=l2_regul),
        kernel_initializer=w_init,
        use_bias=False,
        name=f"{name}",
    )(tensor)

    fmap = BatchNormalization(name=f"bn_{name}")(fmap)

    return ReLU(name=f"relu_{name}")(fmap)


def sepconv_bn_relu(
    tensor: tf.Tensor,
    filters: int,
    kernel_size: int,
    padding: str = "same",
    strides: int = 1,
    dilation_rate: int = 1,
    w_init: str = "he_normal",
    l2_regul: float = 1e-4,
) -> tf.Tensor:
    """SeparableConv2D - BatchNormalization - ReLU module.

    Args:
        tensor (tf.Tensor): Input feature map of the module, size = $(H,W,C)$.
        filters (int): Number of filters used in the `SeparableConv2D` layer.
        kernel_size (int): Size of the convolution kernels used in the `SeparableConv2D` layer.
        padding (str, optional): Padding parameter of the `SeparableConv2D` layer. Defaults to "same".
        strides (int, optional): Strides parameter of the `SeparableConv2D` layer. Defaults to 1.
        dilation_rate (int, optional): Dilation rate of the `SeparableConv2D` layer. Defaults to 1.
        w_init (str, optional): Kernel initialization method used in th `SeparableConv2D` layer. Defaults to "he_normal".
        l2_regul (float, optional): Value of the constraint used for the
            $L_2$ regularization. Defaults to 1e-4.

    Returns:
        Output feature map, size = $(H,W,C)$.
    """

    fmap = SeparableConv2D(
        filters=filters,
        depth_multiplier=1,
        kernel_size=kernel_size,
        padding=padding,
        strides=strides,
        dilation_rate=dilation_rate,
        depthwise_initializer=w_init,
        kernel_regularizer=tf.keras.regularizers.l2(l2=l2_regul),
        use_bias=False,
    )(tensor)

    fmap = BatchNormalization()(fmap)

    return ReLU()(fmap)
