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
    """[summary]

    Args:
        tensor (tf.Tensor): [description]
        filters (int): [description]
        kernel_size (int): [description]
        padding (str, optional): [description]. Defaults to "same".
        strides (int, optional): [description]. Defaults to 1.
        dilation_rate (int, optional): [description]. Defaults to 1.
        w_init (str, optional): [description]. Defaults to "he_normal".
        l2_regul (float, optional): [description]. Defaults to 1e-4.

    Returns:
        tf.Tensor: [description]
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
    """[summary]

    Args:
        tensor (tf.Tensor): [description]
        filters (int): [description]
        kernel_size (int): [description]
        name (str): [description]
        padding (str, optional): [description]. Defaults to "same".
        strides (int, optional): [description]. Defaults to 1.
        dilation_rate (int, optional): [description]. Defaults to 1.
        w_init (str, optional): [description]. Defaults to "he_normal".
        l2_regul (float, optional): [description]. Defaults to 1e-4.

    Returns:
        tf.Tensor: [description]
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
    """[summary]

    Args:
        tensor (tf.Tensor): [description]
        filters (int): [description]
        kernel_size (int): [description]
        padding (str, optional): [description]. Defaults to "same".
        strides (int, optional): [description]. Defaults to 1.
        dilation_rate (int, optional): [description]. Defaults to 1.
        w_init (str, optional): [description]. Defaults to "he_normal".
        l2_regul (float, optional): [description]. Defaults to 1e-4.

    Returns:
        tf.Tensor: [description]
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
