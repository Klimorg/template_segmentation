import tensorflow as tf
from tensorflow.keras.layers import BatchNormalization, Conv2D, ReLU


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
