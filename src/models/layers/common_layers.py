from typing import Any, Dict

import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.keras.layers import (
    Add,
    BatchNormalization,
    Conv2D,
    DepthwiseConv2D,
    Layer,
    ReLU,
    SeparableConv2D,
)


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
    """
    Conv2D - GroupNormalization - ReLU module.

    Args:
        tensor (tf.Tensor): Input feature map of the module, size = $(H,W,C)$.
        filters (int): Number of filters used in the `Conv2D` layer.
        kernel_size (int): Size of the convolution kernels used in the `Conv2D` layer.
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
    """
    Conv2D - BatchNormalization - ReLU module.

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


def bn_relu_conv(
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
    """
    BatchNormalization - ReLU - Conv2D module.

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

    fmap = BatchNormalization(name=f"bn_{name}")(tensor)

    fmap = ReLU(name=f"relu_{name}")(fmap)
    return Conv2D(
        filters=filters,
        kernel_size=kernel_size,
        padding=padding,
        strides=strides,
        dilation_rate=dilation_rate,
        kernel_regularizer=tf.keras.regularizers.l2(l2=l2_regul),
        kernel_initializer=w_init,
        use_bias=False,
        name=f"conv_{name}",
    )(fmap)


def residual_bottleneck(
    tensor: tf.Tensor,
    filters: int,
    name: str,
    strides: int = 1,
    dilation_rate: int = 1,
    shortcut: bool = False,
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
        shortcut (bool, optional): [description]. Defaults to False.

    Returns:
        tf.Tensor: [description]
    """
    inner_filters = filters // 4

    img = BatchNormalization(name=f"bn_bottleneck_{name}")(tensor)
    out = ReLU(name=f"relu_bottleneck_{name}")(img)

    residual_fmap = out
    # main stream
    fmap = bn_relu_conv(
        out,
        inner_filters,
        kernel_size=1,
        strides=strides,
        name=f"1_{name}",
    )
    fmap = bn_relu_conv(
        fmap,
        inner_filters,
        kernel_size=3,
        strides=1,
        dilation_rate=dilation_rate,
        name=f"2_{name}",
    )
    fmap = bn_relu_conv(fmap, filters, kernel_size=1, strides=1, name=f"3_{name}")

    # shortcut
    if shortcut:
        residual_fmap = Conv2D(
            filters=filters,
            kernel_size=1,
            padding="same",
            strides=strides,
            kernel_initializer="he_normal",
            use_bias=False,
            dilation_rate=dilation_rate,
            kernel_regularizer=tf.keras.regularizers.l2(l2=l2_regul),
            name=f"shortcut_{name}",
        )(out)
        residual_fmap = BatchNormalization(name=f"shortcut_bn_{name}")(residual_fmap)

    return Add(name=f"add_{name}")([fmap, residual_fmap])


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
    """
    SeparableConv2D - BatchNormalization - ReLU module.

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


@tf.keras.utils.register_keras_serializable()
class InvertedResidualBottleneck2D(Layer):
    def __init__(
        self,
        expansion_rate: int,
        filters: int,
        strides: int,
        skip_connection: bool,
        l2_regul: float = 1e-4,
        *args,
        **kwargs,
    ) -> None:

        super().__init__(*args, **kwargs)
        self.expansion_rate = expansion_rate
        self.filters = filters
        self.strides = strides
        self.skip_connection = skip_connection
        self.l2_regul = l2_regul

        if self.strides == 2:
            assert (
                self.skip_connection == False
            ), "You can't apply skip connections with strides greater than 1."

    def build(self, input_shape) -> None:

        self.conv1 = Conv2D(
            filters=self.expansion_rate * self.filters,
            kernel_size=1,
            strides=1,
            padding="same",
            kernel_initializer="he_normal",
            use_bias=False,
        )
        self.conv2 = Conv2D(
            filters=self.filters,
            kernel_size=1,
            strides=1,
            padding="same",
            kernel_initializer="he_normal",
            use_bias=False,
        )

        self.act = ReLU(max_value=6)

        self.bn1 = BatchNormalization()
        self.bn2 = BatchNormalization()
        self.bn3 = BatchNormalization()

        self.dwconv = DepthwiseConv2D(
            kernel_size=3,
            strides=self.strides,
            padding="same",
            depth_multiplier=1,
            depthwise_initializer="he_normal",
            use_bias=False,
        )

    def call(self, inputs, training=None) -> tf.Tensor:

        fmap = self.conv1(inputs)
        fmap = self.bn1(fmap)
        fmap = self.act(fmap)

        fmap = self.dwconv(fmap)
        fmap = self.bn2(fmap)
        fmap = self.act(fmap)

        fmap = self.conv2(fmap)
        fmap = self.bn3(fmap)

        if self.skip_connection:
            fmap += inputs

        return fmap

    def get_config(self) -> Dict[str, Any]:
        config = super().get_config()
        config.update(
            {
                "expansion_rate": self.expansion_rate,
                "filters": self.filters,
                "strides": self.strides,
                "skip_connection": self.skip_connection,
                "l2_regul": self.l2_regul,
            },
        )
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)
