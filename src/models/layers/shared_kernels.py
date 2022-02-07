from typing import Any, Dict, List, Tuple

import tensorflow as tf
from tensorflow.keras.layers import (
    AveragePooling2D,
    BatchNormalization,
    Concatenate,
    Conv2D,
    Layer,
    ReLU,
    UpSampling2D,
)


@tf.keras.utils.register_keras_serializable()
class SharedDilatedConv(Layer):
    """
    Description of SharedDilatedConv

    Attributes:
        w (type): Weight of the `tf.nn.conv2d` operation.
        b (type): Bias of the `tf.nn.conv2d` operation.
        dilation_rates (type): Dilations rates used for the `tf.nn.conv2d` operation.
        relu (type):

    Inheritance:
        Layer
    """

    def __init__(
        self,
        filters: int,
        kernel_size: Tuple[int, int],
        strides: Tuple[int, int],
        kernel_initializer: str,
        use_bias: bool,
        bias_initializer=None,
        *args,
        **kwargs,
    ) -> None:
        """[summary]

        Args:
            filters (int, optional): Number of filters in each `tf.nn.conv2d` operations.
            kernel_size (tuple, optional): Size of the convolution kernels in each `tf.nn.conv2d` operation.
            strides (tuple, optional): Stride parameter in each `tf.nn.conv2d` operation.
            kernel_initializer (str, optional): Kernel initialization method used in each `tf.nn.conv2d` operation.
            use_bias (bool): Determine whetther or not use bias.
            bias_initializer ([type], optional): Bias initialization method used in each `tf.nn.conv2d` operation.. Defaults to None.
        """
        super().__init__(*args, **kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        self.kernel_initializer = kernel_initializer
        self.bias_initializer = bias_initializer
        self.strides = strides
        self.use_bias = use_bias
        self.kernel = None
        self.b = None
        self.dilation_rates = [6, 12, 18]

        self.relu = ReLU()

    def build(self, input_shape) -> None:

        *_, n_channels = input_shape
        self.kernel = self.add_weight(
            name="kernel",
            shape=(*self.kernel_size, n_channels, self.filters),
            initializer=self.kernel_initializer,
            trainable=True,
            dtype="float32",
        )
        if self.use_bias:
            self.bias = self.add_weight(
                name="bias",
                shape=(self.filters,),
                initializer=self.bias_initializer,
                trainable=True,
                dtype="float32",
            )
        else:
            self.bias = None

        self.bn1 = BatchNormalization()
        self.bn2 = BatchNormalization()
        self.bn3 = BatchNormalization()

    def call(self, inputs, training=None) -> tf.Tensor:

        x1 = tf.nn.conv2d(
            inputs,
            self.kernel,
            padding="SAME",
            strides=self.strides,
            dilations=self.dilation_rates[0],
        )
        if self.use_bias:
            x1 = x1 + self.bias
        x1 = self.relu(self.bn1(x1))

        x2 = tf.nn.conv2d(
            inputs,
            self.kernel,
            padding="SAME",
            strides=self.strides,
            dilations=self.dilation_rates[1],
        )
        if self.use_bias:
            x2 = x2 + self.bias
        x2 = self.relu(self.bn2(x2))

        x3 = tf.nn.conv2d(
            inputs,
            self.kernel,
            padding="SAME",
            strides=self.strides,
            dilations=self.dilation_rates[2],
        )
        if self.use_bias:
            x3 = x3 + self.bias
        x3 = self.relu(self.bn2(x3))

        return x1 + x2 + x3

    def get_config(self) -> Dict[str, Any]:

        base_config = super().get_config()
        return {
            **base_config,
            "filters": self.filters,
            "kernel_size": self.kernel_size,
            "strides": self.strides,
            "kernel_initializer": self.kernel_initializer,
            "bias_initializer": self.bias_initializer,
            "use_bias": self.use_bias,
        }

    @classmethod
    def from_config(cls, config):
        return cls(**config)


@tf.keras.utils.register_keras_serializable()
class KSAConv2D(Layer):
    """
    Description of KSAConv2D

    Attributes:
        relu (type): `ReLU` function.
        conv1 (type): `Conv2D` layer.
        conv2 (type): `Conv2D` layer.
        conv3 (type): `Conv2D` layer.
        concat (type): `Concatenate` layer.

    Inheritance:
        tf.keras.layers.Layer:
    """

    def __init__(
        self,
        filters: int,
        *args: List[Any],
        **kwargs: Dict[Any, Any],
    ) -> None:
        """Initialization of the class.

        Args:
            filters (int): Number of filters in each `Conv2D` and `SharedDilatedConv` layers.
        """
        super().__init__(*args, **kwargs)
        self.filters = filters

        self.relu = ReLU()
        self.conv1 = Conv2D(
            filters=filters,
            kernel_size=1,
            strides=1,
            padding="same",
            kernel_initializer="he_uniform",
            use_bias=False,
        )
        self.conv2 = Conv2D(
            filters=filters,
            kernel_size=1,
            strides=1,
            padding="same",
            kernel_initializer="he_uniform",
            use_bias=False,
        )
        self.conv3 = Conv2D(
            filters=filters,
            kernel_size=1,
            strides=1,
            padding="same",
            kernel_initializer="he_uniform",
            use_bias=False,
        )

        self.concat = Concatenate(axis=-1)

    def build(self, input_shape) -> None:
        _, height, width, *_ = input_shape

        self.shared_conv = SharedDilatedConv(
            filters=self.filters,
            kernel_size=(3, 3),
            strides=(1, 1),
            kernel_initializer="he_uniform",
            use_bias=False,
        )

        self.bn1 = BatchNormalization()
        self.bn2 = BatchNormalization()

        self.pooling = AveragePooling2D(pool_size=(height, width))
        self.upsample = UpSampling2D(size=(height, width), interpolation="bilinear")

    def call(self, inputs, training=None) -> tf.Tensor:
        fmap1 = self.conv1(inputs)
        fmap1 = self.bn1(fmap1)

        fmap2 = self.shared_conv(inputs)

        fmap3 = self.pooling(inputs)
        fmap3 = self.conv2(fmap3)
        fmap3 = self.upsample(fmap3)

        fmap = self.concat([fmap1, fmap2, fmap3])
        fmap = self.conv3(fmap)
        fmap = self.bn2(fmap)

        return self.relu(fmap)

    def get_config(self) -> Dict[str, Any]:
        base_config = super().get_config()
        return {
            **base_config,
            "filters": self.filters,
        }

    @classmethod
    def from_config(cls, config):
        return cls(**config)
