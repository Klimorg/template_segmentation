from typing import Any, Dict, List

import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.keras.layers import (
    BatchNormalization,
    Concatenate,
    Conv2D,
    Layer,
    ReLU,
    SeparableConv2D,
    UpSampling2D,
)
from tensorflow.keras.models import Sequential


@tf.keras.utils.register_keras_serializable()
class JointPyramidUpsampling(Layer):
    """
    Description of JointPyramidUpsampling

    Joint Pyramid Upsampling module.

    Architecture:
        ![screenshot](./images/jpu_details.svg)


    Attributes:
        conv1 (type): `Conv2D-GroupNormalization-ReLU` layer.
        conv2 (type): `Conv2D-GroupNormalization-ReLU` layer.
        conv3 (type): `Conv2D-GroupNormalization-ReLU` layer.
        conv4 (type): `Conv2D-GroupNormalization-ReLU` layer.
        upsample (type): `Upsampling2D` layer.
        concat (type): `Concatenate` layer.
        sepconv1 (type): `SeparableConv2D-GroupNormalization-ReLU` layer.
        sepconv2 (type): `SeparableConv2D-GroupNormalization-ReLU` layer.
        sepconv4 (type): `SeparableConv2D-GroupNormalization-ReLU` layer.
        sepconv8 (type): `SeparableConv2D-GroupNormalization-ReLU` layer.

    Inheritance:
        tf.keras.layers.Layer:

    Returns:
        tf.Tensor: Output feature map, $(H,W,C)$.
    """

    def __init__(
        self,
        filters: int = 256,
        kernel_size: int = 3,
        strides: int = 1,
        padding: str = "same",
        kernel_initializer: str = "he_uniform",
        l2_regul: float = 1e-4,
        *args: List[Any],
        **kwargs: Dict[str, Any],
    ) -> None:
        """Initilization of the class.

        Args:
            filters (int, optional): Number of filters in each `Conv2D` layers. Defaults to 256.
            kernel_size (tuple, optional): Size of the convolution kernels in each `Conv2D` layers.
                Defaults to 3.
            strides (tuple, optional): Stride parameter in each `Conv2D` layers. Defaults to 1.
            padding (str, optional): Paddinf parameter in each `Conv2D` layers. Defaults to "same".
            kernel_initializer (str, optional): Kernel initialization method used in each `Conv2D` layers.
                Defaults to "he_uniform".
            l2_regul (float, optional): Value of the constraint used for the
                $L_2$ regularization. Defaults to 1e-4.
        """
        super().__init__(*args, **kwargs)

        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.kernel_initializer = kernel_initializer
        self.l2_regul = l2_regul

    def build(self, input_shape) -> None:
        self.conv1 = Sequential(
            [
                Conv2D(
                    filters=self.filters,
                    kernel_size=self.kernel_size,
                    padding=self.padding,
                    use_bias=False,
                    kernel_initializer=self.kernel_initializer,
                    kernel_regularizer=tf.keras.regularizers.l2(l2=self.l2_regul),
                ),
                tfa.layers.GroupNormalization(),
                ReLU(),
            ],
        )
        self.conv2 = Sequential(
            [
                Conv2D(
                    filters=self.filters,
                    kernel_size=self.kernel_size,
                    padding=self.padding,
                    use_bias=False,
                    kernel_initializer=self.kernel_initializer,
                    kernel_regularizer=tf.keras.regularizers.l2(l2=self.l2_regul),
                ),
                tfa.layers.GroupNormalization(),
                ReLU(),
            ],
        )
        self.conv3 = Sequential(
            [
                Conv2D(
                    filters=self.filters,
                    kernel_size=self.kernel_size,
                    padding=self.padding,
                    use_bias=False,
                    kernel_initializer=self.kernel_initializer,
                    kernel_regularizer=tf.keras.regularizers.l2(l2=self.l2_regul),
                ),
                tfa.layers.GroupNormalization(),
                ReLU(),
            ],
        )
        self.conv4 = Sequential(
            [
                Conv2D(
                    filters=self.filters,
                    kernel_size=self.kernel_size,
                    padding=self.padding,
                    use_bias=False,
                    kernel_initializer=self.kernel_initializer,
                    kernel_regularizer=tf.keras.regularizers.l2(l2=self.l2_regul),
                ),
                tfa.layers.GroupNormalization(),
                ReLU(),
            ],
        )

        self.upsample = UpSampling2D(size=(2, 2), interpolation="bilinear")
        self.upsample4 = UpSampling2D(size=(4, 4), interpolation="bilinear")
        self.concat = Concatenate(axis=-1)

        self.sepconv1 = Sequential(
            [
                SeparableConv2D(
                    filters=self.filters,
                    depth_multiplier=1,
                    kernel_size=self.kernel_size,
                    padding=self.padding,
                    strides=self.strides,
                    dilation_rate=1,
                    depthwise_initializer=self.kernel_initializer,
                    kernel_regularizer=tf.keras.regularizers.l2(l2=self.l2_regul),
                    use_bias=False,
                ),
                BatchNormalization(),
                ReLU(),
            ],
        )
        self.sepconv2 = Sequential(
            [
                SeparableConv2D(
                    filters=self.filters,
                    depth_multiplier=1,
                    kernel_size=self.kernel_size,
                    padding=self.padding,
                    strides=self.strides,
                    dilation_rate=1,
                    depthwise_initializer=self.kernel_initializer,
                    kernel_regularizer=tf.keras.regularizers.l2(l2=self.l2_regul),
                    use_bias=False,
                ),
                BatchNormalization(),
                ReLU(),
            ],
        )
        self.sepconv4 = Sequential(
            [
                SeparableConv2D(
                    filters=self.filters,
                    depth_multiplier=1,
                    kernel_size=self.kernel_size,
                    padding=self.padding,
                    strides=self.strides,
                    dilation_rate=1,
                    depthwise_initializer=self.kernel_initializer,
                    kernel_regularizer=tf.keras.regularizers.l2(l2=self.l2_regul),
                    use_bias=False,
                ),
                BatchNormalization(),
                ReLU(),
            ],
        )
        self.sepconv8 = Sequential(
            [
                SeparableConv2D(
                    filters=self.filters,
                    depth_multiplier=1,
                    kernel_size=self.kernel_size,
                    padding=self.padding,
                    strides=self.strides,
                    dilation_rate=1,
                    depthwise_initializer=self.kernel_initializer,
                    kernel_regularizer=tf.keras.regularizers.l2(l2=self.l2_regul),
                    use_bias=False,
                ),
                BatchNormalization(),
                ReLU(),
            ],
        )

    def call(self, inputs, training=None) -> tf.Tensor:

        c3_output, c4_output, c5_output = inputs

        fmap3 = self.conv1(c3_output)
        fmap4 = self.upsample(self.conv2(c4_output))
        fmap5 = self.upsample4(self.conv3(c5_output))

        fmap1 = self.concat([fmap3, fmap4, fmap5])

        sep_fmap1 = self.sepconv1(fmap1)
        sep_fmap2 = self.sepconv2(fmap1)
        sep_fmap4 = self.sepconv4(fmap1)
        sep_fmap8 = self.sepconv8(fmap1)

        fmap2 = self.concat([sep_fmap1, sep_fmap2, sep_fmap4, sep_fmap8])

        return self.conv4(fmap2)

    def get_config(self) -> Dict[str, Any]:

        config = super().get_config()
        config.update(
            {
                "filters": self.filters,
                "kernel_size": self.kernel_size,
                "strides": self.strides,
                "padding": self.padding,
                "kernel_initializer": self.kernel_initializer,
                "l2_regul": self.l2_regul,
            },
        )
        return config
