from typing import Any, Dict, List

import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.keras.layers import Concatenate, Conv2D, ReLU, UpSampling2D
from tensorflow.keras.models import Sequential


# @tf.keras.utils.register_keras_serializable()
class FeaturePyramidNetwork(tf.keras.layers.Layer):
    """
    Description of FeaturePyramidNetwork

    The Feature Pyramid Networks head.

    Architecture:
        ![screen](./images/fpn_head.svg)


    Attributes:
        conv1 (type): `Conv2D` layer.
        conv2 (type): `Conv2D` layer.
        conv3 (type): `Conv2D` layer.
        conv4 (type): `Conv2D` layer.
        upsample (type): `Upsampling2D` layer.

    Inheritance:
        tf.keras.layers.Layer:

    Returns:
        List[tf.Tensor]: A list of feature maps, of dimensions $[(OS4, 256), (OS8, 256), (OS16, 256), (OS32, 256)]$.
    """

    def __init__(
        self,
        filters: int = 256,
        kernel_size: int = 1,
        strides: int = 1,
        padding: str = "same",
        kernel_initializer: str = "he_uniform",
        l2_regul: float = 1e-4,
        *args,
        **kwargs,
    ) -> None:
        """Initialization of the class.

        Args:
            filters (int, optional): Number of filters in each `Conv2D` layers. Defaults to 256.
            kernel_size (tuple, optional): Size of the convolution kernels in each `Conv2D` layers.
                Defaults to 1.
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
        self.conv1 = Conv2D(
            filters=self.filters,
            kernel_size=self.kernel_size,
            strides=self.strides,
            padding=self.padding,
            kernel_initializer=self.kernel_initializer,
            kernel_regularizer=tf.keras.regularizers.l2(l2=self.l2_regul),
        )
        self.conv2 = Conv2D(
            filters=self.filters,
            kernel_size=self.kernel_size,
            strides=self.strides,
            padding=self.padding,
            kernel_initializer=self.kernel_initializer,
            kernel_regularizer=tf.keras.regularizers.l2(l2=self.l2_regul),
        )
        self.conv3 = Conv2D(
            filters=self.filters,
            kernel_size=self.kernel_size,
            strides=self.strides,
            padding=self.padding,
            kernel_initializer=self.kernel_initializer,
            kernel_regularizer=tf.keras.regularizers.l2(l2=self.l2_regul),
        )
        self.conv4 = Conv2D(
            filters=self.filters,
            kernel_size=self.kernel_size,
            strides=self.strides,
            padding=self.padding,
            kernel_initializer=self.kernel_initializer,
            kernel_regularizer=tf.keras.regularizers.l2(l2=self.l2_regul),
        )
        self.upsample = UpSampling2D(size=(2, 2), interpolation="bilinear")

    def call(self, inputs, training=None) -> List[tf.Tensor]:

        c2_output, c3_output, c4_output, c5_output = inputs
        p2_output = self.conv1(c2_output)
        p3_output = self.conv2(c3_output)
        p4_output = self.conv3(c4_output)
        p5_output = self.conv4(c5_output)

        p4_output = p4_output + self.upsample(p5_output)

        p3_output = p3_output + self.upsample(p4_output)

        p2_output = p2_output + self.upsample(p3_output)

        return [p2_output, p3_output, p4_output, p5_output]

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

    @classmethod
    def from_config(cls, config):
        return cls(**config)


# @tf.keras.utils.register_keras_serializable()
class SemanticHeadFPN(tf.keras.layers.Layer):
    """
    Description of SemanticHeadFPN

    The segmentation head added to the FPN.

    Architecture:
        ![screenshot](./images/fpn_segmentation_head.svg)


    Attributes:
        conv1 (type): `Conv2D-GroupNormalization-ReLU` layer.
        conv2 (type): `Conv2D-GroupNormalization-ReLU` layer.
        conv3 (type): `Conv2D-GroupNormalization-ReLU` layer.
        conv4 (type): `Conv2D-GroupNormalization-ReLU` layer.
        conv5 (type): `Conv2D-GroupNormalization-ReLU` layer.
        conv6 (type): `Conv2D-GroupNormalization-ReLU` layer.
        conv7 (type): `Conv2D-GroupNormalization-ReLU` layer.
        upsample (type): `Upsampling2D` layer.
        concat (type): `Concatenate` layer.

    Inheritance:
        tf.keras.layers.Layer

    Returns:
        tf.Tensor: An output feature map of size $OS4$, with 512 filters.
    """

    def __init__(
        self,
        filters: int = 128,
        kernel_size: int = 1,
        strides: int = 1,
        padding: str = "same",
        kernel_initializer: str = "he_uniform",
        l2_regul: float = 1e-4,
        *args,
        **kwargs,
    ) -> None:
        """Initialization of the class.

        Args:
            filters (int, optional): Number of filters in each `Conv2D` layers. Defaults to 128.
            kernel_size (tuple, optional): Size of the convolution kernels in each `Conv2D` layers.
                Defaults to 1.
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
            name="seg1",
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
            name="seg2",
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
            name="seg3",
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
            name="seg4",
        )
        self.conv5 = Sequential(
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
            name="seg5",
        )
        self.conv6 = Sequential(
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
            name="seg6",
        )
        self.conv7 = Sequential(
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
            name="seg7",
        )
        self.upsample = UpSampling2D(size=(2, 2), interpolation="bilinear")
        self.concat = Concatenate(axis=-1)

    def call(self, inputs, training=None) -> tf.Tensor:

        p2_output, p3_output, p4_output, p5_output = inputs

        p5_output = self.upsample(self.conv1(p5_output))
        p5_output = self.upsample(self.conv2(p5_output))
        fmap5 = self.upsample(self.conv3(p5_output))

        p4_output = self.upsample(self.conv4(p4_output))
        fmap4 = self.upsample(self.conv5(p4_output))

        fmap3 = self.upsample(self.conv6(p3_output))

        fmap2 = self.conv7(p2_output)

        return self.concat([fmap5, fmap4, fmap3, fmap2])

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

    @classmethod
    def from_config(cls, config):
        return cls(**config)
