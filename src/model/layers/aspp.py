from typing import Any, Dict

import tensorflow as tf
from tensorflow.keras.layers import (
    AveragePooling2D,
    BatchNormalization,
    Concatenate,
    Conv2D,
    ReLU,
    UpSampling2D,
)
from tensorflow.keras.models import Sequential


class ASPP(tf.keras.layers.Layer):
    """
    Description of ASPP.

    Attributes:
        conv1 (type): `Conv2D-BatchNormalization-ReLU` layer.
        conv2 (type): `Conv2D-BatchNormalization-ReLU` layer.
        conv3 (type): `Conv2D-BatchNormalization-ReLU` layer.
        conv4 (type): `Conv2D-BatchNormalization-ReLU` layer.
        conv5 (type): `Conv2D-BatchNormalization-ReLU` layer.
        conv6 (type): `Conv2D-BatchNormalization-ReLU` layer.

    Inheritance:
        tf.keras.layers.Layer:
    """

    def __init__(
        self,
        filters: int,
        l2_regul: float = 1e-4,
        *args,
        **kwargs,
    ) -> None:
        """Initialization of the class.

        Args:
            filters (int, optional): Number of filters in each `Conv2D` layers.
            l2_regul (float, optional): Value of the constraint used for the
                $L_2$ regularization. Defaults to 1e-4.
        """
        super().__init__(*args, **kwargs)

        self.filters = filters
        self.l2_regul = l2_regul
        self.dilation_rate = [6, 12, 18]

        self.concat = Concatenate(axis=-1)

        self.conv1 = Sequential(
            [
                Conv2D(
                    filters,
                    kernel_size=(1, 1),
                    padding="same",
                    use_bias=False,
                    kernel_initializer="he_uniform",
                    kernel_regularizer=tf.keras.regularizers.l2(l2=l2_regul),
                ),
                BatchNormalization(),
                ReLU(),
            ],
        )
        self.conv2 = Sequential(
            [
                Conv2D(
                    filters,
                    kernel_size=(3, 3),
                    padding="same",
                    use_bias=False,
                    kernel_initializer="he_uniform",
                    dilation_rate=self.dilation_rate[0],
                    kernel_regularizer=tf.keras.regularizers.l2(l2=l2_regul),
                ),
                BatchNormalization(),
                ReLU(),
            ],
        )
        self.conv3 = Sequential(
            [
                Conv2D(
                    filters,
                    kernel_size=(3, 3),
                    padding="same",
                    use_bias=False,
                    kernel_initializer="he_uniform",
                    dilation_rate=self.dilation_rate[1],
                    kernel_regularizer=tf.keras.regularizers.l2(l2=l2_regul),
                ),
                BatchNormalization(),
                ReLU(),
            ],
        )
        self.conv4 = Sequential(
            [
                Conv2D(
                    filters,
                    kernel_size=(3, 3),
                    padding="same",
                    use_bias=False,
                    kernel_initializer="he_uniform",
                    dilation_rate=self.dilation_rate[2],
                    kernel_regularizer=tf.keras.regularizers.l2(l2=l2_regul),
                ),
                BatchNormalization(),
                ReLU(),
            ],
        )

        self.conv5 = Sequential(
            [
                Conv2D(
                    filters,
                    kernel_size=(1, 1),
                    padding="same",
                    use_bias=False,
                    kernel_initializer="he_uniform",
                    dilation_rate=1,
                    kernel_regularizer=tf.keras.regularizers.l2(l2=l2_regul),
                ),
                BatchNormalization(),
                ReLU(),
            ],
        )

        self.conv6 = Sequential(
            [
                Conv2D(
                    filters,
                    kernel_size=(1, 1),
                    padding="same",
                    use_bias=False,
                    kernel_initializer="he_uniform",
                    dilation_rate=1,
                    kernel_regularizer=tf.keras.regularizers.l2(l2=l2_regul),
                ),
                BatchNormalization(),
                ReLU(),
            ],
        )

    def build(self, input_shape) -> None:

        _, height, width, *_ = input_shape
        self.pooling = AveragePooling2D(pool_size=(height, width))
        self.upsample = UpSampling2D(size=(height, width), interpolation="bilinear")

    def call(self, inputs, training=None) -> tf.Tensor:

        fmap1 = self.conv1(inputs)

        fmap2 = self.conv2(inputs)

        fmap3 = self.conv3(inputs)

        fmap4 = self.conv4(inputs)

        fmap_pool = self.pooling(inputs)
        fmap_pool = self.conv5(fmap_pool)
        fmap_pool = self.upsample(fmap_pool)

        fmap = self.concat([fmap_pool, fmap1, fmap2, fmap3, fmap4])

        return self.conv6(fmap)

    def get_config(self) -> Dict[str, Any]:

        config = super().get_config()
        config.update(
            {
                "filters": self.filters,
                "l2_regularization": self.l2_regul,
                "dilation_rate": self.dilation_rate,
            },
        )
        return config
