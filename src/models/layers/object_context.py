from typing import Any, Dict

import tensorflow as tf
from einops.layers.tensorflow import Rearrange
from tensorflow.keras.layers import (
    AveragePooling2D,
    BatchNormalization,
    Concatenate,
    Conv2D,
    Layer,
    ReLU,
    UpSampling2D,
)
from tensorflow.keras.models import Sequential


@tf.keras.utils.register_keras_serializable()
class SelfAttention2D(Layer):
    """
    Description of SelfAttention2D

    Attributes:
        softmax (type): `Softmax` function.
        theta (type): `[Conv2D-BatchNormalization-ReLU]x2` layer.
        phi (type): `[Conv2D-BatchNormalization-ReLU]x2` layer.
        gamma (type): `Conv2D` layer.
        rho (type): `Conv2D` layer.

    Inheritance:
        tf.keras.layers.Layer:
    """

    def __init__(self, filters: int, l2_regul: float = 1e-4, *args, **kwargs) -> None:
        """Initilization of the class.

        Args:
            filters (int, optional): Number of filters in each `Conv2D` layers.
            l2_regul (float, optional): Value of the constraint used for the
                $L_2$ regularization. Defaults to 1e-4.
        """
        super().__init__(*args, **kwargs)

        self.filters = filters
        self.l2_regul = l2_regul

        self.regul = tf.sqrt(2 / self.filters)
        self.softmax = tf.keras.layers.Activation("softmax")

        self.theta = Sequential(
            [
                Conv2D(
                    filters // 2,
                    kernel_size=(1, 1),
                    padding="same",
                    use_bias=False,
                    kernel_initializer="he_uniform",
                    kernel_regularizer=tf.keras.regularizers.l2(l2=l2_regul),
                ),
                BatchNormalization(),
                ReLU(),
                Conv2D(
                    filters // 2,
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

        self.phi = Sequential(
            [
                Conv2D(
                    filters // 2,
                    kernel_size=(1, 1),
                    padding="same",
                    use_bias=False,
                    kernel_initializer="he_uniform",
                    kernel_regularizer=tf.keras.regularizers.l2(l2=l2_regul),
                ),
                BatchNormalization(),
                ReLU(),
                Conv2D(
                    filters // 2,
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

        self.gamma = Conv2D(
            filters // 2,
            kernel_size=(1, 1),
            padding="same",
            use_bias=False,
            kernel_initializer="he_uniform",
            kernel_regularizer=tf.keras.regularizers.l2(l2=l2_regul),
        )

        self.rho = Conv2D(
            filters,
            kernel_size=(1, 1),
            padding="same",
            use_bias=False,
            kernel_initializer="he_uniform",
            kernel_regularizer=tf.keras.regularizers.l2(l2=l2_regul),
        )

    def call(self, inputs, training=None) -> tf.Tensor:

        out_phi = self.phi(inputs)
        out_theta = self.theta(inputs)
        out_gamma = self.gamma(inputs)

        out_product1 = self.regul * tf.matmul(out_theta, out_phi, transpose_b=True)

        w = self.softmax(out_product1)

        out_product2 = tf.matmul(w, out_gamma)

        return self.rho(out_product2)

    def get_config(self) -> Dict[str, Any]:

        config = super().get_config()
        config.update(
            {
                "filters": self.filters,
                "l2_regularization": self.l2_regul,
            },
        )
        return config


@tf.keras.utils.register_keras_serializable()
class ISA2D(Layer):
    """
    Description of ISA2D

    Inheritance:
        tf.keras.layers.Layer:
    """

    def __init__(self, P_h: int, P_w: int, *args, **kwargs) -> None:
        """Initilization of the class.

        Args:
            P_h (int): Number of partitions wanted for the height axis.
            P_w (int): Number of partitions wanted for the width axis.
        """
        super().__init__(*args, **kwargs)

        self.P_h = P_h
        self.P_w = P_w

    def build(self, input_shape) -> None:

        _, height, width, channels = input_shape

        self.attention1 = SelfAttention2D(channels)
        self.attention2 = SelfAttention2D(channels)

        Q_h, Q_w = height // self.P_h, width // self.P_w
        self.global_relation = Rearrange(
            "b (q_h p_h) (q_w p_w) c -> b p_h p_w q_h q_w c",
            q_h=Q_h,
            p_h=self.P_h,
            q_w=Q_w,
            p_w=self.P_w,
        )
        self.global_reshape = Rearrange("b p_h p_w q_h q_w c -> (b p_h p_w) q_h q_w c")

        self.local_relation = Rearrange(
            "(b p_h p_w) q_h q_w c -> b q_h q_w p_h p_w c",
            q_h=Q_h,
            p_h=self.P_h,
            q_w=Q_w,
            p_w=self.P_w,
        )
        self.local_reshape = Rearrange("b q_h q_w p_h p_w c -> (b q_h q_w) p_h p_w c")

        self.final_reshape = Rearrange(
            "(b q_h q_w) p_h p_w c ->  b (p_h q_h) (p_w q_w) c",
            q_h=Q_h,
            p_h=self.P_h,
            q_w=Q_w,
            p_w=self.P_w,
        )

    def call(self, inputs, training=None) -> tf.Tensor:

        # global relation
        fmap = self.global_relation(inputs)
        fmap = self.global_reshape(fmap)
        fmap = self.attention1(fmap)

        # local relation
        fmap = self.local_relation(fmap)
        fmap = self.local_reshape(fmap)
        fmap = self.attention2(fmap)

        return self.final_reshape(fmap)

    def get_config(self) -> Dict[str, Any]:

        config = super().get_config()
        config.update({"P_h": self.P_h, "P_w": self.P_w})
        return config


@tf.keras.utils.register_keras_serializable()
class BaseOC(Layer):
    """
    Description of Base_OC_Module

    Attributes:
        isa_block (type): `ISA2D` layer.
        concat (type): `Concatenate` layer.
        conv (type): `Conv2D-BatchNormalization-ReLU` layer.

    Inheritance:
        tf.keras.layers.Layer:

    """

    def __init__(self, filters: int, l2_regul: float = 1e-4, *args, **kwargs) -> None:
        """Initialization of the class.

        Args:
            filters (int, optional): Number of filters in each `Conv2D` layers.
            l2_regul (float, optional): Value of the constraint used for the
                $L_2$ regularization. Defaults to 1e-4.
        """
        super().__init__(*args, **kwargs)

        self.filters = filters
        self.l2_regul = l2_regul

        self.isa_block = ISA2D(8, 8)
        self.concat = Concatenate(axis=-1)
        self.conv = Sequential(
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

    def call(self, inputs, training=None) -> tf.Tensor:

        attention = self.isa_block(inputs)
        fmap = self.concat([attention, inputs])

        return self.conv(fmap)

    def get_config(self) -> Dict[str, Any]:

        config = super().get_config()
        config.update({"filters": self.filters, "l2_regularization": self.l2_regul})
        return config


@tf.keras.utils.register_keras_serializable()
class ASPP_OC(Layer):
    """
    Description of ASPP_OC

    Attributes:
        dilation_rate (type): Dilations rates used for the `Conv2D` layers.
        isa_block1 (type): `ISA2D` layer.
        isa_block2 (type): `ISA2D` layer.
        isa_block3 (type): `ISA2D` layer.
        isa_block4 (type): `ISA2D` layer.
        concat (type): `Concatenate` layer.
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
            filters (int): Number of filters in each `Conv2D` layers.
            l2_regul (float, optional): Value of the constraint used for the
                $L_2$ regularization. Defaults to 1e-4.
        """
        super().__init__(*args, **kwargs)

        self.filters = filters
        self.l2_regul = l2_regul
        self.dilation_rate = [6, 12, 18]

        self.isa_block1 = ISA2D(8, 8)
        self.isa_block2 = ISA2D(8, 8)
        self.isa_block3 = ISA2D(8, 8)
        self.isa_block4 = ISA2D(8, 8)

        self.concat = Concatenate(axis=-1)

        self.conv1 = Sequential(
            [
                Conv2D(
                    filters=filters,
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
                    filters=filters,
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
                    filters=filters,
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
                    filters=filters,
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
                    filters=filters,
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
                    filters=filters,
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
        fmap1 = self.isa_block1(fmap1)

        fmap2 = self.conv2(inputs)
        fmap2 = self.isa_block2(fmap2)

        fmap3 = self.conv3(inputs)
        fmap3 = self.isa_block3(fmap3)

        fmap4 = self.conv4(inputs)
        fmap4 = self.isa_block4(fmap4)

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
