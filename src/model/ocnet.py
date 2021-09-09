import tensorflow as tf
from tensorflow.keras.layers import (
    Activation,
    AveragePooling2D,
    BatchNormalization,
    Concatenate,
    Conv2D,
    Permute,
    ReLU,
    UpSampling2D,
)
from tensorflow.keras.models import Sequential

from src.model.layers.common_layers import conv_bn_relu


class SelfAttention2D(tf.keras.layers.Layer):
    """
    Description of SelfAttention2D

    Attributes:
        filters (type):
        regul (type):
        filters (type):
        softmax (type):
        theta (type):
        phi (type):
        gamma (type):
        rho (type):

    Inheritance:
        tf.keras.layers.Layer:

    Args:
        filters (undefined):
        **kwargs (undefined):

    """

    def __init__(self, filters: int, l2_regul: float = 1e-4, *args, **kwargs):
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
            ]
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
            ]
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

    def call(self, inputs, training=None):

        out_phi = self.phi(inputs)
        out_theta = self.theta(inputs)
        out_gamma = self.gamma(inputs)

        out_product1 = self.regul * tf.matmul(out_theta, out_phi, transpose_b=True)

        w = self.softmax(out_product1)

        out_product2 = tf.matmul(w, out_gamma)

        return self.rho(out_product2)

    def get_config(self):

        config = super().get_config()
        config.update(
            {
                "filters": self.filters,
                "l2_regularization": self.l2_regul,
            }
        )
        return config


class ISA2D(tf.keras.layers.Layer):
    """
    Description of ISA2D

    Attributes:
        P_h (type):
        P_w (type):

    Inheritance:
        tf.keras.layers.Layer:

    Args:
        P_h (undefined):
        P_w (undefined):
        **kwargs (undefined):

    """

    def __init__(self, P_h, P_w, **kwargs):
        super().__init__(**kwargs)

        self.P_h = P_h
        self.P_w = P_w

    def build(self, input_shape):

        self.attention1 = SelfAttention2D(input_shape[-1])
        self.attention2 = SelfAttention2D(input_shape[-1])

    def call(self, inputs, training=None):

        _, H, W, C = tf.keras.backend.int_shape(inputs)
        Q_h, Q_w = H // self.P_h, W // self.P_w

        # global relation
        fmap = tf.reshape(inputs, [-1, Q_h, self.P_h, Q_w, self.P_w, C])
        fmap = Permute((4, 1, 2, 3, 5))(fmap)
        fmap = tf.reshape(fmap, [-1, Q_h, Q_w, C])
        fmap = self.attention1(fmap)

        # local relation
        fmap = tf.reshape(fmap, [-1, self.P_h, self.P_w, Q_h, Q_w, C])
        fmap = Permute((3, 4, 1, 2, 5))(fmap)
        fmap = tf.reshape(fmap, [-1, self.P_h, self.P_w, C])
        fmap = self.attention2(fmap)

        # reshape
        fmap = tf.reshape(fmap, [-1, Q_h, Q_w, self.P_h, self.P_w, C])
        fmap = Permute((3, 1, 2, 4, 5))(fmap)

        return tf.reshape(fmap, [-1, H, W, C])

    def get_config(self):

        config = super().get_config()
        config.update({"P_h": self.P_h, "P_w": self.P_w})
        return config


class Base_OC_Module(tf.keras.layers.Layer):
    """
    Description of Base_OC_Module

    Attributes:
        filters (type):
        isa_block (type):
        concat (type):
        conv_bn_relu (type):

    Inheritance:
        tf.keras.layers.Layer:

    Args:
        filters (undefined):
        **kwargs (undefined):

    """

    def __init__(self, filters: int, l2_regul: float = 1e-4, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.filters = filters
        self.l2_regul = l2_regul

        self.isa_block = ISA2D(8, 8)
        self.concat = Concatenate(axis=-1)
        self.conv_bn_relu = Sequential(
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
            ]
        )

    def call(self, inputs, training=None):

        attention = self.isa_block(inputs)
        fmap = self.concat([attention, inputs])

        return self.conv_bn_relu(fmap)

    def get_config(self):

        config = super().get_config()
        config.update({"filters": self.filters, "l2_regularization": self.l2_regul})
        return config


class ASPP_OC(tf.keras.layers.Layer):
    """
    Description of ASPP_OC

    Attributes:
        filters (type):
        l2_regul (type):
        dilation_rate (type):
        isa_block1 (type):
        isa_block2 (type):
        isa_block3 (type):
        isa_block4 (type):
        concat (type):
        conv1 (type):
        conv2 (type):
        conv3 (type):
        conv4 (type):
        conv5 (type):
        conv6 (type):

    Inheritance:
        tf.keras.layers.Layer:

    Args:
        filters (int):
        l2_regul (float=1e-4):
        *args (undefined):
        **kwargs (undefined):

    """

    def __init__(
        self,
        filters: int,
        l2_regul: float = 1e-4,
        *args,
        **kwargs,
    ):
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
                    filters,
                    kernel_size=(1, 1),
                    padding="same",
                    use_bias=False,
                    kernel_initializer="he_uniform",
                    kernel_regularizer=tf.keras.regularizers.l2(l2=l2_regul),
                ),
                BatchNormalization(),
                ReLU(),
            ]
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
            ]
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
            ]
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
            ]
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
            ]
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
            ]
        )

    def build(self, input_shape):
        _, height, width, *_ = input_shape
        self.pooling = AveragePooling2D(pool_size=(height, width))
        self.upsample = UpSampling2D(size=(height, width), interpolation="bilinear")

    def call(self, inputs, training=None):

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

    def get_config(self):

        config = super().get_config()
        config.update(
            {
                "filters": self.filters,
                "l2_regularization": self.l2_regul,
                "dilation_rate": self.dilation_rate,
            }
        )
        return config


def get_segmentation_module(
    n_classes: int,
    backbone: tf.keras.Model,
    architecture: str,
    filters: int,
    name: str,
) -> tf.keras.Model:
    """Instantiate the segmentation head module for the segmentation task.

    Args:
        n_classes (int): Number of classes in the segmentation task.
        backbone (tf.keras.Model): CNN used as backbone/feature extractor.
        name (str): Name of the segmentation head module.

    Returns:
        A semantic segmentation model.
    """

    fmap = backbone.outputs[1]

    if architecture == "base_ocnet":
        fmap = conv_bn_relu(fmap, filters=1024, kernel_size=3, name="pre_OCP_conv")
        fmap = Base_OC_Module(filters=filters)(fmap)
    elif architecture == "aspp_ocnet":
        fmap = ASPP_OC(filters=filters)(fmap)

    fmap = Conv2D(
        filters=n_classes,
        kernel_size=(3, 3),
        padding="same",
        kernel_regularizer=tf.keras.regularizers.l2(l2=1e-4),
        kernel_initializer="he_uniform",
        use_bias=False,
    )(fmap)

    fmap = Activation("softmax")(fmap)

    out = UpSampling2D(size=8, interpolation="bilinear")(fmap)

    return tf.keras.Model(inputs=[backbone.inputs], outputs=[out], name=name)
