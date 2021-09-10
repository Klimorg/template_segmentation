import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.keras.layers import (
    Activation,
    AveragePooling2D,
    BatchNormalization,
    Concatenate,
    Conv2D,
    Permute,
    ReLU,
    SeparableConv2D,
    UpSampling2D,
)
from tensorflow.keras.models import Sequential


class ASPP(tf.keras.layers.Layer):
    """
    Description of ASPP

    Attributes:
        filters (type):
        l2_regul (type):
        dilation_rate (type):
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

        fmap2 = self.conv2(inputs)

        fmap3 = self.conv3(inputs)

        fmap4 = self.conv4(inputs)

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


class FeaturePyramidNetwork(tf.keras.layers.Layer):
    """
    Description of FeaturePyramidNetwork

    The Feature Pyramid Networks head.

    Architecture:
        ![screen](./images/fpn_head.svg)


    Attributes:
        filters (type):
        kernel_size (type):
        strides (type):
        padding (type):
        kernel_initializer (type):
        l2_regul (type):
        conv1 (type):
        conv2 (type):
        conv3 (type):
        conv4 (type):
        upsample (type):

    Inheritance:
        tf.keras.layers.Layer:

    Args:
       filters (int, optional): Number of filters in each `Conv2D` layers. Defaults to 256.
        kernel_size (tuple, optional): Size of the convolution kernels in each `Conv2D` layers.
            Defaults to (1, 1).
        strides (tuple, optional): Stride parameter in each `Conv2D` layers. Defaults to (1, 1).
        padding (str, optional): Paddinf parameter in each `Conv2D` layers. Defaults to "same".
        kernel_initializer (str, optional): Kernel initialization method used in each `Conv2D` layers.
            Defaults to "he_uniform".
        l2_regul (float, optional): Value of the constraint used for the
            $L_2$ regularization. Defaults to 1e-4.
        *args (undefined):
        **kwargs (undefined):

    Returns:
        A list of feature maps, of dimensions $[(OS4, 256), (OS8, 256), (OS16, 256), (OS32, 256)]$.
    """

    def __init__(
        self,
        filters: int = 256,
        kernel_size: int = 1,
        strides: int = 1,
        padding: str = "same",
        kernel_initializer: str = "he_uniform",
        l2_regul: float = 1e-4,
        backbone=None,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.kernel_initializer = kernel_initializer
        self.l2_regul = l2_regul
        self.backbone = backbone

    def build(self, input_shape):
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

    def call(self, inputs, training=None):

        c2_output, c3_output, c4_output, c5_output = inputs
        p2_output = self.conv1(c2_output)
        p3_output = self.conv2(c3_output)
        p4_output = self.conv3(c4_output)
        p5_output = self.conv4(c5_output)

        p4_output = p4_output + self.upsample(p5_output)

        p3_output = p3_output + self.upsample(p4_output)

        p2_output = p2_output + self.upsample(p3_output)

        return [p2_output, p3_output, p4_output, p5_output]

    def get_config(self):

        config = super().get_config()
        config.update(
            {
                "filters": self.filters,
                "kernel_size": self.kernel_size,
                "strides": self.strides,
                "padding": self.padding,
                "kernel_initializer": self.kernel_initializer,
                "l2_regul": self.l2_regul,
            }
        )
        return config


class SemanticHeadFPN(tf.keras.layers.Layer):
    """
    Description of SemanticHeadFPN

    The segmentation head added to the FPN.

    Architecture:
        ![screenshot](./images/fpn_segmentation_head.svg)


    Attributes:
        filters (type):
        kernel_size (type):
        strides (type):
        padding (type):
        kernel_initializer (type):
        l2_regul (type):
        conv1 (type):
        conv2 (type):
        conv3 (type):
        conv4 (type):
        conv5 (type):
        conv6 (type):
        conv7 (type):
        upsample (type):
        concat (type):

    Inheritance:
        tf.keras.layers.Layer:

    Args:
        filters (int, optional): [description]. Defaults to 128.
        kernel_size (int, optional): [description]. Defaults to 1.
        strides (int, optional): [description]. Defaults to 1.
        padding (str, optional): [description]. Defaults to "same".
        kernel_initializer (str, optional): [description]. Defaults to "he_uniform".
        l2_regul (float, optional): [description]. Defaults to 1e-4.
        *args (undefined):
        **kwargs (undefined):

    Inputs:
        p2_output (tf.Tensor): Feature map coming from the `p2_output` FPN head, output stride 4.
        p3_output (tf.Tensor): Feature map coming from the `p3_output` FPN head, output stride 8.
        p4_output (tf.Tensor): Feature map coming from the `p4_output` FPN head, output stride 16.
        p5_output (tf.Tensor): Feature map coming from the `p5_output` FPN head, output stride 32.

    Returns:
        An output feature map of size $OS4$, with 512 filters.

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
    ):
        super().__init__(*args, **kwargs)

        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.kernel_initializer = kernel_initializer
        self.l2_regul = l2_regul

    def build(self, input_shape):
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

    def call(self, inputs, training=None):

        p2_output, p3_output, p4_output, p5_output = inputs

        p5_output = self.upsample(self.conv1(p5_output))
        p5_output = self.upsample(self.conv2(p5_output))
        fmap5 = self.upsample(self.conv3(p5_output))

        p4_output = self.upsample(self.conv4(p4_output))
        fmap4 = self.upsample(self.conv5(p4_output))

        fmap3 = self.upsample(self.conv6(p3_output))

        fmap2 = self.conv7(p2_output)

        return self.concat([fmap5, fmap4, fmap3, fmap2])

    def get_config(self):

        config = super().get_config()
        config.update(
            {
                "filters": self.filters,
                "kernel_size": self.kernel_size,
                "strides": self.strides,
                "padding": self.padding,
                "kernel_initializer": self.kernel_initializer,
                "l2_regul": self.l2_regul,
            }
        )
        return config


class JointPyramidUpsampling(tf.keras.layers.Layer):
    """
    Description of JointPyramidUpsampling

    Joint Pyramid Upsampling module.

    Architecture:
        ![screenshot](./images/jpu_details.svg)


    Attributes:
        filters (type):
        kernel_size (type):
        strides (type):
        padding (type):
        kernel_initializer (type):
        l2_regul (type):
        conv1 (type):
        conv2 (type):
        conv3 (type):
        conv4 (type):
        upsample (type):
        concat (type):
        sepconv1 (type):
        sepconv2 (type):
        sepconv4 (type):
        sepconv8 (type):

    Inheritance:
        tf.keras.layers.Layer:

    Args:
        filters (int, optional): [description]. Defaults to 256.
        kernel_size (int, optional): [description]. Defaults to 1.
        strides (int, optional): [description]. Defaults to 1.
        padding (str, optional): [description]. Defaults to "same".
        kernel_initializer (str, optional): [description]. Defaults to "he_uniform".
        l2_regul (float, optional): [description]. Defaults to 1e-4.
        *args (undefined):
        **kwargs (undefined):

    Inputs:
        endpoints (List[tf.Tensor]): OS8, OS16, and OS32 endpoint feature maps of the backbone.

    Returns:
        Output feature map, $(H,W,C)$.
    """

    def __init__(
        self,
        filters: int = 256,
        kernel_size: int = 3,
        strides: int = 1,
        padding: str = "same",
        kernel_initializer: str = "he_uniform",
        l2_regul: float = 1e-4,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.kernel_initializer = kernel_initializer
        self.l2_regul = l2_regul

    def build(self, input_shape):
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
            ]
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
            ]
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
            ]
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
            ]
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
            ]
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
            ]
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
            ]
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
            ]
        )

    def call(self, inputs, training=None):

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

    def get_config(self):

        config = super().get_config()
        config.update(
            {
                "filters": self.filters,
                "kernel_size": self.kernel_size,
                "strides": self.strides,
                "padding": self.padding,
                "kernel_initializer": self.kernel_initializer,
                "l2_regul": self.l2_regul,
            }
        )
        return config
