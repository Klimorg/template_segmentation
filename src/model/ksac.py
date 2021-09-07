import tensorflow as tf
from tensorflow.keras.layers import (
    Activation,
    AveragePooling2D,
    BatchNormalization,
    Concatenate,
    Conv2D,
    UpSampling2D,
)

from src.model.layers.common_layers import conv_bn_relu


class SharedDilatedConv(tf.keras.layers.Layer):
    """
    Description of SharedDilatedConv

    Attributes:
        filters (type):
        kernel_size (type):
        kernel_initializer (type):
        bias_initializer (type):
        strides (type):
        use_bias (type):
        w (type):
        b (type):
        dilation_rates (type):
        relu (type):

    Inheritance:
        tf.keras.layers.Layer:

    Args:
        filters (undefined):
        kernel_size (undefined):
        strides (undefined):
        kernel_initializer (undefined):
        use_bias (undefined):
        bias_initializer=None (undefined):
        *args (undefined):
        **kwargs (undefined):

    """

    def __init__(
        self,
        filters,
        kernel_size,
        strides,
        kernel_initializer,
        use_bias,
        bias_initializer=None,
        *args,
        **kwargs,
    ):
        """[summary]

        Args:
            filters ([type]): [description]
            kernel_size ([type]): [description]
            strides ([type]): [description]
            kernel_initializer ([type]): [description]
            use_bias ([type]): [description]
            bias_initializer ([type], optional): [description]. Defaults to None.
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

        self.relu = tf.keras.layers.ReLU()

    def build(self, input_shape):

        *_, n_channels = input_shape
        self.kernel = self.add_weight(
            name="kernel",
            shape=(*self.kernel_size, n_channels, self.filters),
            initializer=self.kernel_initializer,
            trainable=True,
            dtype="float32",
        )
        if self.use_bias:
            self.b = self.add_weight(
                name="bias",
                shape=(self.filters,),
                initializer=self.bias_initializer,
                trainable=True,
                dtype="float32",
            )
        else:
            self.b = None

        self.bn1 = tf.keras.layers.BatchNormalization()
        self.bn2 = tf.keras.layers.BatchNormalization()
        self.bn3 = tf.keras.layers.BatchNormalization()

    def call(self, inputs, training=None):

        x1 = tf.nn.conv2d(
            inputs,
            self.kernel,
            padding="SAME",
            strides=self.strides,
            dilations=self.dilation_rates[0],
        )
        if self.use_bias:
            x1 = x1 + self.b
        x1 = self.relu(self.bn1(x1))

        x2 = tf.nn.conv2d(
            inputs,
            self.kernel,
            padding="SAME",
            strides=self.strides,
            dilations=self.dilation_rates[1],
        )
        if self.use_bias:
            x2 = x2 + self.b
        x2 = self.relu(self.bn2(x2))

        x3 = tf.nn.conv2d(
            inputs,
            self.kernel,
            padding="SAME",
            strides=self.strides,
            dilations=self.dilation_rates[2],
        )
        if self.use_bias:
            x3 = x3 + self.b
        x3 = self.relu(self.bn2(x3))

        return x1 + x2 + x3

    def get_config(self):
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


def ksac_module(fmap, filters):
    """[summary]

    Args:
        fmap ([type]): [description]
        filters ([type]): [description]

    Returns:
        [type]: [description]
    """

    height, width = fmap.shape.as_list()[1:3]

    fmap1 = Conv2D(
        filters=filters,
        kernel_size=(1, 1),
        strides=(1, 1),
        padding="same",
        kernel_initializer="he_uniform",
        use_bias=False,
    )(fmap)
    fmap1 = BatchNormalization()(fmap1)

    fmap2 = SharedDilatedConv(
        filters=filters,
        kernel_size=(3, 3),
        strides=(1, 1),
        kernel_initializer="he_uniform",
        use_bias=False,
    )(fmap)

    fmap3 = AveragePooling2D(pool_size=(height, width))(fmap)
    fmap3 = Conv2D(
        filters=filters,
        kernel_size=(1, 1),
        strides=(1, 1),
        padding="same",
        kernel_initializer="he_uniform",
        use_bias=False,
    )(fmap3)
    fmap3 = UpSampling2D(size=(height, width), interpolation="bilinear")(fmap3)

    fmap = Concatenate(axis=-1)([fmap1, fmap2, fmap3])

    return conv_bn_relu(tensor=fmap, filters=filters, kernel_size=1, name="ksac_module")


def decoder(fmap1, fmap2, filters):
    """[summary]

    Args:
        fmap1 ([type]): [description]
        fmap2 ([type]): [description]
        filters ([type]): [description]

    Returns:
        [type]: [description]
    """

    fmap1 = UpSampling2D(size=(4, 4), interpolation="bilinear")(fmap1)

    fmap2 = conv_bn_relu(tensor=fmap2, filters=filters, kernel_size=1, name="decoder1")

    fmap = Concatenate(axis=-1)([fmap1, fmap2])

    return conv_bn_relu(tensor=fmap, filters=filters, kernel_size=3, name="decoder2")


def get_segmentation_module(
    n_classes: int,
    backbone: tf.keras.Model,
    name: str,
    ksac_filters: int,
    decoder_filters: int,
) -> tf.keras.Model:
    """[summary]

    Args:
        n_classes (int): [description]
        backbone (tf.keras.Model): [description]
        name (str): [description]

    Returns:
        tf.keras.Model: [description]
    """

    c2_output, _, c4_output, _ = backbone.outputs

    fm = ksac_module(c4_output, filters=ksac_filters)

    fmap = decoder(fm, c2_output, filters=decoder_filters)

    fmap = UpSampling2D(size=(4, 4), interpolation="bilinear")(fmap)

    fmap = Conv2D(
        filters=n_classes,
        kernel_size=(3, 3),
        strides=(1, 1),
        padding="same",
        kernel_initializer="he_uniform",
    )(fmap)

    out = Activation("softmax")(fmap)

    return tf.keras.Model(inputs=[backbone.inputs], outputs=[out], name=name)
