import tensorflow as tf
from tensorflow.keras.layers import (
    Activation,
    AveragePooling2D,
    BatchNormalization,
    Concatenate,
    Conv2D,
    ReLU,
    UpSampling2D,
)


def conv_bn_relu(
    tensor: tf.Tensor,
    num_filters: int,
    kernel_size: int,
    padding: str = "same",
    strides: int = 1,
    dilation_rate: int = 1,
    w_init: str = "he_normal",
) -> tf.Tensor:

    fmap = Conv2D(
        filters=num_filters,
        kernel_size=kernel_size,
        padding=padding,
        strides=strides,
        dilation_rate=dilation_rate,
        kernel_regularizer=tf.keras.regularizers.l2(l2=1e-4),
        kernel_initializer=w_init,
        use_bias=False,
    )(tensor)

    # fmap = tfa.layers.GroupNormalization()(fmap)

    fmap = BatchNormalization()(fmap)

    return ReLU()(fmap)


class SharedDilatedConv(tf.keras.layers.Layer):
    def __init__(
        self,
        filters,
        kernel_size,
        strides,
        kernel_initializer,
        bias_initializer,
        use_bias,
        *args,
        **kwargs
    ):
        super(SharedDilatedConv, self).__init__()
        self.filters = filters
        self.kernel_size = kernel_size
        self.kernel_initializer = kernel_initializer
        self.bias_initializer = bias_initializer
        self.strides = strides
        self.use_bias = use_bias
        self.w = None
        self.b = None
        self.dilation_rates = [6, 12, 18]

        self.relu = tf.keras.layers.ReLU()
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.bn2 = tf.keras.layers.BatchNormalization()
        self.bn3 = tf.keras.layers.BatchNormalization()

    def build(self, input_shape):
        *_, n_channels = input_shape
        self.w = tf.Variable(
            initial_value=self.kernel_initializer(
                shape=(*self.kernel_size, n_channels, self.filters), dtype="float32"
            ),
            trainable=True,
        )
        if self.use_bias:
            self.b = tf.Variable(
                initial_value=self.bias_initializer(
                    shape=(self.filters,), dtype="float32"
                ),
                trainable=True,
            )

    def call(self, inputs, training=None):
        x1 = tf.nn.conv2d(
            inputs,
            self.w,
            padding="SAME",
            strides=self.strides,
            dilations=self.dilation_rates[0],
        )
        if self.use_bias:
            x1 = x1 + self.b
        x1 = self.relu(self.bn1(x1))

        x2 = tf.nn.conv2d(
            inputs,
            self.w,
            padding="SAME",
            strides=self.strides,
            dilations=self.dilation_rates[1],
        )
        if self.use_bias:
            x2 = x2 + self.b
        x2 = self.relu(self.bn2(x2))

        x3 = tf.nn.conv2d(
            inputs,
            self.w,
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
        kernel_initializer=tf.initializers.HeUniform(seed=42),
        bias_initializer=tf.initializers.Zeros(),
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

    return conv_bn_relu(tensor=fmap, num_filters=filters, kernel_size=1)


def decoder(fmap1, fmap2, filters):

    fmap1 = UpSampling2D(size=(4, 4), interpolation="bilinear")(fmap1)

    fmap2 = conv_bn_relu(tensor=fmap2, num_filters=filters, kernel_size=1)

    fmap = Concatenate(axis=-1)([fmap1, fmap2])

    return conv_bn_relu(tensor=fmap, num_filters=filters, kernel_size=3)


def get_ksac_model(n_classes: int, backbone: tf.keras.Model) -> tf.keras.Model:

    c2_output, _, c4_output, _ = backbone.outputs

    fm = ksac_module(c4_output, filters=256)

    fmap = decoder(fm, c2_output, filters=128)

    fmap = UpSampling2D(size=(4, 4), interpolation="bilinear")(fmap)

    fmap = Conv2D(
        filters=n_classes,
        kernel_size=(3, 3),
        strides=(1, 1),
        padding="same",
        kernel_initializer="he_uniform",
    )(fmap)

    out = Activation("softmax")(fmap)

    return tf.keras.Model(inputs=[backbone.inputs], outputs=[out])
