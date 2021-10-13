from typing import Any, Dict, List

import numpy as np
import tensorflow as tf
from loguru import logger
from tensorflow.keras.layers import (
    BatchNormalization,
    Conv2D,
    Dense,
    DepthwiseConv2D,
    Input,
    LayerNormalization,
    MaxPool2D,
    ReLU,
)
from tensorflow.keras.models import Model, Sequential


class ConvTokenizer(tf.keras.layers.Layer):
    def __init__(
        self,
        filters: int = 64,
        l2_regul: float = 1e-4,
        *args,
        **kwargs,
    ) -> None:

        super().__init__(*args, **kwargs)

        self.filters = filters
        self.l2_regul = l2_regul

        self.block = Sequential(
            [
                Conv2D(
                    filters // 2,
                    kernel_size=3,
                    strides=2,
                    padding="same",
                    use_bias=False,
                    kernel_initializer="he_uniform",
                    kernel_regularizer=tf.keras.regularizers.l2(l2=self.l2_regul),
                ),
                BatchNormalization(),
                ReLU(),
                Conv2D(
                    filters // 2,
                    kernel_size=3,
                    strides=1,
                    padding="same",
                    use_bias=False,
                    kernel_initializer="he_uniform",
                    kernel_regularizer=tf.keras.regularizers.l2(l2=self.l2_regul),
                ),
                BatchNormalization(),
                ReLU(),
                Conv2D(
                    filters,
                    kernel_size=3,
                    strides=1,
                    padding="same",
                    use_bias=False,
                    kernel_initializer="he_uniform",
                    kernel_regularizer=tf.keras.regularizers.l2(l2=self.l2_regul),
                ),
                BatchNormalization(),
                ReLU(),
                MaxPool2D(pool_size=3, strides=2, padding="same"),
            ],
        )

    def call(self, inputs, training=None) -> tf.Tensor:
        return self.block(inputs)

    def get_config(self) -> Dict[str, Any]:

        config = super().get_config()
        config.update(
            {
                "filters": self.filters,
                "l2_regularization": self.l2_regul,
            },
        )
        return config


class ConvStage(tf.keras.layers.Layer):
    def __init__(
        self,
        num_blocks: int = 2,
        filters_in: int = 128,
        filters_out: int = 64,
        filters_downsample: int = 64,
        l2_regul: float = 1e-4,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)

        self.num_blocks = num_blocks
        self.filters_in = filters_in
        self.filters_out = filters_out
        self.filters_downsample = filters_downsample
        self.l2_regul = l2_regul

        self.conv_blocks = [
            Sequential(
                [
                    Conv2D(
                        self.filters_in,
                        kernel_size=1,
                        strides=1,
                        padding="same",
                        use_bias=False,
                        kernel_initializer="he_uniform",
                        kernel_regularizer=tf.keras.regularizers.l2(l2=self.l2_regul),
                    ),
                    BatchNormalization(),
                    ReLU(),
                    Conv2D(
                        self.filters_in,
                        kernel_size=3,
                        strides=1,
                        padding="same",
                        use_bias=False,
                        kernel_initializer="he_uniform",
                        kernel_regularizer=tf.keras.regularizers.l2(l2=self.l2_regul),
                    ),
                    BatchNormalization(),
                    ReLU(),
                    Conv2D(
                        self.filters_out,
                        kernel_size=1,
                        strides=1,
                        padding="same",
                        use_bias=False,
                        kernel_initializer="he_uniform",
                        kernel_regularizer=tf.keras.regularizers.l2(l2=self.l2_regul),
                    ),
                    BatchNormalization(),
                    ReLU(),
                ],
            )
            for _ in range(num_blocks)
        ]

        self.downsample = Conv2D(
            self.filters_downsample,
            kernel_size=3,
            strides=2,
            padding="same",
            use_bias=False,
            kernel_initializer="he_uniform",
            kernel_regularizer=tf.keras.regularizers.l2(l2=self.l2_regul),
        )

    def call(self, inputs, trainable=None) -> tf.Tensor:
        fmap = inputs
        for block in self.conv_blocks:
            fmap = fmap + block(fmap)

        return self.downsample(fmap)

    def get_config(self) -> Dict[str, Any]:

        config = super().get_config()
        config.update(
            {
                "num_blocks": self.num_blocks,
                "filters_in": self.filters_in,
                "filters_out": self.filters_out,
                "filters_downsample": self.filters_downsample,
                "l2_regularization": self.l2_regul,
            },
        )
        return config


class ConvDownsample(tf.keras.layers.Layer):
    def __init__(
        self,
        filters: int,
        l2_regul: float = 1e-4,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)

        self.filters = filters
        self.l2_regul = l2_regul

        self.downsample = Conv2D(
            filters,
            kernel_size=3,
            strides=2,
            padding="same",
            kernel_initializer="he_uniform",
            kernel_regularizer=tf.keras.regularizers.l2(l2=l2_regul),
        )

    def call(self, inputs) -> tf.Tensor:

        return self.downsample(inputs)

    def get_config(self) -> Dict[str, Any]:

        config = super().get_config()
        config.update(
            {
                "filters": self.filters,
                "l2_regularization": self.l2_regul,
            },
        )
        return config


# Referred from: github.com:rwightman/pytorch-image-models.
# https://keras.io/examples/vision/cct/#stochastic-depth-for-regularization
class StochasticDepth(tf.keras.layers.Layer):
    def __init__(
        self,
        drop_prop,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)

        self.drop_prob = drop_prop

    def call(self, inputs, training=None) -> tf.Tensor:
        if training:
            keep_prob = tf.cast(1 - self.drop_prob, dtype=inputs.dtype)
            shape = (tf.shape(inputs)[0],) + (1,) * (len(tf.shape(inputs)) - 1)
            random_tensor = keep_prob + tf.random.uniform(
                shape,
                0,
                1,
                dtype=inputs.dtype,
            )
            random_tensor = tf.floor(random_tensor)
            return (inputs / keep_prob) * random_tensor
        return inputs

    # def cast_inputs(self, inputs):
    #     # Casts to float16, the policy's lowest-precision dtype
    #     return self._mixed_precision_policy.cast_to_lowest(inputs)

    def get_config(self) -> Dict[str, Any]:

        config = super().get_config()
        config.update({"drop_prob": self.drop_prob})
        return config


class Identity(tf.keras.layers.Layer):
    def __init__(self) -> None:
        super().__init__(name="IdentityTF")

    def call(self, inputs) -> tf.Tensor:
        return inputs


class Mlp(tf.keras.layers.Layer):
    def __init__(
        self,
        fc1_units: int,
        fc2_units: int,
        l2_regul: float = 1e-4,
        *args,
        **kwargs,
    ) -> None:

        super().__init__(*args, **kwargs)

        self.fc1_units = fc1_units
        self.fc2_units = fc2_units
        self.l2_regul = l2_regul

        self.gelu = tf.keras.activations.gelu

    def build(self, input_shape) -> None:

        self.fc1 = Dense(
            units=self.fc1_units,
            kernel_initializer="he_uniform",
            kernel_regularizer=tf.keras.regularizers.l2(l2=self.l2_regul),
        )

        self.fc2 = Dense(
            units=self.fc2_units,
            kernel_initializer="he_uniform",
            kernel_regularizer=tf.keras.regularizers.l2(l2=self.l2_regul),
        )

    def call(self, inputs, training=None) -> tf.Tensor:
        fmap = self.fc1(inputs)
        fmap = self.gelu(fmap)
        return self.fc2(fmap)

    def get_config(self) -> Dict[str, Any]:

        config = super().get_config()
        config.update(
            {
                "fc1_units": self.fc1_units,
                "fc2_units": self.fc2_units,
                "l2_regularization": self.l2_regul,
            },
        )
        return config


class ConvMLPStage(tf.keras.layers.Layer):
    def __init__(
        self,
        expansion_units: int,
        units: int,
        stochastic_depth_rate: float = 0.1,
        l2_regul: float = 1e-4,
        *args,
        **kwargs,
    ) -> None:

        super().__init__(*args, **kwargs)
        self.expansion_units = expansion_units
        self.units = units
        self.stochastic_depth_rate = stochastic_depth_rate
        self.l2_regul = l2_regul

        self.norm1 = LayerNormalization()
        self.norm2 = LayerNormalization()
        self.connect_norm = LayerNormalization()

    def build(self, input_shape) -> None:

        self.channel_mlp1 = Mlp(fc1_units=self.expansion_units, fc2_units=self.units)
        self.channel_mlp2 = Mlp(fc1_units=self.expansion_units, fc2_units=self.units)
        self.stochastic_drop = (
            StochasticDepth(drop_prop=self.stochastic_depth_rate)
            if self.stochastic_depth_rate > 0
            else Identity()
        )

        self.depth_conv = DepthwiseConv2D(
            depth_multiplier=1,
            kernel_size=3,
            strides=1,
            padding="same",
            use_bias=False,
            kernel_initializer="he_uniform",
            kernel_regularizer=tf.keras.regularizers.l2(l2=self.l2_regul),
        )

    def call(self, inputs, training=None) -> tf.Tensor:

        fmap = inputs + self.stochastic_drop(self.channel_mlp1(self.norm1(inputs)))
        fmap = self.depth_conv(self.connect_norm(fmap))
        return fmap + self.stochastic_drop(self.channel_mlp2(self.norm2(inputs)))

    def get_config(self) -> Dict[str, Any]:

        config = super().get_config()
        config.update(
            {
                "expansion_units": self.expansion_units,
                "units": self.units,
                "stochastic_depth_rate": self.stochastic_depth_rate,
                "l2_regularization": self.l2_regul,
            },
        )
        return config


class BasicStage(tf.keras.layers.Layer):
    def __init__(
        self,
        num_blocks: int,
        units: int,
        mlp_ratio: int = 1,
        stochastic_depth_rate: float = 0.1,
        downsample: bool = True,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)

        self.num_blocks = num_blocks
        self.units = units
        self.mlp_ratio = mlp_ratio
        self.stochastic_depth_rate = stochastic_depth_rate
        self.downsample = downsample

    def build(self, input_shape) -> None:

        dpr = [
            rates
            for rates in np.linspace(0, self.stochastic_depth_rate, self.num_blocks)
        ]

        self.blocks = [
            ConvMLPStage(
                expansion_units=int(self.units * self.mlp_ratio),
                units=self.units,
                stochastic_depth_rate=dpr[idx],
            )
            for idx in range(self.num_blocks)
        ]

        self.downsample_mlp = (
            ConvDownsample(filters=int(self.units * 2))
            if self.downsample
            else Identity()
        )

    def call(self, inputs, trainable=None) -> tf.Tensor:

        for blk in self.blocks:
            inputs = blk(inputs)

        return self.downsample_mlp(inputs)

    def get_config(self) -> Dict[str, Any]:

        config = super().get_config()
        config.update(
            {
                "num_blocks": self.num_blocks,
                "units": self.units,
                "mlp_ratio": self.mlp_ratio,
                "stochastic_depth_rate": self.stochastic_depth_rate,
                "downsample": self.downsample,
            },
        )
        return config


def get_feature_extractor(
    img_shape: List[int],
    channels: int,
    n_conv_blocks: int,
    num_blocks: List[int],
    units: List[int],
    mlp_ratios: List[int],
):

    img_input = Input(img_shape)

    fmap = ConvTokenizer(filters=channels, name="tokenizer")(img_input)  # channels

    fmap = ConvStage(
        num_blocks=n_conv_blocks,
        filters_out=channels,
        filters_downsample=units[0],
        name="conv",
    )(fmap)
    fmap = BasicStage(
        num_blocks=num_blocks[0],
        units=units[1],
        mlp_ratio=mlp_ratios[0],
        downsample=True,
        name="mlp1",
    )(fmap)
    fmap = BasicStage(
        num_blocks=num_blocks[1],
        units=units[2],
        mlp_ratio=mlp_ratios[1],
        downsample=True,
        name="mlp2",
    )(fmap)
    fmap_out = BasicStage(
        num_blocks=num_blocks[2],
        units=units[3],
        mlp_ratio=mlp_ratios[2],
        downsample=False,
        name="mlp3",
    )(fmap)

    return Model(img_input, fmap_out)


def get_backbone(
    img_shape: List[int],
    channels: int,
    n_conv_blocks: int,
    num_blocks: List[int],
    units: List[int],
    mlp_ratios: List[int],
    backbone_name: str,
) -> tf.keras.Model:
    """Instantiate the model and use it as a backbone (feature extractor) for a semantic segmentation task.

    Args:
        img_shape (List[int]): [description]
        channels (int): [description]
        n_conv_blocks (int): [description]
        num_blocks (List[int]): [description]
        units (List[int]): [description]
        mlp_ratios (List[int]): [description]
        backbone_name (str): [description]

    Returns:
        A `tf.keras` model.
    """
    backbone = get_feature_extractor(
        img_shape=img_shape,
        channels=channels,
        n_conv_blocks=n_conv_blocks,
        num_blocks=num_blocks,
        units=units,
        mlp_ratios=mlp_ratios,
    )

    endpoint_layers = [
        "tokenizer",
        "conv",
        "mlp1",
        "mlp3",
    ]

    os4_output, os8_output, os16_output, os32_output = [
        backbone.get_layer(layer_name).output for layer_name in endpoint_layers
    ]

    height = img_shape[1]
    logger.info(f"os4_output OS : {int(height/os4_output.shape.as_list()[1])}")
    logger.info(f"os8_output OS : {int(height/os8_output.shape.as_list()[1])}")
    logger.info(f"os16_output OS : {int(height/os16_output.shape.as_list()[1])}")
    logger.info(f"os32_output OS : {int(height/os32_output.shape.as_list()[1])}")

    return Model(
        inputs=[backbone.input],
        outputs=[os4_output, os8_output, os16_output, os32_output],
        name=backbone_name,
    )


if __name__ == "__main__":

    # convmlp-xs
    channels = 64
    n_conv_blocks = 2
    num_blocks = [2, 2, 2]
    units = [128, 128, 256, 512]
    mlp_ratios = [2, 2, 2]

    mod = get_backbone(
        img_shape=(224, 224, 3),
        channels=channels,
        n_conv_blocks=n_conv_blocks,
        num_blocks=num_blocks,
        units=units,
        mlp_ratios=mlp_ratios,
        backbone_name="convmlp-xs",
    )
    mod.summary()
