from typing import Any, Dict, List

import numpy as np
import tensorflow as tf
from loguru import logger
from tensorflow.keras.activations import gelu
from tensorflow.keras.layers import (
    Activation,
    BatchNormalization,
    Conv2D,
    Dense,
    DepthwiseConv2D,
    Input,
    Layer,
    LayerNormalization,
    MaxPool2D,
    ReLU,
)
from tensorflow.keras.models import Model, Sequential


@tf.keras.utils.register_keras_serializable()
class ConvNextBlock(Layer):
    def __init__(
        self,
        filters: int,
        expansion_filters: int,
        l2_regul: float = 1e-4,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)

        self.filters = filters
        self.expansion_filters = expansion_filters
        self.l2_regul = l2_regul

        self.gelu = gelu

        self.layer_norm = LayerNormalization()

    def build(self, input_shape) -> None:
        batch_size, width, height, channels = input_shape

        self.dw_conv = DepthwiseConv2D(
            kernel_size=7,
            strides=1,
            padding="same",
            use_bias=False,
            depthwise_initializer="he_uniform",
            depthwise_regularizer=tf.keras.regularizers.l2(l2=self.l2_regul),
        )

        self.inverted_bottleneck = Conv2D(
            filters=self.expansion_filters,
            kernel_size=1,
            padding="same",
            use_bias=False,
            kernel_initializer="he_uniform",
            kernel_regularizer=tf.keras.regularizers.l2(l2=self.l2_regul),
        )

        self.conv = Conv2D(
            filters=self.filters,
            kernel_size=1,
            padding="same",
            use_bias=False,
            kernel_initializer="he_uniform",
            kernel_regularizer=tf.keras.regularizers.l2(l2=self.l2_regul),
        )

    def call(self, inputs, training=None) -> tf.Tensor:
        pass

    def get_config(self) -> Dict[str, Any]:
        config = super().get_config()
        config.update(
            {},
        )
        return config

    @classmethod
    def from_config(cls, config) -> None:
        return cls(**config)
