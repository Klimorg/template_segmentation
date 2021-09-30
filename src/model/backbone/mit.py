from typing import Any, Dict, List

import numpy as np
import tensorflow as tf
from loguru import logger
from tensorflow.keras.layers import (
    BatchNormalization,
    Conv2D,
    Dense,
    DepthwiseConv2D,
    Dropout,
    Input,
    LayerNormalization,
    MaxPool2D,
    ReLU,
    Reshape,
)
from tensorflow.keras.models import Model, Sequential


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
                shape, 0, 1, dtype=inputs.dtype
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


class OverlapPatchEmbed(tf.keras.layers.Layer):
    def __init__(
        self,
        patch_size: int = 7,
        strides: int = 4,
        emb_dim: int = 768,
        l2_regul: float = 1e-4,
        *args,
        **kwargs,
    ) -> None:

        super().__init__(*args, **kwargs)

        self.patch_size = patch_size
        self.strides = strides
        self.emb_dim = emb_dim
        self.l2_regul = l2_regul

        self.norm = LayerNormalization()

    def build(self, input_shape) -> None:

        _, height, width, channels = input_shape

        self.H = height // self.patch_size
        self.W = width // self.patch_size

        self.proj = Conv2D(
            self.emb_dim,
            kernel_size=self.patch_size,
            strides=self.strides,
            padding="same",
            use_bias=False,
            kernel_initializer="he_uniform",
            kernel_regularizer=tf.keras.regularizers.l2(l2=self.l2_regul),
        )

        self.reshape = Reshape(target_shape=(self.H * self.W, -1))

    def call(self, inputs, training=None) -> tf.Tensor:

        fmap = self.proj(inputs)
        fmap = self.reshape(fmap)
        return self.norm(fmap)

    def get_config(self) -> Dict[str, Any]:

        config = super().get_config()
        config.update(
            {
                "img_size": self.img_size,
                "patch_size": self.patch_size,
                "strides": self.strides,
                "emb_dim": self.emb_dim,
                "l2_regul": self.l2_regul,
            }
        )
        return config


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

        _, units, _ = input_shape

        height = int(tf.sqrt(float(units)))
        width = int(tf.sqrt(float(units)))

        self.square_reshape = Reshape(target_shape=(height, width, -1))
        self.wide_reshape = Reshape(target_shape=(units, -1))

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
        fmap = self.fc1(inputs)

        fmap = self.square_reshape(fmap)
        fmap = self.depth_conv(fmap)
        fmap = self.wide_reshape(fmap)

        fmap = self.gelu(fmap)
        return self.fc2(fmap)

    def get_config(self) -> Dict[str, Any]:

        config = super().get_config()
        config.update(
            {
                "fc1_units": self.fc1_units,
                "fc2_units": self.fc2_units,
                "l2_regularization": self.l2_regul,
            }
        )
        return config


class Attention(tf.keras.layers.Layer):
    def __init__(
        self,
        units: int,
        num_heads: int = 8,
        attn_drop: float = 0,
        proj_drop: float = 0,
        downsample_ratio: int = 1,
        l2_regul: float = 1e-4,
        *args,
        **kwargs,
    ) -> None:

        super().__init__(*args, **kwargs)
        assert (
            units % num_heads == 0
        ), f"dim {units} should be divided by num_heads {num_heads}."

        self.units = units
        self.num_heads = num_heads
        self.head_dims = units // num_heads
        self.l2_regul = l2_regul
        self.attn_drop = attn_drop
        self.proj_drop = proj_drop
        self.downsample_ratio = downsample_ratio

        self.scale = 1 / tf.sqrt(self.head_dims)

        self.softmax = tf.keras.activations.softmax

    def build(self, input_shape) -> None:

        _, units, _ = input_shape

        height = int(tf.sqrt(float(units)))
        width = int(tf.sqrt(float(units)))

        self.square_reshape = Reshape(target_shape=(height, width, -1))
        self.wide_reshape = Reshape(target_shape=(units, -1))

        self.query = Dense(
            units=self.units,
            use_bias=False,
            kernel_initializer="he_uniform",
            kernel_regularizer=tf.keras.regularizers.l2(l2=self.l2_regul),
        )

        self.key_value = Dense(
            units=self.units * 2,
            use_bias=False,
            kernel_initializer="he_uniform",
            kernel_regularizer=tf.keras.regularizers.l2(l2=self.l2_regul),
        )

        self.proj = Dense(
            units=self.units,
            kernel_initializer="he_uniform",
            kernel_regularizer=tf.keras.regularizers.l2(l2=self.l2_regul),
        )

        self.att_drop = Dropout(rate=self.attn_drop)
        self.pro_drop = Dropout(rate=self.proj_drop)

        if self.downsample_ratio > 1:
            self.downsample = Conv2D(
                self.units,
                kernel_size=self.downsample_ratio,
                strides=self.downsample_ratio,
                padding="same",
                use_bias=False,
                kernel_initializer="he_uniform",
                kernel_regularizer=tf.keras.regularizers.l2(l2=self.l2_regul),
            )
            self.norm = LayerNormalization()

    def call(self, inputs, training=None) -> tf.Tensor:
        fmap = self.fc1(inputs)

        fmap = self.square_reshape(fmap)
        fmap = self.depth_conv(fmap)
        fmap = self.wide_reshape(fmap)

        fmap = self.gelu(fmap)
        return self.fc2(fmap)

    def get_config(self) -> Dict[str, Any]:

        config = super().get_config()
        config.update(
            {
                "fc1_units": self.fc1_units,
                "fc2_units": self.fc2_units,
                "l2_regularization": self.l2_regul,
            }
        )
        return config


if __name__ == "__main__":

    fmap = np.random.rand(1, 224, 224, 3)

    out = OverlapPatchEmbed()(fmap)
    out = Mlp(512, 512)(out)
    print(f"{out.shape.as_list()}")
