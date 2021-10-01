from typing import Any, Dict, List

import numpy as np
import tensorflow as tf
from loguru import logger
from tensorflow.keras.layers import (
    Conv2D,
    Dense,
    DepthwiseConv2D,
    Dropout,
    Input,
    LayerNormalization,
    Permute,
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
        patch_size: int,
        strides: int,
        emb_dim: int,
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

        self.H = height // self.strides
        self.W = width // self.strides

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
        print(f"proj : {fmap.shape}")
        fmap = self.reshape(fmap)
        print(f"reshape : {fmap.shape}")
        return self.norm(fmap)

    def get_config(self) -> Dict[str, Any]:

        config = super().get_config()
        config.update(
            {
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
        fc_units: int,
        num_heads: int = 8,
        attn_drop_prob: float = 0,
        proj_drop_prob: float = 0,
        attn_reduction_ratio: int = 1,
        l2_regul: float = 1e-4,
        *args,
        **kwargs,
    ) -> None:

        super().__init__(*args, **kwargs)
        assert (
            fc_units % num_heads == 0
        ), f"dim {fc_units} should be divided by num_heads {num_heads}."

        self.fc_units = fc_units
        self.num_heads = num_heads
        self.attn_drop_prob = attn_drop_prob
        self.proj_drop_prob = proj_drop_prob
        self.attn_reduction_ratio = attn_reduction_ratio
        self.l2_regul = l2_regul

        self.head_dims = fc_units / num_heads
        self.scale = 1 / tf.sqrt(self.head_dims)

        self.softmax = tf.keras.activations.softmax

    def build(self, input_shape) -> None:

        _, units, _ = input_shape

        height = int(tf.sqrt(float(units)))
        width = int(tf.sqrt(float(units)))

        reduction_height = height // self.attn_reduction_ratio
        reduction_width = width // self.attn_reduction_ratio

        self.heads_reshape = Reshape(target_shape=(units, self.num_heads, -1))
        self.square_reshape = Reshape(target_shape=(height, width, -1))
        self.wide_reshape = Reshape(target_shape=(units, -1))
        self.wide_reduction_reshape = Reshape(
            target_shape=(reduction_height * reduction_width, -1)
        )
        self.kv_reshape = Reshape(
            target_shape=(-1, 2, self.num_heads, int(self.head_dims))
        )

        self.query = Dense(
            units=self.fc_units,
            use_bias=False,
            kernel_initializer="he_uniform",
            kernel_regularizer=tf.keras.regularizers.l2(l2=self.l2_regul),
        )

        self.key_value = Dense(
            units=self.fc_units * 2,
            use_bias=False,
            kernel_initializer="he_uniform",
            kernel_regularizer=tf.keras.regularizers.l2(l2=self.l2_regul),
        )

        self.proj = Dense(
            units=self.fc_units,
            kernel_initializer="he_uniform",
            kernel_regularizer=tf.keras.regularizers.l2(l2=self.l2_regul),
        )

        self.attn_drop = Dropout(rate=self.attn_drop_prob)
        self.proj_drop = Dropout(rate=self.proj_drop_prob)

        self.permute = Permute((2, 1, 3))

        if self.attn_reduction_ratio > 1:
            self.attn_conv = Conv2D(
                self.fc_units,
                kernel_size=self.attn_reduction_ratio,
                strides=self.attn_reduction_ratio,
                padding="same",
                use_bias=False,
                kernel_initializer="he_uniform",
                kernel_regularizer=tf.keras.regularizers.l2(l2=self.l2_regul),
            )
            self.norm = LayerNormalization()

    def call(self, inputs, training=None) -> tf.Tensor:
        queries = self.query(inputs)

        queries = self.heads_reshape(queries)
        queries = self.permute(queries)

        fmap = inputs
        if self.attn_reduction_ratio > 1:
            fmap = self.square_reshape(fmap)
            fmap = self.attn_conv(fmap)
            fmap = self.wide_reduction_reshape(fmap)
            fmap = self.norm(fmap)

        fmap = self.key_value(fmap)
        fmap = self.kv_reshape(fmap)
        fmap = tf.transpose(fmap, perm=[2, 0, 3, 1, 4])
        keys, values = tf.split(fmap, num_or_size_splits=2)
        keys = tf.squeeze(keys, axis=0)
        values = tf.squeeze(values, axis=0)

        attn = tf.matmul(queries, keys, transpose_b=True) * self.scale
        attn = self.softmax(attn)
        attn = self.attn_drop(attn)

        x = tf.matmul(attn, values)
        x = tf.transpose(x, perm=[0, 2, 1, 3])
        x = self.wide_reshape(x)
        x = self.proj(x)

        return self.proj_drop(x)

    def get_config(self) -> Dict[str, Any]:

        config = super().get_config()
        config.update(
            {
                "fc_units": self.fc_units,
                "num_heads": self.num_heads,
                "attn_drop_prob": self.attn_drop_prob,
                "proj_drop_prob": self.proj_drop_prob,
                "attn_reduction_ratio": self.attn_reduction_ratio,
                "l2_regul": self.l2_regul,
            }
        )
        return config


class FFNAttentionBlock(tf.keras.layers.Layer):
    def __init__(
        self,
        fc_units: int,
        num_heads: int = 8,
        mlp_ratio: int = 4,
        attn_drop_prob: float = 0,
        proj_drop_prob: float = 0,
        attn_reduction_ratio: int = 1,
        stochastic_depth_rate: float = 0.1,
        *args,
        **kwargs,
    ) -> None:

        super().__init__(*args, **kwargs)

        self.fc_units = fc_units
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.attn_drop_prob = attn_drop_prob
        self.proj_drop_prob = proj_drop_prob
        self.attn_reduction_ratio = attn_reduction_ratio
        self.stochastic_depth_rate = stochastic_depth_rate

    def build(self, input_shape) -> None:

        self.attn = Attention(
            fc_units=self.fc_units,
            num_heads=self.num_heads,
            attn_drop_prob=self.attn_drop_prob,
            proj_drop_prob=self.proj_drop_prob,
            attn_reduction_ratio=self.attn_reduction_ratio,
        )

        self.stochastic_drop = (
            StochasticDepth(drop_prop=self.stochastic_depth_rate)
            if self.stochastic_depth_rate > 0
            else Identity()
        )

        self.mlp = Mlp(
            fc1_units=self.fc_units * self.mlp_ratio,
            fc2_units=self.fc_units,
        )

        self.norm1 = LayerNormalization()
        self.norm2 = LayerNormalization()

    def call(self, inputs, training=None) -> tf.Tensor:
        print(f"Input : {inputs.shape}")

        fmap = self.stochastic_drop(self.attn(self.norm1(inputs)))
        print(f"fmap : {fmap.shape}")
        fmap = inputs + fmap
        fmap = fmap + self.stochastic_drop(self.mlp(self.norm2(fmap)))

        return fmap

    def get_config(self) -> Dict[str, Any]:

        config = super().get_config()
        config.update(
            {
                "fc_units": self.fc_units,
                "num_heads": self.num_heads,
                "mlp_ratio": self.mlp_ratio,
                "attn_drop_prob": self.attn_drop_prob,
                "proj_drop_prob": self.proj_drop_prob,
                "attn_reduction_ratio": self.attn_reduction_ratio,
                "stochastic_depth_rate": self.stochastic_depth_rate,
            }
        )
        return config


class StageBlock(tf.keras.layers.Layer):
    def __init__(
        self,
        fc_units: int,
        depth: int,
        num_heads: int = 8,
        mlp_ratio: int = 4,
        attn_drop_prob: float = 0,
        proj_drop_prob: float = 0,
        attn_reduction_ratio: int = 1,
        stochastic_depth_rate: float = 0.1,
        *args,
        **kwargs,
    ) -> None:

        super().__init__(*args, **kwargs)

        self.fc_units = fc_units
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.attn_drop_prob = attn_drop_prob
        self.proj_drop_prob = proj_drop_prob
        self.attn_reduction_ratio = attn_reduction_ratio
        self.stochastic_depth_rate = stochastic_depth_rate
        self.depth = depth

    def build(self, input_shape) -> None:

        self.blocks = [
            FFNAttentionBlock(
                fc_units=self.fc_units,
                num_heads=self.num_heads,
                mlp_ratio=self.mlp_ratio,
                attn_drop_prob=self.attn_drop_prob,
                proj_drop_prob=self.proj_drop_prob,
                attn_reduction_ratio=self.attn_reduction_ratio,
                stochastic_depth_rate=self.stochastic_depth_rate,
            )
            for _ in range(self.depth)
        ]

    def call(self, inputs, training=None) -> tf.Tensor:

        for blk in self.blocks:
            inputs = blk(inputs)

        return inputs

    def get_config(self) -> Dict[str, Any]:

        config = super().get_config()
        config.update(
            {
                "fc_units": self.fc_units,
                "depth": self.depth,
                "num_heads": self.num_heads,
                "mlp_ratio": self.mlp_ratio,
                "attn_drop_prob": self.attn_drop_prob,
                "proj_drop_prob": self.proj_drop_prob,
                "attn_reduction_ratio": self.attn_reduction_ratio,
                "stochastic_depth_rate": self.stochastic_depth_rate,
            }
        )
        return config


class SquareReshape(tf.keras.layers.Layer):
    def __init__(
        self,
        *args,
        **kwargs,
    ) -> None:

        super().__init__(*args, **kwargs)

    def build(self, input_shape) -> None:

        _, units, _ = input_shape

        height = int(tf.sqrt(float(units)))
        width = int(tf.sqrt(float(units)))

        self.square_reshape = Reshape(target_shape=(height, width, -1))

    def call(self, inputs, training=None) -> tf.Tensor:

        return self.square_reshape(inputs)

    def get_config(self) -> Dict[str, Any]:

        config = super().get_config()

        return config


def get_mix_vision_transformer(
    img_shape: List[int],
    patch_size: List[int],
    strides: List[int],
    emb_dims: List[int],
    num_heads: List[int],
    mlp_ratios: List[int],
    proj_drop_prob: float,
    attn_drop_prob: float,
    stochastic_depth_rate: float,
    attn_reduction_ratios: List[int],
    depths: List[int],
) -> tf.keras.Model:
    """Instantiate a MiT model.

    Returns:
        A `tf.keras` model.
    """

    dpr = [rates for rates in np.linspace(0, stochastic_depth_rate, np.sum(depths))]

    img_input = Input(img_shape)

    fmap = OverlapPatchEmbed(
        patch_size=patch_size[0], strides=strides[0], emb_dim=emb_dims[0]
    )(img_input)

    fmap = StageBlock(
        fc_units=emb_dims[0],
        depth=depths[0],
        num_heads=num_heads[0],
        mlp_ratio=mlp_ratios[0],
        attn_drop_prob=attn_drop_prob,
        proj_drop_prob=proj_drop_prob,
        attn_reduction_ratio=attn_reduction_ratios[0],
        stochastic_depth_rate=dpr[0],
        name="stage_1",
    )(fmap)
    fmap = LayerNormalization()(fmap)
    fmap = SquareReshape()(fmap)

    fmap = OverlapPatchEmbed(
        patch_size=patch_size[1], strides=strides[1], emb_dim=emb_dims[1]
    )(fmap)

    fmap = StageBlock(
        fc_units=emb_dims[1],
        depth=depths[1],
        num_heads=num_heads[1],
        mlp_ratio=mlp_ratios[1],
        attn_drop_prob=attn_drop_prob,
        proj_drop_prob=proj_drop_prob,
        attn_reduction_ratio=attn_reduction_ratios[1],
        stochastic_depth_rate=dpr[1],
        name="stage_2",
    )(fmap)
    fmap = LayerNormalization()(fmap)
    fmap = SquareReshape()(fmap)

    fmap = OverlapPatchEmbed(
        patch_size=patch_size[2], strides=strides[2], emb_dim=emb_dims[2]
    )(fmap)

    fmap = StageBlock(
        fc_units=emb_dims[2],
        depth=depths[2],
        num_heads=num_heads[2],
        mlp_ratio=mlp_ratios[2],
        attn_drop_prob=attn_drop_prob,
        proj_drop_prob=proj_drop_prob,
        attn_reduction_ratio=attn_reduction_ratios[2],
        stochastic_depth_rate=dpr[2],
        name="stage_3",
    )(fmap)
    fmap = LayerNormalization()(fmap)
    fmap = SquareReshape()(fmap)

    fmap = OverlapPatchEmbed(
        patch_size=patch_size[3], strides=strides[3], emb_dim=emb_dims[3]
    )(fmap)

    fmap = StageBlock(
        fc_units=emb_dims[3],
        depth=depths[3],
        num_heads=num_heads[3],
        mlp_ratio=mlp_ratios[3],
        attn_drop_prob=attn_drop_prob,
        proj_drop_prob=proj_drop_prob,
        attn_reduction_ratio=attn_reduction_ratios[3],
        stochastic_depth_rate=dpr[3],
        name="stage_4",
    )(fmap)
    fmap = LayerNormalization()(fmap)
    fmap = SquareReshape()(fmap)

    return Model(img_input, fmap)


if __name__ == "__main__":

    fmap = np.random.rand(1, 224, 224, 3)

    # patch_size = [7, 3, 3, 3]  # K
    # strides = [4, 2, 2, 2]  # S
    # emb_dims = [32, 64, 160, 256]  # C
    # attn_reduction_ratios = [8, 4, 2, 1]  # R
    # num_heads = [1, 2, 5, 8]  # N
    # mlp_ratios = [8, 8, 4, 4]  # E
    # depths = [2, 2, 2, 2]  # L
    # proj_drop_prob = 0
    # attn_drop_prob = 0
    # stochastic_depth_rate = 0

    # patch_size = [7, 3, 3, 3]  # K
    # strides = [4, 2, 2, 2]  # S
    # emb_dims = [64, 128, 320, 512]  # C
    # attn_reduction_ratios = [8, 4, 2, 1]  # R
    # num_heads = [1, 2, 5, 8]  # N
    # mlp_ratios = [8, 8, 4, 4]  # E
    # depths = [2, 2, 2, 2]  # L
    # proj_drop_prob = 0
    # attn_drop_prob = 0
    # stochastic_depth_rate = 0

    # B2 24.2
    patch_size = [7, 3, 3, 3]  # K
    strides = [4, 2, 2, 2]  # S
    emb_dims = [64, 128, 320, 512]  # C
    attn_reduction_ratios = [8, 4, 2, 1]  # R
    num_heads = [1, 2, 5, 8]  # N
    mlp_ratios = [8, 8, 4, 4]  # E
    depths = [3, 3, 6, 3]  # L
    proj_drop_prob = 0
    attn_drop_prob = 0
    stochastic_depth_rate = 0

    # fmap = OverlapPatchEmbed(
    #     patch_size=patch_size[0], strides=patch_size[0], emb_dim=emb_dims[0]
    # )(fmap)

    # out = StageBlock(
    #     fc_units=emb_dims[0],
    #     depth=depths[0],  # L
    #     num_heads=num_heads[0],  # N
    #     mlp_ratio=mlp_ratios[0],  # E
    #     attn_drop_prob=attn_drop_prob,
    #     proj_drop_prob=proj_drop_prob,
    #     attn_reduction_ratio=attn_reduction_ratios[0],  # R
    #     stochastic_depth_rate=0,
    #     name="stage_1",
    # )(fmap)

    # out = StageBlock(fc_units=16, depth=4)(fmap)
    model = get_mix_vision_transformer(
        img_shape=[224, 224, 3],
        patch_size=patch_size,
        strides=strides,
        emb_dims=emb_dims,
        num_heads=num_heads,
        mlp_ratios=mlp_ratios,
        proj_drop_prob=proj_drop_prob,
        attn_drop_prob=attn_drop_prob,
        stochastic_depth_rate=stochastic_depth_rate,
        attn_reduction_ratios=attn_reduction_ratios,
        depths=depths,
    )
    out = model(fmap)
    print(f"{out.shape.as_list()}")
    model.summary()
