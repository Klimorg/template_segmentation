from typing import Any, Dict, List

import numpy as np
import tensorflow as tf
from loguru import logger
from tensorflow.keras import Model
from tensorflow.keras.activations import swish
from tensorflow.keras.layers import (
    Concatenate,
    Conv2D,
    Dense,
    Input,
    LayerNormalization,
    MultiHeadAttention,
    Reshape,
)

from src.model.layers.common_layers import InvertedResidualBottleneck2D


@tf.keras.utils.register_keras_serializable()
class Transformer(tf.keras.layers.Layer):
    def __init__(
        self,
        fc1_units: int,
        fc2_units: int,
        num_heads: int = 8,
        emb_dim: int = 64,
        l2_regul: float = 1e-4,
        *args,
        **kwargs,
    ) -> None:

        super().__init__(*args, **kwargs)
        self.fc1_units = fc1_units
        self.fc2_units = fc2_units
        self.num_heads = num_heads
        self.emb_dim = emb_dim
        self.l2_regul = l2_regul

        self.act = swish

    def build(self, input_shape) -> None:

        self.norm1 = LayerNormalization()
        self.norm2 = LayerNormalization()

        self.mhsa = MultiHeadAttention(
            num_heads=self.num_heads,
            key_dim=self.emb_dim,
            use_bias=False,
            kernel_initializer="he_uniform",
            kernel_regularizer=tf.keras.regularizers.l2(l2=self.l2_regul),
        )

        self.fc1 = Dense(
            self.fc1_units,
            kernel_initializer="he_uniform",
            kernel_regularizer=tf.keras.regularizers.l2(l2=self.l2_regul),
        )

        self.fc2 = Dense(
            self.fc2_units,
            kernel_initializer="he_uniform",
            kernel_regularizer=tf.keras.regularizers.l2(l2=self.l2_regul),
        )

    def call(self, inputs, training=None) -> tf.Tensor:

        mhsa_in = self.norm1(inputs)
        mhsa_out = self.mhsa(mhsa_in, mhsa_in)
        mhsa_out = inputs + mhsa_out

        fc_in = self.norm2(mhsa_out)
        fc_in = self.fc1(fc_in)
        fc_in = self.act(fc_in)
        fc_out = self.fc2(fc_in)

        return mhsa_out + fc_out

    def get_config(self) -> Dict[str, Any]:
        config = super().get_config()
        config.update(
            {
                "fc1_units": self.fc1_units,
                "fc2_units": self.fc2_units,
                "num_heads": self.num_heads,
                "emb_dim": self.emb_dim,
                "l2_regul": self.l2_regul,
            },
        )
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


@tf.keras.utils.register_keras_serializable()
class MobileViT2D(tf.keras.layers.Layer):
    def __init__(
        self,
        expansion_rate: int,
        # fc_units: int,
        filters: int,
        patch_size: int = 2,
        num_heads: int = 8,
        emb_dim: int = 64,
        repetitions: int = 2,
        l2_regul: float = 1e-4,
        *args,
        **kwargs,
    ) -> None:

        super().__init__(*args, **kwargs)
        self.expansion_rate = expansion_rate
        self.filters = filters
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.emb_dim = emb_dim
        self.repetitions = repetitions
        self.l2_regul = l2_regul

    def build(self, input_shape) -> None:

        _, height, width, _ = input_shape
        num_patches = int(height * width // self.patch_size ** 2)

        self.unfold = Reshape(
            target_shape=(self.patch_size ** 2, num_patches, self.filters),
        )
        self.fold = Reshape(target_shape=(height, width, self.filters))

        self.conv3x3_1 = Conv2D(
            filters=self.filters,
            kernel_size=3,
            strides=1,
            padding="same",
            use_bias=False,
            kernel_initializer="he_uniform",
            kernel_regularizer=tf.keras.regularizers.l2(l2=self.l2_regul),
        )

        self.conv3x3_2 = Conv2D(
            filters=self.filters,
            kernel_size=3,
            strides=1,
            padding="same",
            use_bias=False,
            kernel_initializer="he_uniform",
            kernel_regularizer=tf.keras.regularizers.l2(l2=self.l2_regul),
        )

        self.conv1x1_1 = Conv2D(
            filters=self.filters,
            kernel_size=1,
            strides=1,
            padding="same",
            use_bias=False,
            kernel_initializer="he_uniform",
            kernel_regularizer=tf.keras.regularizers.l2(l2=self.l2_regul),
        )

        self.conv1x1_2 = Conv2D(
            filters=self.filters,
            kernel_size=1,
            strides=1,
            padding="same",
            use_bias=False,
            kernel_initializer="he_uniform",
            kernel_regularizer=tf.keras.regularizers.l2(l2=self.l2_regul),
        )

        self.concat = Concatenate(axis=-1)

        self.transformer_block = [
            Transformer(
                fc1_units=self.expansion_rate * self.emb_dim,
                fc2_units=self.filters,
                num_heads=self.num_heads,
                emb_dim=self.emb_dim,
            )
            for _ in range(self.repetitions)
        ]

    def call(self, inputs, training=None) -> tf.Tensor:

        fmap = self.conv3x3_1(inputs)
        fmap = self.conv1x1_1(fmap)

        fmap = self.unfold(fmap)

        for block in self.transformer_block:
            fmap = block(fmap)

        fmap = self.fold(fmap)

        fmap = self.conv1x1_2(fmap)
        fmap = self.concat([fmap, inputs])
        return self.conv3x3_2(fmap)

    def get_config(self) -> Dict[str, Any]:
        config = super().get_config()
        config.update(
            {
                "expansion_rate": self.expansion_rate,
                "filters": self.filters,
                "patch_size": self.patch_size,
                "num_heads": self.num_heads,
                "emb_dim": self.emb_dim,
                "repetitions": self.repetitions,
                "l2_regul": self.l2_regul,
            },
        )
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


def get_feature_extractor(
    img_shape: List[int],
    expansion_rate: List[int],
    filters: List[int],
    emb_dim: List[int],
    repetitions: List[int],
    num_heads: int,
):

    # Input block
    img_input = Input(img_shape)

    fmap = Conv2D(
        filters=filters[0],
        kernel_size=3,
        strides=2,
        padding="same",
        kernel_initializer="he_normal",
        use_bias=False,
        name="conv_input",
    )(img_input)
    fmap = swish(fmap)
    fmap = InvertedResidualBottleneck2D(
        expansion_rate=expansion_rate[0],
        filters=filters[1],
        strides=1,
        skip_connection=True,
        name="ivrb0",
    )(fmap)
    fmap = swish(fmap)

    # Block 1
    fmap = InvertedResidualBottleneck2D(
        expansion_rate=expansion_rate[0],
        filters=filters[2],
        strides=2,
        skip_connection=False,
        name="ivrb1",
    )(fmap)
    fmap = swish(fmap)
    fmap = InvertedResidualBottleneck2D(
        expansion_rate=expansion_rate[0],
        filters=filters[3],
        strides=1,
        skip_connection=True,
        name="ivrb2",
    )(fmap)
    fmap = swish(fmap)
    fmap = InvertedResidualBottleneck2D(
        expansion_rate=expansion_rate[0],
        filters=filters[4],
        strides=1,
        skip_connection=True,
        name="ivrb3",
    )(fmap)
    fmap = swish(fmap)

    # Block 2
    fmap = InvertedResidualBottleneck2D(
        expansion_rate=expansion_rate[0],
        filters=filters[5],
        strides=2,
        skip_connection=False,
        name="ivrb4",
    )(fmap)
    fmap = swish(fmap)
    fmap = MobileViT2D(
        expansion_rate=expansion_rate[1],
        filters=filters[6],
        emb_dim=emb_dim[0],
        repetitions=repetitions[0],
        num_heads=num_heads,
        name="MobileViT2D_block2",
    )(fmap)
    fmap = swish(fmap)

    # Block 3
    fmap = InvertedResidualBottleneck2D(
        expansion_rate=expansion_rate[0],
        filters=filters[7],
        strides=2,
        skip_connection=False,
        name="ivrb5",
    )(fmap)
    fmap = swish(fmap)
    fmap = MobileViT2D(
        expansion_rate=expansion_rate[1],
        filters=filters[8],
        emb_dim=emb_dim[1],
        repetitions=repetitions[1],
        num_heads=num_heads,
        name="MobileViT2D_block3",
    )(fmap)
    fmap = swish(fmap)

    # Block 4
    fmap = InvertedResidualBottleneck2D(
        expansion_rate=expansion_rate[0],
        filters=filters[9],
        strides=2,
        skip_connection=False,
        name="ivrb6",
    )(fmap)
    fmap = swish(fmap)
    fmap = MobileViT2D(
        expansion_rate=expansion_rate[1],
        filters=filters[10],
        emb_dim=emb_dim[2],
        repetitions=repetitions[2],
        num_heads=num_heads,
        name="MobileViT2D_block4",
    )(fmap)
    fmap = swish(fmap)
    fmap_out = Conv2D(
        filters=filters[11],
        kernel_size=1,
        strides=1,
        padding="same",
        kernel_initializer="he_normal",
        use_bias=False,
        name="conv_output",
    )(fmap)

    return Model(img_input, fmap_out)


def get_backbone(
    img_shape: List[int],
    expansion_rate: List[int],
    filters: List[int],
    emb_dim: List[int],
    repetitions: List[int],
    num_heads: int,
    backbone_name: str,
) -> tf.keras.Model:
    """Instantiate the model and use it as a backbone (feature extractor) for a semantic segmentation task.

    Args:
        img_shape (List[int]): [description]
        expansion_rate (List[int]): [description]
        filters (List[int]): [description]
        emb_dim (List[int]): [description]
        repetitions (List[int]): [description]
        num_heads (int): [description]
        backbone_name (str): [description]

    Returns:
        A `tf.keras` model.
    """

    backbone = get_feature_extractor(
        img_shape=img_shape,
        expansion_rate=expansion_rate,
        filters=filters,
        emb_dim=emb_dim,
        repetitions=repetitions,
        num_heads=num_heads,
    )

    endpoint_layers = [
        "ivrb3",
        "MobileViT2D_block2",
        "MobileViT2D_block3",
        "conv_output",
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
    fmap = np.random.rand(1, 256, 256, 3)

    # MobileVit-XXS
    # expansion_rate = [2, 2]
    # filters = [16, 16, 24, 24, 24, 48, 48, 64, 64, 80, 80, 320]
    # emb_dim = [64, 80, 96]
    # repetitions = [2, 4, 3]
    # num_heads = 2

    # MobileVit-XS
    # expansion_rate = [4, 2]
    # filters = [16, 16, 48, 48, 48, 64, 64, 80, 80, 96, 96, 384]
    # emb_dim = [96, 120, 144]
    # repetitions = [2, 4, 3]
    # num_heads = 2

    # MobileVit-S
    expansion_rate = [4, 2]
    filters = [16, 16, 64, 64, 64, 96, 96, 128, 128, 160, 160, 640]
    emb_dim = [144, 192, 240]
    repetitions = [2, 4, 3]
    num_heads = 2

    model = get_backbone(
        img_shape=(256, 256, 3),
        expansion_rate=expansion_rate,
        filters=filters,
        emb_dim=emb_dim,
        repetitions=repetitions,
        num_heads=num_heads,
        backbone_name="MobileViT-S",
    )
    out = model(fmap)
    model.summary()
