from typing import Any, Dict, Optional, Tuple

import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow import keras
from tensorflow.keras.layers import Dense, Dropout, Layer

# patch_size = (2, 2)  # 2-by-2 sized patches
# dropout_rate = 0.03  # Dropout rate
# num_heads = 8  # Attention heads
# embed_dim = 64  # Embedding dimension
# num_mlp = 256  # MLP layer size
# qkv_bias = True  # Convert embedded patches to query, key, and values with a learnable additive value
# window_size = 2  # Size of attention window
# shift_size = 1  # Size of shifting window
# image_dimension = 32  # Initial image size

# num_patch_x = input_shape[0] // patch_size[0]
# num_patch_y = input_shape[1] // patch_size[1]

# learning_rate = 1e-3
# batch_size = 128
# num_epochs = 40
# validation_split = 0.1
# weight_decay = 0.0001
# label_smoothing = 0.1


def window_partition(x, window_size):
    _, height, width, channels = x.shape
    patch_num_y = height // window_size
    patch_num_x = width // window_size
    x = tf.reshape(
        x,
        shape=(-1, patch_num_y, window_size, patch_num_x, window_size, channels),
    )
    x = tf.transpose(x, (0, 1, 3, 2, 4, 5))
    windows = tf.reshape(x, shape=(-1, window_size, window_size, channels))
    return windows


def window_reverse(windows, window_size, height, width, channels):
    patch_num_y = height // window_size
    patch_num_x = width // window_size
    x = tf.reshape(
        windows,
        shape=(-1, patch_num_y, patch_num_x, window_size, window_size, channels),
    )
    x = tf.transpose(x, perm=(0, 1, 3, 2, 4, 5))
    x = tf.reshape(x, shape=(-1, height, width, channels))
    return x


# a remplacer par StochasticDepth(tf.keras.layers.Layer) dans convmlp
@tf.keras.utils.register_keras_serializable()
class DropPath(Layer):
    def __init__(self, drop_prob: Optional[float] = None, **kwargs):
        super(DropPath, self).__init__(**kwargs)
        self.drop_prob = drop_prob

    def call(self, x):
        input_shape = tf.shape(x)
        batch_size = input_shape[0]
        rank = x.shape.rank
        shape = (batch_size,) + (1,) * (rank - 1)
        random_tensor = (1 - self.drop_prob) + tf.random.uniform(shape, dtype=x.dtype)
        path_mask = tf.floor(random_tensor)
        output = tf.math.divide(x, 1 - self.drop_prob) * path_mask
        return output

    def get_config(self) -> Dict[str, Any]:

        config = super().get_config()
        config.update({"drop_prob": self.drop_prob})
        return config


class WindowAttention(Layer):
    def __init__(
        self,
        dim: int,
        window_size: Tuple[int, int],
        num_heads: int,
        qkv_bias: bool = True,
        dropout_rate: float = 0.0,
        l2_regul: float = 1e-4,
        *args,
        **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)

        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        self.l2_regul = l2_regul

        self.scale = (dim // num_heads) ** -0.5
        self.qkv = Dense(
            units=dim * 3,
            use_bias=qkv_bias,
            kernel_initializer="he_uniform",
            kernel_regularizer=tf.keras.regularizers.l2(l2=self.l2_regul),
        )
        self.dropout = Dropout(dropout_rate)
        self.proj = Dense(
            units=dim,
            kernel_initializer="he_uniform",
            kernel_regularizer=tf.keras.regularizers.l2(l2=self.l2_regul),
        )

    def build(self, input_shape) -> None:
        num_window_elements = (2 * self.window_size[0] - 1) * (
            2 * self.window_size[1] - 1
        )
        self.relative_position_bias_table = self.add_weight(
            shape=(num_window_elements, self.num_heads),
            initializer=tf.initializers.Zeros(),
            trainable=True,
        )
        coords_h = np.arange(self.window_size[0])
        coords_w = np.arange(self.window_size[1])
        coords_matrix = np.meshgrid(coords_h, coords_w, indexing="ij")
        coords = np.stack(coords_matrix)
        coords_flatten = coords.reshape(2, -1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.transpose([1, 2, 0])
        relative_coords[:, :, 0] += self.window_size[0] - 1
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)

        self.relative_position_index = tf.Variable(
            initial_value=tf.convert_to_tensor(relative_position_index),
            trainable=False,
        )

    def call(self, x, mask=None):
        _, size, channels = x.shape
        head_dim = channels // self.num_heads
        x_qkv = self.qkv(x)
        x_qkv = tf.reshape(x_qkv, shape=(-1, size, 3, self.num_heads, head_dim))
        x_qkv = tf.transpose(x_qkv, perm=(2, 0, 3, 1, 4))
        q, k, v = x_qkv[0], x_qkv[1], x_qkv[2]
        q = q * self.scale
        k = tf.transpose(k, perm=(0, 1, 3, 2))
        attn = q @ k

        num_window_elements = self.window_size[0] * self.window_size[1]
        relative_position_index_flat = tf.reshape(
            self.relative_position_index,
            shape=(-1,),
        )
        relative_position_bias = tf.gather(
            self.relative_position_bias_table,
            relative_position_index_flat,
        )
        relative_position_bias = tf.reshape(
            relative_position_bias,
            shape=(num_window_elements, num_window_elements, -1),
        )
        relative_position_bias = tf.transpose(relative_position_bias, perm=(2, 0, 1))
        attn = attn + tf.expand_dims(relative_position_bias, axis=0)

        if mask is not None:
            nW = mask.get_shape()[0]
            mask_float = tf.cast(
                tf.expand_dims(tf.expand_dims(mask, axis=1), axis=0),
                tf.float32,
            )
            attn = (
                tf.reshape(attn, shape=(-1, nW, self.num_heads, size, size))
                + mask_float
            )
            attn = tf.reshape(attn, shape=(-1, self.num_heads, size, size))
            attn = keras.activations.softmax(attn, axis=-1)
        else:
            attn = keras.activations.softmax(attn, axis=-1)
        attn = self.dropout(attn)

        x_qkv = attn @ v
        x_qkv = tf.transpose(x_qkv, perm=(0, 2, 1, 3))
        x_qkv = tf.reshape(x_qkv, shape=(-1, size, channels))
        x_qkv = self.proj(x_qkv)
        x_qkv = self.dropout(x_qkv)
        return x_qkv
