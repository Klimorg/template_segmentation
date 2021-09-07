from typing import List, Tuple

import numpy as np
import tensorflow as tf
from loguru import logger
from tensorflow.keras import Model
from tensorflow.keras.layers import (
    Activation,
    Add,
    BatchNormalization,
    Concatenate,
    Conv2D,
    DepthwiseConv2D,
    GlobalAveragePooling2D,
    Input,
    ReLU,
    Reshape,
)


def ghost_module(
    fmap: tf.Tensor,
    out: int,
    ratio: int,
    convkernel: Tuple[int, int],
    dwkernel: Tuple[int, int],
    name: str,
    l2_regul: float = 1e-4,
) -> tf.Tensor:
    """Primary module of the GhostNet architecture.

    Args:
        fmap (tf.Tensor): Input feature map of the module, size = $(H,W,C)$.
        out (int): Number of channels of the output feature map.
        ratio (int): Define the ratio between the number of filters of the Conv2D layer
            and the number of filters of the `DepthwiseConv2D` in the last `Concatenate`
            layer. `depth_multiplier` of the `DepthwiseConv2D` layer is also defined as
            `ratio-1`.
        convkernel (Tuple[int, int]): Number of convolution kernels in the `Conv2D` layer.
        dwkernel (Tuple[int, int]): Number of convolution kernels in the `DepthwiseConv2D` layer.
        name (str): Name of the module.
        l2_regul (float, optional): Value of the constraint used for the
            $L_2$ regularization. Defaults to 1e-4.

    Returns:
        Output feature map, size = $(H,W,\mathrm{out})$
    """

    filters = int(np.ceil(out / ratio))
    channels = int(out - filters)

    fmap = Conv2D(
        filters=filters,
        kernel_size=convkernel,
        strides=(1, 1),
        padding="same",
        use_bias=False,
        kernel_initializer="he_uniform",
        kernel_regularizer=tf.keras.regularizers.l2(l2=l2_regul),
        name=f"conv_ghost_module_{name}",
    )(fmap)

    dwfmap = DepthwiseConv2D(
        kernel_size=dwkernel,
        strides=(1, 1),
        padding="same",
        depth_multiplier=ratio - 1,
        use_bias=False,
        depthwise_initializer="he_uniform",
        depthwise_regularizer=tf.keras.regularizers.l2(l2=l2_regul),
        name=f"depthconv_ghost_module_{name}",
    )(fmap)

    return Concatenate(axis=-1, name=f"concat_ghost_module_{name}")(
        [fmap, dwfmap[:, :, :, :channels]]
    )


def se_module(
    fmap_in: tf.Tensor,
    ratio: int,
    filters: int,
    name: str,
    l2_regul: float = 1e-4,
) -> tf.Tensor:
    """Squeeze-and-Excitation Module.

    Architecture:
        ![architecture](./images/se_module.svg)

        Source : [ArXiv link](https://arxiv.org/abs/1709.01507)

    Args:
        fmap_in (tf.Tensor): Input feature map of the module, size = $(H,W,C)$.
        ratio (int): Define the ratio of filters used in the squeeze operation of the modle (the first Conv2D).
        filters (int): Numbers of filters used in the excitation operation of the module (the second Conv2D).
        name (str): Name of the module.
        l2_regul (float, optional): Value of the constraint used for the
            $L_2$ regularization. Defaults to 1e-4.
    Returns:
        Output feature map, size = $(H,W,C)$.
    """

    channels = int(fmap_in.shape[-1])

    fmap = GlobalAveragePooling2D(name=f"gap_se_module_{name}")(fmap_in)
    fmap = Reshape((1, 1, channels), name=f"reshape_se_module_{name}")(fmap)

    fmap = Conv2D(
        filters=int(filters / ratio),
        kernel_size=(1, 1),
        strides=(1, 1),
        padding="same",
        use_bias=False,
        kernel_initializer="he_uniform",
        kernel_regularizer=tf.keras.regularizers.l2(l2=l2_regul),
        name=f"conv1_se_module_{name}",
    )(fmap)
    fmap = ReLU(name=f"relu_se_module_{name}")(fmap)
    fmap = Conv2D(
        filters=filters,
        kernel_size=(1, 1),
        strides=(1, 1),
        padding="same",
        use_bias=False,
        kernel_initializer="he_uniform",
        kernel_regularizer=tf.keras.regularizers.l2(l2=l2_regul),
        name=f"conv2_se_module_{name}",
    )(fmap)

    excitation = Activation("sigmoid", name=f"sigmoid_se_module_{name}")(fmap)

    return fmap_in * excitation


def ghost_bottleneck_module(
    fmap_in: tf.Tensor,
    dwkernel: int,
    strides: int,
    exp: int,
    out: int,
    ratio: int,
    use_se: bool,
    name: str,
    l2_regul: float = 1e-4,
) -> tf.Tensor:
    """Ghost Bottleneck Module, the backbone of the GhostNet model.

    Args:
        fmap_in (tf.Tensor): Input feature map of the module, size = $(H,W,C)$.
        dwkernel (int): Number of convolution kernels in the `DepthwiseConv2D` layer.
        strides (int): Stride used in the `DepthwiseConv2D` layers.
        exp (int): Number of filters used as an expansion operation in the first `ghost_module`.
        out (int): Number of filters/channels of the output feature map.
        ratio (int): Define the ratio in the `ghost_module` between the number of filters of the Conv2D layer
            and the number of filters of the `DepthwiseConv2D` in the last `Concatenate`
            layer. `depth_multiplier` of the `DepthwiseConv2D` layer is also defined as
            `ratio-1`.
        use_se (bool): Determine whether or not use a squeeze-and-excitation module before
            the last `ghost_module` layer.
        name (str): Name of the module.
        l2_regul (float, optional): Value of the constraint used for the
            $L_2$ regularization. Defaults to 1e-4.
    Returns:
        Output feature map, size = $(H,W,\mathrm{out})$.
    """

    fmap_shortcut = DepthwiseConv2D(
        kernel_size=dwkernel,
        strides=strides,
        padding="same",
        depth_multiplier=ratio - 1,
        activation=None,
        use_bias=False,
        depthwise_initializer="he_uniform",
        depthwise_regularizer=tf.keras.regularizers.l2(l2=l2_regul),
        name=f"depthconv_gbneck_module_{name}",
    )(fmap_in)
    fmap_shortcut = BatchNormalization(name=f"bn1_gbneck_module_{name}")(fmap_shortcut)

    fmap_shortcut = Conv2D(
        filters=out,
        kernel_size=(1, 1),
        strides=(1, 1),
        padding="same",
        activation=None,
        use_bias=False,
        kernel_initializer="he_uniform",
        kernel_regularizer=tf.keras.regularizers.l2(l2=l2_regul),
        name=f"conv_gbneck_module_{name}",
    )(fmap_shortcut)
    fmap_shortcut = BatchNormalization(name=f"bn2_gbneck_module_{name}")(fmap_shortcut)

    fmap = ghost_module(
        fmap=fmap_in,
        out=exp,
        ratio=ratio,
        convkernel=(1, 1),
        dwkernel=(3, 3),
        name=f"1_gbneck_module_{name}",
    )
    fmap = BatchNormalization(name=f"bn3_gbneck_module_{name}")(fmap)
    fmap = ReLU(name=f"relu_gbneck_module_{name}")(fmap)

    if strides > 1:
        fmap = DepthwiseConv2D(
            kernel_size=dwkernel,
            strides=strides,
            padding="same",
            depth_multiplier=ratio - 1,
            activation=None,
            use_bias=False,
            depthwise_initializer="he_uniform",
            depthwise_regularizer=tf.keras.regularizers.l2(l2=l2_regul),
            name=f"depthconv_s2_gbneck_module_{name}",
        )(fmap)
        fmap = BatchNormalization(name=f"bn4_gbneck_module_{name}")(fmap)

    if use_se:
        fmap = se_module(
            fmap_in=fmap, filters=exp, ratio=ratio, name=f"gbneck_module_{name}"
        )

    fmap = ghost_module(
        fmap=fmap,
        out=out,
        ratio=ratio,
        convkernel=(1, 1),
        dwkernel=(3, 3),
        name=f"2_gbneck_module_{name}",
    )
    fmap = BatchNormalization(name=f"bn5_gbneck_module_{name}")(fmap)

    return Add(name=f"add_gbneck_module_{name}")([fmap_shortcut, fmap])


def get_ghostnet(
    img_shape: List[int],
) -> tf.keras.Model:
    """Instantiate a GhostNet model.

    Args:
        img_shape (List[int]): Input shape of the images in the dataset.

    Returns:
        A `tf.keras` model.
    """

    dwkernels = [3, 3, 3, 5, 5, 3, 3, 3, 3, 3, 3, 5, 5, 5, 5, 5]
    strides = [1, 2, 1, 2, 1, 2, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1]
    exps = [16, 48, 72, 72, 120, 240, 200, 184, 184, 480, 672, 672, 960, 960, 960, 960]
    outs = [16, 24, 24, 40, 40, 80, 80, 80, 80, 112, 112, 160, 160, 160, 160, 160]
    ratios = [2] * 16
    use_ses = [
        False,
        False,
        False,
        True,
        True,
        False,
        False,
        False,
        False,
        True,
        True,
        True,
        False,
        True,
        False,
        True,
        False,
    ]
    l2_reguls = [1e-4] * 16

    img_input = Input(img_shape)

    fmap = Conv2D(
        filters=16,
        kernel_size=(3, 3),
        strides=(2, 2),
        padding="same",
        use_bias=False,
        kernel_initializer="he_uniform",
    )(img_input)

    fmap = ghost_bottleneck_module(
        fmap_in=fmap,
        dwkernel=dwkernels[0],
        strides=strides[0],
        exp=exps[0],
        out=outs[0],
        ratio=ratios[0],
        use_se=use_ses[0],
        l2_regul=l2_reguls[0],
        name="1",
    )
    fmap = ghost_bottleneck_module(
        fmap_in=fmap,
        dwkernel=dwkernels[1],
        strides=strides[1],
        exp=exps[1],
        out=outs[1],
        ratio=ratios[1],
        use_se=use_ses[1],
        l2_regul=l2_reguls[1],
        name="1_2",
    )

    fmap = ghost_bottleneck_module(
        fmap_in=fmap,
        dwkernel=dwkernels[2],
        strides=strides[2],
        exp=exps[2],
        out=outs[2],
        ratio=ratios[2],
        use_se=use_ses[2],
        l2_regul=l2_reguls[2],
        name="2_1",
    )
    fmap = ghost_bottleneck_module(
        fmap_in=fmap,
        dwkernel=dwkernels[3],
        strides=strides[3],
        exp=exps[3],
        out=outs[3],
        ratio=ratios[3],
        use_se=use_ses[3],
        l2_regul=l2_reguls[3],
        name="2_2",
    )

    fmap = ghost_bottleneck_module(
        fmap_in=fmap,
        dwkernel=dwkernels[4],
        strides=strides[4],
        exp=exps[4],
        out=outs[4],
        ratio=ratios[4],
        use_se=use_ses[4],
        l2_regul=l2_reguls[4],
        name="3_1",
    )
    fmap = ghost_bottleneck_module(
        fmap_in=fmap,
        dwkernel=dwkernels[5],
        strides=strides[5],
        exp=exps[5],
        out=outs[5],
        ratio=ratios[5],
        use_se=use_ses[5],
        l2_regul=l2_reguls[5],
        name="3_2",
    )

    fmap = ghost_bottleneck_module(
        fmap_in=fmap,
        dwkernel=dwkernels[6],
        strides=strides[6],
        exp=exps[6],
        out=outs[6],
        ratio=ratios[6],
        use_se=use_ses[6],
        l2_regul=l2_reguls[6],
        name="4_1",
    )
    fmap = ghost_bottleneck_module(
        fmap_in=fmap,
        dwkernel=dwkernels[7],
        strides=strides[7],
        exp=exps[7],
        out=outs[7],
        ratio=ratios[7],
        use_se=use_ses[7],
        l2_regul=l2_reguls[7],
        name="4_2",
    )
    fmap = ghost_bottleneck_module(
        fmap_in=fmap,
        dwkernel=dwkernels[8],
        strides=strides[8],
        exp=exps[8],
        out=outs[8],
        ratio=ratios[8],
        use_se=use_ses[8],
        l2_regul=l2_reguls[8],
        name="4_3",
    )
    fmap = ghost_bottleneck_module(
        fmap_in=fmap,
        dwkernel=dwkernels[9],
        strides=strides[9],
        exp=exps[9],
        out=outs[9],
        ratio=ratios[9],
        use_se=use_ses[9],
        l2_regul=l2_reguls[9],
        name="4_4",
    )
    fmap = ghost_bottleneck_module(
        fmap_in=fmap,
        dwkernel=dwkernels[10],
        strides=strides[10],
        exp=exps[10],
        out=outs[10],
        ratio=ratios[10],
        use_se=use_ses[10],
        l2_regul=l2_reguls[10],
        name="4_5",
    )
    fmap = ghost_bottleneck_module(
        fmap_in=fmap,
        dwkernel=dwkernels[11],
        strides=strides[11],
        exp=exps[11],
        out=outs[11],
        ratio=ratios[11],
        use_se=use_ses[11],
        l2_regul=l2_reguls[11],
        name="4_6",
    )

    fmap = ghost_bottleneck_module(
        fmap_in=fmap,
        dwkernel=dwkernels[12],
        strides=strides[12],
        exp=exps[12],
        out=outs[12],
        ratio=ratios[12],
        use_se=use_ses[12],
        l2_regul=l2_reguls[12],
        name="5_1",
    )
    fmap = ghost_bottleneck_module(
        fmap_in=fmap,
        dwkernel=dwkernels[13],
        strides=strides[13],
        exp=exps[13],
        out=outs[13],
        ratio=ratios[13],
        use_se=use_ses[13],
        l2_regul=l2_reguls[13],
        name="5_2",
    )
    fmap = ghost_bottleneck_module(
        fmap_in=fmap,
        dwkernel=dwkernels[14],
        strides=strides[14],
        exp=exps[14],
        out=outs[14],
        ratio=ratios[14],
        use_se=use_ses[14],
        l2_regul=l2_reguls[14],
        name="5_3",
    )
    fmap = ghost_bottleneck_module(
        fmap_in=fmap,
        dwkernel=dwkernels[15],
        strides=strides[15],
        exp=exps[15],
        out=outs[15],
        ratio=ratios[15],
        use_se=use_ses[15],
        l2_regul=l2_reguls[15],
        name="5_4",
    )

    return Model(img_input, fmap)


def get_backbone(img_shape: List[int], backbone_name: str) -> tf.keras.Model:
    """Instantiate the model and use it as a backbone (feature extractor) for a semantic segmentation task.

    Args:
        img_shape (List[int]): Input shape of the images/masks in the dataset.
        backbone_name (str): Name of the backbone.

    Returns:
        A `tf.keras` model.
    """

    backbone = get_ghostnet(
        img_shape=img_shape,
    )

    endpoint_layers = [
        "concat_ghost_module_2_gbneck_module_1_2",
        "concat_ghost_module_2_gbneck_module_2_2",
        "concat_ghost_module_2_gbneck_module_3_2",
        "concat_ghost_module_2_gbneck_module_5_4",
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
