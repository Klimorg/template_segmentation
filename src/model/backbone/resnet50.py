from typing import List

from loguru import logger
from tensorflow import keras


def get_backbone(img_shape: List[int]):

    backbone = keras.applications.ResNet50(include_top=False, input_shape=img_shape)

    endpoint_layers = [
        "conv2_block3_out",
        "conv3_block4_out",
        "conv4_block6_out",
        "conv5_block3_out",
    ]

    c2_output, c3_output, c4_output, c5_output = [
        backbone.get_layer(layer_name).output for layer_name in endpoint_layers
    ]

    height = img_shape[1]

    logger.info(f"c2_output OS : {int(height/c2_output.shape.as_list()[1])}")
    logger.info(f"c3_output OS : {int(height/c3_output.shape.as_list()[1])}")
    logger.info(f"c4_output OS : {int(height/c4_output.shape.as_list()[1])}")
    logger.info(f"c5_output OS : {int(height/c5_output.shape.as_list()[1])}")

    return keras.Model(
        inputs=[backbone.inputs], outputs=[c2_output, c3_output, c4_output, c5_output]
    )


if __name__ == "__main__":

    import numpy as np

    img_shape = [224, 224, 3]
    tensor = np.random.rand(1, img_shape[0], img_shape[1], img_shape[2])

    model = get_backbone(img_shape)
