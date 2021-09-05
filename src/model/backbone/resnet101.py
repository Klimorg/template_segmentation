from loguru import logger
from tensorflow import keras


def get_resnet101(img_shape: List[int], backbone_name: str):

    backbone = keras.applications.ResNet101(include_top=False, input_shape=img_shape)
    c2_output, c3_output, c4_output, c5_output = [
        backbone.get_layer(layer_name).output
        for layer_name in [
            "conv2_block3_out",
            "conv3_block4_out",
            "conv4_block23_out",
            "conv5_block3_out",
        ]
    ]
    height = img_shape[1]

    logger.info(f"c2_output OS : {int(height/c2_output.shape.as_list()[1])}")
    logger.info(f"c3_output OS : {int(height/c3_output.shape.as_list()[1])}")
    logger.info(f"c4_output OS : {int(height/c4_output.shape.as_list()[1])}")
    logger.info(f"c5_output OS : {int(height/c5_output.shape.as_list()[1])}")

    return keras.Model(
        inputs=[backbone.inputs],
        outputs=[c2_output, c3_output, c4_output, c5_output],
        name=backbone_name,
    )
