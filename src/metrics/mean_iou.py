import tensorflow as tf


@tf.keras.utils.register_keras_serializable()
class UpdatedMeanIoU(tf.keras.metrics.MeanIoU):
    def __init__(
        self,
        num_classes: int,
        y_true: tf.Tensor = None,
        y_pred: tf.Tensor = None,
        name: str = "mean_iou",
        dtype=None,
    ):
        """Modified Mean IoU (Intersection over Union) to work for sparse labels.

        We have to modify the definition of the Mean IoU if we want to consider sparse outputs for the segmentation.
        From [StackOverflow](https://stackoverflow.com/questions/61824470/dimensions-mismatch-error-when-using-tf-metrics-meaniou-with-sparsecategorical).


        Args:
            y_true (tf.Tensor, optional): The ground truth values. Defaults to None.
            y_pred (tf.Tensor, optional): The predicted labels. Defaults to None.
            num_classes (int, optional): Number of classes for the segmentation task. Defaults to None.
            name (str, optional): Name of the metric. Defaults to "mean_iou".
            dtype ([type], optional): . Defaults to None.
        """
        super().__init__(num_classes=num_classes, name=name, dtype=dtype)

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred = tf.math.argmax(y_pred, axis=-1)
        return super().update_state(y_true, y_pred, sample_weight)

    # A voir si ça marche
    def get_config(self):

        config = super().get_config()
        return config
