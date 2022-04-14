import tensorflow as tf
from tensorflow.keras.losses import Loss


class FocalLoss(Loss):
    def __init__(
        self,
        # num_classes: int,
        gamma: float = 2,
        alpha: float = 0.75,
        eps: float = 1e-7,
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        # self.num_classes = num_classes
        self.gamma = gamma
        self.alpha = alpha
        self.eps = eps

    def call(self, y_true, y_pred):
        logits = tf.math.log(tf.clip_by_value(y_pred, self.eps, 1 - self.eps))

        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=y_true,
            logits=logits,
        )

        loss = self.alpha * loss * (1 - logits) ** self.gamma  # focal loss

        return tf.reduce_sum(loss)

    def get_config(self):
        base_config = super().get_config()
        return {
            **base_config,
            "gamma": self.gamma,
            "alpha": self.alpha,
            "eps": self.eps,
        }
