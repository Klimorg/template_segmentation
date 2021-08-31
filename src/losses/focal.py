import itertools

import tensorflow as tf
from loguru import logger

_EPSILON = tf.keras.backend.epsilon()


class RetinaNetClassificationLoss(tf.losses.Loss):
    """Implements Focal loss"""

    def __init__(self, alpha, gamma):
        super(RetinaNetClassificationLoss, self).__init__(
            reduction="none", name="RetinaNetClassificationLoss"
        )
        self._alpha = alpha
        self._gamma = gamma

    def call(self, y_true, y_pred):
        axis = -1

        cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(
            labels=y_true, logits=y_pred
        )

        # logits = tf.math.log(tf.clip_by_value(y_pred, _EPSILON, 1 - _EPSILON))
        # cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
        #     labels=y_true,
        #     logits=logits,
        # )

        probs = tf.nn.sigmoid(y_pred)

        alpha = tf.where(tf.equal(y_true, 1), self._alpha, (1 - self._alpha))

        pt = tf.where(tf.equal(y_true, 1), probs, 1 - probs)

        loss = alpha * tf.pow(1.0 - pt, self._gamma) * cross_entropy

        return tf.reduce_sum(loss, axis=-1)
