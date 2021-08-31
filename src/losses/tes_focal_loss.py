"""Multiclass focal loss implementation."""
#    __                          _     _
#   / _|                        | |   | |
#  | |_    ___     ___    __ _  | |   | |   ___    ___   ___
#  |  _|  / _ \   / __|  / _` | | |   | |  / _ \  / __| / __|
#  | |   | (_) | | (__  | (_| | | |   | | | (_) | \__ \ \__ \
#  |_|    \___/   \___|  \__,_| |_|   |_|  \___/  |___/ |___/

# from https://github.com/artemmavrin/focal-loss/blob/master/src/focal_loss/_categorical_focal_loss.py

import itertools
from typing import Any, Optional

import tensorflow as tf

_EPSILON = tf.keras.backend.epsilon()


def sparse_categorical_focal_loss(
    y_true: tf.Tensor,
    y_pred: tf.Tensor,
    alpha: float,
    gamma: float,
    *,
    class_weight: Optional[Any] = None,
    from_logits: bool = False,
    axis: int = -1,
) -> tf.Tensor:
    """
    Focal loss function for multiclass classification with integer labels.

    This loss function generalizes multiclass softmax cross-entropy by
    introducing a hyperparameter called the *focusing parameter* that allows
    hard-to-classify examples to be penalized more heavily relative to
    easy-to-classify examples.

    In the multiclass setting, with integer labels $y$, focal loss is
    defined as

    \[
        L(y, \hat{\mathbf{p}})
        = -(1 - \hat{p}_y )^\gamma \log(\hat{p}_y)
    \]

    where

    *   $y \in \{0, \ldots, K - 1\}$ is an integer class label ($K$
        denotes the number of classes),
    *   $\hat{\mathbf{p}} = (\hat{p}_0, \ldots, \hat{p}_{K-1})
        \in [0, 1]^K$ is a vector representing an estimated probability
        distribution over the $K$ classes,
    *   $\gamma$ (gamma, not $y$) is the *focusing parameter* that
        specifies how much higher-confidence correct predictions contribute to
        the overall loss (the higher the $\gamma$, the higher the rate at
        which easy-to-classify examples are down-weighted).

    The usual multiclass softmax cross-entropy loss is recovered by setting
    $\gamma = 0$.

    Args:
        y_true (tf.Tensor): Integer class labels.
        y_pred (tf.Tensor): Either probabilities or logits, depending on the `from_logits`
            parameter.
        alpha (float):
        gamma (float): The focusing parameter $\gamma$. Higher values of `gamma` make
            easy-to-classify examples contribute less to the loss relative to
            hard-to-classify examples. Must be non-negative. This can be a
            one-dimensional tensor, in which case it specifies a focusing parameter
            for each class.
        class_weight (Optional[Any], optional): Weighting factor for each of the $k$ classes. If not specified,
            then all classes are weighted equally. Defaults to None.
        from_logits (bool, optional): Whether `y_pred` contains logits or probabilities.. Defaults to False.
        axis (int, optional): Channel axis in the `y_pred` tensor.. Defaults to -1.


    Returns:
        tf.Tensor: The focal loss for each example.

    Examples:
        This function computes the per-example focal loss between a one-dimensional
        integer label vector and a two-dimensional prediction matrix:
        >>> import numpy as np
        >>> from focal_loss import sparse_categorical_focal_loss
        >>> y_true = [0, 1, 2]
        >>> y_pred = [[0.8, 0.1, 0.1], [0.2, 0.7, 0.1], [0.2, 0.2, 0.6]]
        >>> loss = sparse_categorical_focal_loss(y_true, y_pred, gamma=2)
        >>> np.set_printoptions(precision=3)
        >>> print(loss.numpy())
        [0.009 0.032 0.082]

    Warnings:
        This function does not reduce its output to a scalar, so it cannot be passed
        to `tf.keras.Model.compile` as a `loss` argument. Instead, use the
        wrapper class `focal_loss.SparseCategoricalFocalLoss`.

    References:
        T. Lin, P. Goyal, R. Girshick, K. He and P. Dollár. Focal loss for
        dense object detection. IEEE Transactions on Pattern Analysis and
        Machine Intelligence, 2018.
        (`DOI <https://doi.org/10.1109/TPAMI.2018.2858826>`__)
        (`arXiv preprint <https://arxiv.org/abs/1708.02002>`__)
    """
    # Process focusing parameter
    gamma = tf.convert_to_tensor(gamma, dtype=tf.dtypes.float32)
    gamma_rank = gamma.shape.rank
    scalar_gamma = gamma_rank == 0

    # Process class weight
    if class_weight is not None:
        class_weight = tf.convert_to_tensor(class_weight, dtype=tf.dtypes.float32)

    # Process prediction tensor
    y_pred = tf.convert_to_tensor(y_pred)
    y_pred_rank = y_pred.shape.rank
    if y_pred_rank is not None:
        axis %= y_pred_rank
        if axis != y_pred_rank - 1:
            # Put channel axis last for sparse_softmax_cross_entropy_with_logits
            perm = list(
                itertools.chain(range(axis), range(axis + 1, y_pred_rank), [axis])
            )
            y_pred = tf.transpose(y_pred, perm=perm)
    elif axis != -1:
        raise ValueError(
            f"Cannot compute sparse categorical focal loss with axis={axis} on "
            "a prediction tensor with statically unknown rank."
        )
    y_pred_shape = tf.shape(y_pred)

    # Process ground truth tensor
    y_true = tf.dtypes.cast(y_true, dtype=tf.dtypes.int64)
    y_true_rank = y_true.shape.rank

    if y_true_rank is None:
        raise NotImplementedError(
            "Sparse categorical focal loss not supported "
            "for target/label tensors of unknown rank"
        )

    reshape_needed = (
        y_true_rank is not None
        and y_pred_rank is not None
        and y_pred_rank != y_true_rank + 1
    )
    if reshape_needed:
        y_true = tf.reshape(y_true, [-1])
        y_pred = tf.reshape(y_pred, [-1, y_pred_shape[-1]])

    if from_logits:
        logits = y_pred
        probs = tf.nn.softmax(y_pred, axis=-1)
    else:
        probs = y_pred
        logits = tf.math.log(tf.clip_by_value(y_pred, _EPSILON, 1 - _EPSILON))

    xent_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=y_true,
        logits=logits,
    )

    y_true_rank = y_true.shape.rank

    probs = tf.gather(probs, y_true, axis=-1, batch_dims=y_true_rank)
    if not scalar_gamma:
        gamma = tf.gather(gamma, y_true, axis=0, batch_dims=y_true_rank)

    focal_modulation = (1 - probs) ** gamma
    loss = alpha * focal_modulation * xent_loss

    if class_weight is not None:
        class_weight = tf.gather(class_weight, y_true, axis=0, batch_dims=y_true_rank)
        loss *= class_weight

    if reshape_needed:
        loss = tf.reshape(loss, y_pred_shape[:-1])

    return loss


@tf.keras.utils.register_keras_serializable()
class SparseCategoricalFocalLoss(tf.keras.losses.Loss):
    """Focal loss function for multiclass classification with integer labels.

    This loss function generalizes multiclass softmax cross-entropy by
    introducing a hyperparameter $\gamma` (gamma), called the
    *focusing parameter*, that allows hard-to-classify examples to be penalized
    more heavily relative to easy-to-classify examples.

    This class is a wrapper around `focal_loss.sparse_categorical_focal_loss`.
    See the documentation there for details about this loss function.

    Examples:
        An instance of this class is a callable that takes a rank-one tensor of
        integer class labels `y_true` and a tensor of model predictions `y_pred` and
        returns a scalar tensor obtained by reducing the per-example focal loss (the
        default reduction is a batch-wise average).

        >>> from focal_loss import SparseCategoricalFocalLoss
        >>> loss_func = SparseCategoricalFocalLoss(gamma=2)
        >>> y_true = [0, 1, 2]
        >>> y_pred = [[0.8, 0.1, 0.1], [0.2, 0.7, 0.1], [0.2, 0.2, 0.6]]
        >>> loss_func(y_true, y_pred)
        <tf.Tensor: shape=(), dtype=float32, numpy=0.040919524>

        Use this class in the `tf.keras` API like any other multiclass
        classification loss function class that accepts integer labels found in
        `tf.keras.losses` (e.g., `tf.keras.losses.SparseCategoricalCrossentropy`:

        ```python
            # Typical usage
            model = tf.keras.Model(...)
            model.compile(
                optimizer=...,
                loss=SparseCategoricalFocalLoss(gamma=2),  # Used here like a tf.keras loss
                metrics=...,
            )
            history = model.fit(...)
        ```

    """

    def __init__(
        self,
        gamma: float,
        alpha: float,
        class_weight: Optional[Any] = None,
        from_logits: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.alpha = alpha
        self.gamma = gamma
        self.class_weight = class_weight
        self.from_logits = from_logits

    def get_config(self):
        """Returns the config of the layer.

        A layer config is a Python dictionary containing the configuration of a
        layer. The same layer can be re-instantiated later (without its trained
        weights) from this configuration.
        Returns

        Dict :  This layer's config.
        """
        config = super().get_config()
        config.update(
            gamma=self.gamma,
            alpha=self.alpha,
            class_weight=self.class_weight,
            from_logits=self.from_logits,
        )
        return config

    def call(self, y_true: tf.Tensor, y_pred: tf.Tensor):
        """Compute the per-example focal loss.

        This method simply calls `focal_loss.sparse_categorical_focal_loss` with the appropriate
        arguments.

        Args:
            y_true (tf.Tensor): Integer class labels.
            y_pred (tf.Tensor): Either probabilities or logits, depending on the `from_logits`
                parameter.

        Returns:
            (tf.Tensor): The per-example focal loss. Reduction to a scalar is handled by
                this layer's `focal_loss.SparseCateogiricalFocalLoss.__call__` method.
        """

        return sparse_categorical_focal_loss(
            y_true=y_true,
            y_pred=y_pred,
            class_weight=self.class_weight,
            alpha=self.alpha,
            gamma=self.gamma,
            from_logits=self.from_logits,
        )
