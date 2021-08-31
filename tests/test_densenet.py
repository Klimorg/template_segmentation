import numpy as np
import pytest

from src.model.tiramisu import get_segmentation_model


@pytest.fixture
def feature_map() -> np.ndarray:
    return np.random.random((1, 512, 512, 3))


def test_feedforward_step_fc56(feature_map) -> None:
    model = get_segmentation_model(
        filters=12,
        nb_blocks=5,
        nb_layers=[4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4],
        nb_filters=48,
        n_classes=2,
        img_shape=(512, 512, 3),
    )

    y_pred = model(feature_map)

    assert y_pred.shape == (1, 512, 512, 2)


# def test_feedforward_step_fc67(feature_map) -> None:
#     model = get_segmentation_model(
#         filters=16,
#         nb_blocks=5,
#         nb_layers=[5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5],
#         nb_filters=48,
#         n_classes=2,
#         img_shape=(512, 512, 3),
#     )

#     y_pred = model(feature_map)

#     assert y_pred.shape == (1, 512, 512, 2)


# def test_feedforward_step_fc103(feature_map) -> None:
#     model = get_segmentation_model(
#         filters=16,
#         nb_blocks=5,
#         nb_layers=[4, 5, 7, 10, 12, 15, 12, 10, 7, 5, 4],
#         nb_filters=48,
#         n_classes=2,
#         img_shape=(512, 512, 3),
#     )

#     y_pred = model(feature_map)

#     assert y_pred.shape == (1, 512, 512, 2)
