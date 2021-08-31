import numpy as np
import pytest

from src.model.unet import get_segmentation_model


@pytest.fixture
def feature_map() -> np.ndarray:
    return np.random.random((1, 512, 512, 3))


def test_feedforward_step(feature_map) -> None:
    model = get_segmentation_model(
        filters=64,
        n_classes=2,
        img_shape=(512, 512, 3),
    )

    y_pred = model(feature_map)

    assert y_pred.shape == (1, 512, 512, 2)
