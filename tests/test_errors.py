from pathlib import Path

import pytest

from src.errors.img_errors import (
    EmptyImageDatasetError,
    EmptyMaskDatasetError,
    ImageMaskMismatchError,
    validate_images_masks,
    validate_non_empty_img_list,
    validate_non_empty_mask_list,
)
from src.errors.labelization_errors import (
    EmptyLabelizationFilesError,
    LabelizationError,
    PolygonError,
    validate_non_empty_vgg_files,
    validate_polygons,
)


@pytest.fixture
def img_paths_list():

    return [
        Path("img1"),
        Path("img2"),
        Path("img3"),
    ]


@pytest.fixture
def msk_paths_list():

    return [
        Path("msk1"),
        Path("msk2"),
        Path("msk3"),
        Path("msk4"),
    ]


@pytest.fixture
def empty_list():

    return []


@pytest.fixture
def X_coordinates():

    return [[0, 4, 6], [12, 3, 9]]


@pytest.fixture
def Y_coordinates_errors():

    return [[0, 1, 2]]


@pytest.fixture
def Y_coordinates():

    return [[0, 1, 2], [10, 12, 15]]


@pytest.fixture
def labels():

    return [1, 2]


@pytest.fixture
def labels_errors():

    return [1]


def test_validate_images_masks(img_paths_list, msk_paths_list):
    with pytest.raises(ImageMaskMismatchError):
        validate_images_masks(img_paths_list, msk_paths_list)


def test_validate_non_empty_img_list(empty_list):
    with pytest.raises(EmptyImageDatasetError):
        validate_non_empty_img_list(empty_list)


def test_validate_non_empty_mask_list(empty_list):
    with pytest.raises(EmptyMaskDatasetError):
        validate_non_empty_mask_list(empty_list)


def test_validate_polygons_coordinates(X_coordinates, Y_coordinates_errors, labels):
    with pytest.raises(PolygonError):
        validate_polygons(
            X_coordinates=X_coordinates,
            Y_coordinates=Y_coordinates_errors,
            labels=labels,
        )


def test_validate_polygons_labels(X_coordinates, Y_coordinates, labels_errors):
    with pytest.raises(LabelizationError):
        validate_polygons(
            X_coordinates=X_coordinates,
            Y_coordinates=Y_coordinates,
            labels=labels_errors,
        )


def test_validate_non_empty_vgg_files(empty_list):
    with pytest.raises(EmptyLabelizationFilesError):
        validate_non_empty_vgg_files(empty_list)
