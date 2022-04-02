import pytest
from omegaconf import OmegaConf

from src.utils.make_masks import JsonFormat, SegmentationMasks


@pytest.fixture
def coco_masks() -> SegmentationMasks:
    """Returns a test class.

    Returns:
        SegmentationMasks: The class we test here. Defined in `src.utils.make_masks.py`
    """

    test_config = {
        "metadatas": {"height": 3036, "width": 4024, "data_format": "coco"},
        "class_dict": {"Background": 0, "Petri_box": 1, "Moisissure": 2, "Levure": 3},
        "raw_datas": {
            "labels": "datas/raw_datas/ML/labels/",  # address of the json file
            "masks": "datas/raw_datas/ML/masks/",  # where we store generated masks
        },
    }

    segmentation_config = OmegaConf.create(test_config)

    data_format = JsonFormat(segmentation_config.metadatas.data_format)

    return SegmentationMasks(
        segmentation_config=segmentation_config,
        data_format=data_format,
    )


@pytest.fixture
def vgg_masks() -> SegmentationMasks:
    """Returns a test class.

    Returns:
        SegmentationMasks: The class we test here. Defined in `src.utils.make_masks.py`
    """

    test_config = {
        "metadatas": {"height": 3036, "width": 4024, "data_format": "vgg"},
        "class_dict": {"Background": 0, "Petri_box": 1, "Moisissure": 2, "Levure": 3},
        "raw_datas": {
            "labels": "/tests/test_datas/",  # address of the json file
            "masks": "/tests/test_datas/masks/",  # where we store generated masks
        },
    }

    segmentation_config = OmegaConf.create(test_config)

    data_format = JsonFormat(segmentation_config.metadatas.data_format)

    return SegmentationMasks(
        segmentation_config=segmentation_config,
        data_format=data_format,
    )


def test_constructor(coco_masks, vgg_masks) -> None:

    assert isinstance(coco_masks, SegmentationMasks)
    assert isinstance(vgg_masks, SegmentationMasks)
