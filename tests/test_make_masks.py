import pytest

from src.utils.make_masks import SegmentationMasks


@pytest.fixture
def coco_masks() -> SegmentationMasks:
    """Returns a test class.

    Returns:
        SegmentationMasks: The class we test here. Defined in `src.utils.make_masks.py`
    """
    return SegmentationMasks()
