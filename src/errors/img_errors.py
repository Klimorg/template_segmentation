from pathlib import Path
from typing import List


class ImageError(ValueError):
    """[summary]

    Args:
        ValueError ([type]): [description]
    """


class ImageDimensionError(ImageError):
    """[summary]

    Args:
        ImageError ([type]): [description]
    """


class ImageChannelError(ImageError):
    """[summary]

    Args:
        ImageError ([type]): [description]
    """


class ExtensionError(ImageError):
    """[summary]

    Args:
        ImageError ([type]): [description]
    """


class ImageMaskMismatchError(ImageError):
    """[summary]

    Args:
        ImageError ([type]): [description]
    """


class EmptyImageDatasetError(ImageError):
    """[summary]

    Args:
        ImageError ([type]): [description]
    """


class EmptyMaskDatasetError(ImageError):
    """[summary]

    Args:
        ImageError ([type]): [description]
    """


def validate_images_masks(images: List[Path], masks: List[Path]) -> None:
    """[summary]

    Args:
        images ([type]): [description]
        masks ([type]): [description]

    Raises:
        ImageMaskMismatchError: [description]
    """

    if len(images) - len(masks) != 0:
        raise ImageMaskMismatchError(len(images), len(masks))


def validate_non_empty_img_list(item_list):
    """[summary]

    Args:
        item_list ([type]): [description]

    Raises:
        EmptyImageDatasetError: [description]
    """

    if item_list:
        pass
    else:
        raise EmptyImageDatasetError(item_list)


def validate_non_empty_mask_list(item_list):
    """[summary]

    Args:
        item_list ([type]): [description]

    Raises:
        EmptyMaskDatasetError: [description]
    """

    if item_list:
        pass
    else:
        raise EmptyMaskDatasetError(item_list)
