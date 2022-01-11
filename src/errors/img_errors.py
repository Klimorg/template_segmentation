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


def validate_images_masks(images, masks):

    if len(images) - len(masks) != 0:
        raise ImageMaskMismatchError(len(images), len(masks))


def validate_non_empty_img_list(item_list):

    if item_list:
        pass
    else:
        raise EmptyImageDatasetError(item_list)


def validate_non_empty_mask_list(item_list):

    if item_list:
        pass
    else:
        raise EmptyMaskDatasetError(item_list)
