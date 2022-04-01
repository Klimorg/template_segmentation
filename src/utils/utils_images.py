from itertools import product
from pathlib import Path

from loguru import logger
from omegaconf import OmegaConf
from PIL import Image

from src.errors.img_errors import ImageMaskMismatchError, validate_images_masks
from src.utils.utils import get_items_list

segmentation_config = OmegaConf.load("configs/datasets/datasets.yaml")


def crop(image_path: Path, mask_path: Path, stride: int, overlap: int):
    """Given a image and a segmentation mask, generate tiles from them by cropping.

    Given an image and a segmentation mask, resize them and them decompose them in tiles
    of size $H=W=$`stride`.

    The first iteration of the loop creates non overlapping tiles, then we restart the cropping processus but this time
    at the coordinates (`overlap`, `overlap`) of the resized original images and masks, the second loop starts at
    (2*`overlap`, 2*`overlap`), etc. Then save all the tiles in `.jpg` format for the images and `.png` format for the masks.

    Args:
        image_path (Path): The path of the image to open to start the tilling processus.
        mask_path (Path): The path of the mask to open to start the tilling processus.
        stride (int): Height, width of the cropped image, mask.
        overlap (int): How much pixels you want to overlap between each iteration.
    """

    image = Image.open(image_path).resize((1024, 1024))
    mask = Image.open(mask_path).resize((1024, 1024), resample=Image.NEAREST)

    image_name = Path(image_path).stem
    mask_name = Path(mask_path).stem

    width, height = image.size

    chunk = height // overlap

    for mult in range(chunk - 1):
        grid = list(
            product(
                range(mult * overlap, height - height % stride, stride),
                range(mult * overlap, width - width % stride, stride),
            ),
        )
        for idy, idx in grid:
            box = (idx, idy, idx + stride, idy + stride)

            dir_out_image = Path(segmentation_config.raw_dataset.images)
            dir_out_mask = Path(segmentation_config.raw_dataset.masks)

            image_name_cropped = f"{image_name}_{idy}_{idx}{'.jpg'}"
            mask_name_cropped = f"{mask_name}_{idy}_{idx}{'.png'}"

            image_out = Path(dir_out_image) / Path(image_name_cropped)
            mask_out = Path(dir_out_mask) / Path(mask_name_cropped)

            image.crop(box).save(image_out)
            mask.crop(box).save(mask_out)

    logger.info(f"Done.")


def tile():
    """Apply tilling processus on a list of images, masks."""

    stride = segmentation_config.raw_dataset.crop_size
    overlap = 128

    logger.info("Searching for images and corresponding masks.")
    images_paths = get_items_list(
        directory=segmentation_config.raw_datas.images,
        extension=".jpg",
    )

    masks_paths = get_items_list(
        directory=segmentation_config.raw_datas.masks,
        extension=".png",
    )

    try:
        validate_images_masks(images=images_paths, masks=masks_paths)
    except ImageMaskMismatchError as err:
        logger.error(f"The number of images and labels aren't the same : {err}")
        raise

    logger.info("Looping through images and masks for cropping.")
    for image_path, mask_path in zip(images_paths, masks_paths):
        crop(image_path, mask_path, stride, overlap)


if __name__ == "__main__":
    tile()
