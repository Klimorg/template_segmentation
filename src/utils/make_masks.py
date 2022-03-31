from pathlib import Path
from typing import Dict, List, Tuple, Union

import numpy as np
from loguru import logger
from omegaconf import DictConfig, ListConfig, OmegaConf
from PIL import Image, ImageDraw

from src.errors.img_errors import ImageMaskMismatchError, validate_images_masks
from src.errors.labelization_errors import (
    EmptyLabelizationFilesError,
    LabelizationError,
    PolygonError,
    validate_non_empty_vgg_files,
    validate_polygons,
)
from src.utils.convert import coco2vgg
from src.utils.pydantic_data_models import JsonFormat, VggAnnotations
from src.utils.utils import get_items_list

PolygonVertices = List[float]


class SegmentationMasks(object):
    """
    Description of Segmentation.

    Class used to generate semantic segmentation masks, and apply various operations on them.

    Attributes:
        segmentation_config (Dict[str, Any]): The loaded yaml file containing the configuration parameters for the datasets.

    Inheritance:
        object: The base class of the class hierarchy, used only to enforce WPS306.
        See https://wemake-python-stylegui.de/en/latest/pages/usage/violations/consistency.html#consistency.

    **Main function** : `generate_masks()`

    """

    def __init__(
        self,
        segmentation_config: Union[DictConfig, ListConfig],
        data_format: JsonFormat,
    ):
        """Initialization of the class."""

        self.segmentation_config = segmentation_config
        self.data_format = data_format

    def get_data(
        self,
        image: str,
        dataset,
    ) -> Tuple[List[PolygonVertices], List[PolygonVertices], List[str]]:
        """Return the coordinates of polygons vertices and corresponding labels.

        Given an `image_name`, parse the `segmentation_datas` contained in the VGG json file `all_coordinates_and_labels`
        and return them in 3 lists :

        * Two lists of vertices.
        * One list of labels.

        The first two lists correspond to the lists of vertices of the polygons defining the segmentation regions. For each
        of them, an element is a list containing the coordinates ($x$ or $y$) of the vertices of a polygon. I.e. if `X_coordinates`
        and `Y_coordinates` look like the following.

        ```python
        X_coordinates = [[0,4,6], [12,3,9]]
        Y_coordinates = [[0,1,2], [10,12,15]]
        ```

        That means we have two poygons inthe given image, each of them having three vertices (ie two triangles), the
        vertices of the first triangles being : $(0,0)$, $(4,1)$, and $(6,2)$.

        Args:
            image_name (str): The name of the image for which we parse the VGG json file to collect polygons coordinates an
                labels.
            coordinates_and_labels (JsonDict): The loaded VGG json file to parse.

        Returns:
            Tuple[List[PolygonVertices], List[PolygonVertices], List[str]]: The lists of polygon vertices and labels of the given image.
        """

        X_coordinates = []
        Y_coordinates = []
        labels = []  # pre-allocate lists to fill in a for loop

        polygons = list(dataset[image].regions)

        for polygon in polygons:
            # cycle through each polygon of the given image
            # get the x and y points from the dictionary
            X_coordinates.append(
                dataset[image].regions[polygon].shape_attributes.all_points_x,
            )
            Y_coordinates.append(
                dataset[image].regions[polygon].shape_attributes.all_points_y,
            )
            # get the labels corresponding to the polygons
            labels.append(dataset[image].regions[polygon].region_attributes.label)

        try:
            validate_polygons(
                X_coordinates=X_coordinates,
                Y_coordinates=Y_coordinates,
                labels=labels,
            )
        except PolygonError as err1:
            logger.error(
                f"There is something wrong in the json file, the numbers of X and Y coordinates don't match: {err1}",
            )
            raise
        except LabelizationError as err2:
            logger.error(
                f"There should be one label for each polygon. This is not the case here: {err2}",
            )
            raise

        return (
            Y_coordinates,
            X_coordinates,
            labels,
        )  # image coordinates are flipped relative to json coordinates

    def get_polygon_masks(
        self,
        X_coordinates: List[PolygonVertices],
        Y_coordinates: List[PolygonVertices],
        labels: List[str],
        img_height: int,
        img_width: int,
        classes: Dict[str, int],
    ) -> List[np.ndarray]:
        """Transforms triplets (Y_coordinates,X_coordinates,labels) into numpy arrays.

        given a triplet (Y_coordinates,X_coordinates,labels) for a given image, transforms it into a $(P,H,W)$ numpy array
        where $P$ is the number of segmentation polygons and $H,W$ the height and width.

        Args:
            X_coordinates (PolygonVertices): x-axis coordinates of the polygon vertices.
            Y_coordinates (PolygonVertices): y-axis coordinates of the polygon vertices.
            labels (List[str]): Labels corresponding to the polygons.
            img_height (int): Height of the image.
            img_width (int): Width of the image.
            class_dict (Dict[str, int]): Man-made dictionnary to convert the labels of the VGG json file (str) into integers.

        Returns:
            List[np.ndarray]: $(P,H,W)$ numpy array containing the polygons.
        """

        masks = []

        for y, x, label in zip(X_coordinates, Y_coordinates, labels):
            mask = np.zeros((img_height, img_width))
            # the ImageDraw.Draw().polygon function we will use to create the mask
            # requires the x's and y's are interweaved, which is what the following
            # one-liner does
            polygon = np.vstack((x, y)).reshape((-1,), order="F").tolist()

            # create a mask image of the right size and infill according to the polygon
            if img_height > img_width:
                x, y = y, x
                img = Image.new("L", (img_height, img_width), 0)

            elif img_width > img_height:
                x, y = y, x
                img = Image.new("L", (img_width, img_height), 0)

            else:
                img = Image.new("L", (img_height, img_width), 0)
            ImageDraw.Draw(img).polygon(polygon, outline=0, fill=1)
            # turn into a numpy array
            # m = np.flipud(np.rot90(np.array(img)))
            m = np.array(img)

            try:
                mask[m == 1] = classes[label]
            except Exception:
                mask[m.T == 1] = classes[label]

            masks.append(mask)

        return masks

    def get_masks_from_json(self, json_file) -> None:
        """Given a json file containing segmentation masks information in VGG format, generate the masks.

        Open the json file `json_file` to retrieve the coordinates and labels of the segmentation masks and the
        associated images. Then loop over each images and applys `get_data` and `get_polygon_masks` to obtain for
        each image a $(P,H,W)$ numpy array where $P$ is the number of segmentation polygons and $H,W$ the height and width.

        Apply `np.max` on each $(P,H,W)$ array to get a unique segmentation mask of size $(H,W)$, then save it in `.png`
        format.

        Args:
            json_file ([type]): Json file containing segmentation masks information in VGG format.
        """
        masks = []

        if self.data_format == JsonFormat.vgg:
            vgg_dataset = VggAnnotations.parse_file(json_file)
        else:
            vgg_dataset = coco2vgg(coco_json_model_path=json_file)

        images = sorted(vgg_dataset)

        for image in images:

            X_coordinates, Y_coordinates, labels = self.get_data(
                image=image,
                dataset=vgg_dataset,
            )

            polygon_masks = self.get_polygon_masks(
                X_coordinates,
                Y_coordinates,
                labels,
                self.segmentation_config.metadatas.height,
                self.segmentation_config.metadatas.width,
                self.segmentation_config.class_dict,
            )

            mask = np.array(polygon_masks)

            logger.info(f"Made polygon masks of shape {mask.shape}.")

            segmentation_mask = np.max(mask, axis=0)

            image_name_stem = Path(f"{image}").stem
            address = Path(self.segmentation_config.raw_datas.masks) / Path(
                f"{image_name_stem}_mask.png",
            )
            segmentation_mask = Image.fromarray(segmentation_mask).convert("L")
            segmentation_mask.save(address)

            masks.append(segmentation_mask)

        try:
            validate_images_masks(images=images, masks=masks)
        except ImageMaskMismatchError as err:
            logger.warning(f"The number of images and labels aren't the same : {err}")
            raise

    def generate_masks(self):
        """Main function, list all json files in VGG format containig segmentation informations and generates masks.

        1. List json files.
        2. For each of them, generate segmentation masks.
        1. Then apply tilling processus on all the generated masks and images.
        """

        json_files = get_items_list(
            directory=self.segmentation_config.raw_datas.labels,
            extension=".json",
        )

        try:
            validate_non_empty_vgg_files(item_list=json_files)
        except EmptyLabelizationFilesError as img_err:
            logger.error(
                f"There are no files found, are you sure of your extension ? : {img_err}",
            )
            raise

        logger.info(f"Found {len(json_files)} json files.")

        for json_file in json_files:
            self.get_masks_from_json(json_file=json_file)


if __name__ == "__main__":

    segmentation_config = OmegaConf.load("configs/datasets/datasets.yaml")

    try:
        data_format = JsonFormat(segmentation_config.metadatas.data_format)
    except ValueError:
        logger.error(
            "Wrong JsonFormat, check the metadatas.data_format in datasets.yaml.",
        )
        raise

    seg = SegmentationMasks(
        segmentation_config=segmentation_config,
        data_format=data_format,
    )
    seg.generate_masks()

    # test_config = {
    #     "metadatas": {"height": 3036, "width": 4024, "data_format": "vgg"},
    #     "class_dict": {"Background": 0, "Petri_box": 1, "Moisissure": 2, "Levure": 3},
    #     "raw_datas": {
    #         "labels": "datas/raw_datas/ML/labels/",  # address of the json file
    #         "masks": "datas/raw_datas/ML/masks/",  # where we store generated masks
    #     },
    # }

    # segmentation_config = OmegaConf.create(test_config)
    # print(f"{segmentation_config}")

    # data_format = JsonFormat(segmentation_config.metadatas.data_format)
    # print(f"{data_format}")
