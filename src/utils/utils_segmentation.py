import json
from itertools import product
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import yaml
from loguru import logger
from PIL import Image, ImageDraw

from utils import get_items_list

JsonDict = Dict[str, Any]
PolygonVertices = List[float]


class Segmentation(object):
    def __init__(self, raw_masks_path):

        self.raw_masks_json = raw_masks_path

        # Get the address of the dataset info
        with open("configs/datasets/datasets.yaml") as datasets:
            config_dataset = yaml.safe_load(datasets)

        metadatas = config_dataset["metadatas"]

        # logger.info("Loading the address of the vgg json segmentation datas file.")
        # raw_masks_json = metadatas["masks"]

        logger.info(
            "Loading the vgg json file and gathering the keys, i.e. the image names."
        )
        # Load the vgg json file and get the keys, i.e. the image names.
        with open(self.raw_masks_json) as vgg_json:
            self.all_coordinates_and_labels = json.load(vgg_json)
            self.image_names = sorted(self.all_coordinates_and_labels.keys())

        logger.info(f"Found {len(self.image_names)} images for segmentation task.")

        logger.info("Getting height and width metadatas")
        self.height = metadatas["height"]
        self.width = metadatas["width"]

        logger.info("Loading the class dictionnary for the segmentation.")
        self.class_dict = config_dataset["class_dict"]

        logger.info("Getting directory paths of raw images and masks.")
        raw_datas = config_dataset["raw_datas"]
        raw_dataset = config_dataset["raw_dataset"]

        self.raw_datas_images = raw_datas["images"]
        self.raw_datas_masks = raw_datas["masks"]

        self.raw_dataset_images = raw_dataset["images"]
        self.raw_dataset_masks = raw_dataset["masks"]
        self.crop_size = raw_dataset["crop_size"]

    def get_data(
        self,
        image_name: str,
        coordinates_and_labels: JsonDict,
    ) -> Tuple[List[PolygonVertices], List[PolygonVertices], List[str]]:
        """Return the coordinates of polygons vertices and corresponding labels.

        Given an `image_name`, parse the `segmentation_datas` contained in the VGG json file `all_coordinates_and_labels`
        and return them in 3 lists :
        - Two lists of vertices.
        - One list of labels.

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
            coordinates_and_labels (JsonDict, optional): The loaded VGG json file to parse.

        Returns:
            Tuple[List[PolygonVertices], List[PolygonVertices], List[str]]: The lists of polygon vertices and labels of the given image.
        """

        X_coordinates = []
        Y_coordinates = []
        labels = []  # pre-allocate lists to fill in a for loop

        segmentation_datas = coordinates_and_labels[image_name]

        for polygon in segmentation_datas["regions"]:
            # cycle through each polygon of the given image
            # get the x and y points from the dictionary
            X_coordinates.append(
                segmentation_datas["regions"][polygon]["shape_attributes"][
                    "all_points_x"
                ]
            )
            Y_coordinates.append(
                segmentation_datas["regions"][polygon]["shape_attributes"][
                    "all_points_y"
                ]
            )
            # get the labels corresponding to the polygons
            labels.append(
                segmentation_datas["regions"][polygon]["region_attributes"]["label"]
            )

        assert len(X_coordinates) == len(Y_coordinates)
        assert len(Y_coordinates) == len(labels)
        logger.info(f"Found {len(labels)} segmentation masks for {image_name}")

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

        given a triplet (Y_coordinates,X_coordinates,labels) for a given image, transforms it into a (P,H,W) numpy array
        where P is the number of segmentation polygons and H,W the height and width.

        Args:
            X_coordinates (PolygonVertices): x-axis coordinates of the polygon vertices.
            Y_coordinates (PolygonVertices): y-axis coordinates of the polygon vertices.
            labels (List[str]): Labels corresponding to the polygons.
            img_height (int): Height of the image.
            img_width (int): Width of the image.
            class_dict (Dict[str, int]): Man-made dictionnary to convert the labels of the VGG json file (str) into integers.

        Returns:
            List[np.ndarray]: (P,H,W) numpy array containing the polygons.
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
            m = np.flipud(np.rot90(np.array(img)))
            try:
                mask[m == 1] = classes[label]
            except:
                mask[m.T == 1] = classes[label]

            masks.append(mask)

        return masks

    def get_segmentation_mask(
        self,
        save: bool = True,
    ):
        masks = []
        image_names_list = self.image_names
        coordinates_and_labels = self.all_coordinates_and_labels

        for image_name in image_names_list:

            X_coordinates, Y_coordinates, labels = self.get_data(
                image_name, coordinates_and_labels
            )

            polygon_masks = self.get_polygon_masks(
                X_coordinates,
                Y_coordinates,
                labels,
                self.height,
                self.width,
                self.class_dict,
            )

            mask = np.array(polygon_masks)

            logger.info(f"Made polygon masks of shape {mask.shape}.")

            segmentation_mask = np.max(mask, axis=0)

            if save:
                image_name_stem = Path(f"{image_name}").stem
                address = Path(self.raw_datas_masks) / Path(
                    f"{image_name_stem}_mask.png"
                )
                segmentation_mask = Image.fromarray(segmentation_mask).convert("L")
                segmentation_mask.save(address)

            masks.append(segmentation_mask)

        assert len(self.image_names) == len(masks)

    # 1. liste les images et les masques
    # 2. boucle sur zip(images, masques)
    # 1. Resize (1024,1024)
    # 2. on crop en 224 par 224
    # 3. on sauvegarde
    # 3. repeat

    def crop(self, image_path: Path, mask_path: Path, stride: int, overlap: int):

        image = Image.open(image_path).resize((1024, 1024))
        mask = Image.open(mask_path).resize((1024, 1024), resample=Image.NEAREST)

        image_name = Path(image_path).stem
        mask_name = Path(mask_path).stem

        width, height = image.size

        overlap_y = height // overlap
        # overlap_x = width // overlap

        for mult in range(overlap_y):

            grid = list(
                product(
                    range(mult * overlap, height - height % stride, stride),
                    range(mult * overlap, width - width % stride, stride),
                )
            )
            for idy, idx in grid:
                box = (idx, idy, idx + stride, idy + stride)

                dir_out_image = Path(self.raw_dataset_images)
                dir_out_mask = Path(self.raw_dataset_masks)

                image_name_cropped = f"{image_name}_{idy}_{idx}{'.jpg'}"
                mask_name_cropped = f"{mask_name}_{idy}_{idx}{'.png'}"

                image_out = Path(dir_out_image) / Path(image_name_cropped)
                mask_out = Path(dir_out_mask) / Path(mask_name_cropped)

                image.crop(box).save(image_out)
                mask.crop(box).save(mask_out)
            logger.info(f"Done for {image_name}, {mask_name} with start at {mult}.")

    def tile(
        self,
    ):
        stride = self.crop_size
        overlap = 128

        logger.info("Searching for images and corresponding masks.")
        images_paths = get_items_list(directory=self.raw_datas_images, extension=".jpg")
        masks_paths = get_items_list(directory=self.raw_datas_masks, extension=".png")

        assert len(images_paths) == len(masks_paths)

        logger.info("Looping through images and masks for cropping.")
        for image_path, mask_path in zip(images_paths, masks_paths):
            self.crop(image_path, mask_path, stride, overlap)


if __name__ == "__main__":

    seg = Segmentation()

    seg.get_segmentation_mask()
    seg.tile()
