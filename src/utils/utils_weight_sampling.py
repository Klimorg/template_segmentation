from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import tensorflow as tf
import yaml
from loguru import logger
from omegaconf import OmegaConf
from PIL import Image

config = OmegaConf.load("configs/params.yaml")

from .utils import get_items_list

JsonDict = Dict[str, Any]
PolygonVertices = List[float]


class WeightBalancing(object):
    def __init__(self, repo_path: Optional[Path] = None, extension: str = ".png"):
        self.extension = extension
        self.repo_path = repo_path

        self.datasets = OmegaConf.load("configs/datasets/datasets.yaml")

        logger.info("Loading the class dictionnary for the segmentation.")
        self.class_dict = self.datasets.class_dict
        self.directory = self.datasets.raw_dataset.masks
        self.crop_size = self.datasets.raw_dataset.crop_size
        self.frequencies = self.get_median_frequency_balancing()

    def get_median_frequency_balancing(self):
        """[summary]

        alpha_c := median_freq / freq(c)
        freq(c) := number of pixels of class c divided by the total number of pixels in
        images where c is present
        median_freq := median({freq(c) : c in classes})

        Note:
            https://arxiv.org/pdf/1411.4734.pdf

            Section 6.3.2

        Args:
            directory ([type]): [description]
            labels (JsonDict, optional): [description]. Defaults to class_dict.
            extension (str, optional): [description]. Defaults to ".png".

        Returns:
            [type]: [description]
        """

        # get all masks
        masks_paths = get_items_list(self.directory, self.extension)
        logger.info(f"Found {len(masks_paths)} masks")

        frequencies = []

        for label, _ in self.class_dict.items():
            logger.info(f"Computing pixel frequency for class {label}")
            pixels_of_class = 0  # total number of pixels of class c
            pixels_present_in_mask = (
                0  # total number of pixels in images where c is present
            )
            masks_with_class = 0  # total number of masks where the class c is present

            for mask_path in masks_paths:
                mask = Image.open(mask_path)
                mask = np.asarray(mask).astype("uint8")

                number_of_pixels_in_class = len(mask[mask == self.class_dict[label]])

                pixels_of_class += number_of_pixels_in_class

                if number_of_pixels_in_class > 0:
                    pixels_present_in_mask += self.crop_size ** 2
                    masks_with_class += 1

            frequency = pixels_of_class / pixels_present_in_mask

            logger.info(f"Frequency of class {label} in the whole dataset {frequency}")
            logger.info(f"Found {masks_with_class} masks with the class {label}")
            frequencies.append(frequency)

        median_frequency = np.median(frequencies)

        balancing = [
            float(median_frequency / frequency_class) for frequency_class in frequencies
        ]

        logger.info(
            f"The weights for the median frequency balancing method are the following ones {balancing}."
        )

        OmegaConf.save(
            {"balancing": balancing},
            "configs/weight_sampling/weight.yaml",
            resolve=True,
        )

        return balancing

    def get_dataset_infos(self):

        masks_paths = get_items_list(self.directory, self.extension)
        logger.info(f"Found {len(masks_paths)} masks")

        subtotal = []

        for label, _ in self.class_dict.items():

            sub25 = 0
            from25to50 = 0
            from50to75 = 0
            sup75 = 0

            for mask_path in masks_paths:
                mask = Image.open(mask_path)
                mask = np.asarray(mask).astype("uint8")

                number_of_pixels_in_class = len(mask[mask == self.class_dict[label]])

                sub_frequency = number_of_pixels_in_class / self.crop_size ** 2

                if number_of_pixels_in_class > 0:

                    if sub_frequency < 0.25:
                        sub25 += 1
                    elif sub_frequency < 0.5:
                        from25to50 += 1
                    elif sub_frequency < 0.75:
                        from50to75 += 1
                    else:
                        sup75 += 1

            subtotal.append(
                {
                    "sub_25": sub25,
                    "from_25_to_50": from25to50,
                    "from_50_to_75": from50to75,
                    "sup_75": sup75,
                }
            )

        descriptif = pd.DataFrame(
            subtotal,
            columns=["sub_25", "from_25_to_50", "from_50_to_75", "sup_75"],
            index=["Background", "Petri_box", "Moisissure", "Levure"],
        )

        print(f"{descriptif}")


def add_sample_weights(image, label):
    # The weights for each class, with the constraint that:
    #     somme(class_weights) = 1.0
    class_weights = tf.constant(
        [0.6736764277552485, 0.9097746315154771, 1.1100914521609606, 2.3813523569399893]
    )
    class_weights = class_weights / tf.reduce_sum(class_weights)

    # Create an image of `sample_weights` by using the label at each pixel as an
    # index into the `class weights` .
    sample_weights = tf.gather(class_weights, indices=tf.cast(label, tf.int32))

    return image, label, sample_weights


if __name__ == "__main__":
    wb = WeightBalancing()
    wb.get_dataset_infos()
