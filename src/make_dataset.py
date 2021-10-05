import csv
import random
from pathlib import Path
from typing import List, Optional, Tuple

import typer
import yaml
from loguru import logger
from omegaconf import OmegaConf
from sklearn.model_selection import train_test_split

from src.utils.utils import get_items_list, set_seed

reproducibility_params = OmegaConf.load("configs/params.yaml")
datasets = OmegaConf.load("configs/datasets/datasets.yaml")

# with open("configs/params.yaml") as reproducibility_params:
#     config = yaml.safe_load(reproducibility_params)["prepare"]

# with open("configs/datasets/datasets.yaml") as datasets:
#     address = yaml.safe_load(datasets)

# raw_dataset_images = address["raw_dataset"]["images"]
# raw_dataset_masks = address["raw_dataset"]["masks"]

# train_dataset_address = address["prepared_dataset"]["train"]
# val_dataset_address = address["prepared_dataset"]["val"]
# test_dataset_address = address["prepared_dataset"]["test"]

# random_seed = config["seed"]
# split = config["split"]

app = typer.Typer()


def save_as_csv(filenames: List[Path], labels: List[Path], destination: Path) -> None:
    """Save two lists of observations, labels as a csv files.

    Args:
        filenames (List[str]): List of images addresses.
        labels (List[str]): Liste of labels addresses.
        destination (Path): Location of the saved csv file.
    """
    logger.info(f"Saving dataset in {destination}.")

    header = ["filename", "mask"]
    with open(destination, "w", newline="") as saved_csv:
        writer = csv.writer(saved_csv, delimiter=",")
        writer.writerow(header)
        writer.writerows(zip(filenames, labels))


observations_list = List[Path]
labels_list = List[Path]
Datasets = Tuple[
    observations_list,
    labels_list,
    observations_list,
    labels_list,
    observations_list,
    labels_list,
]


def create_train_val_test_datasets(
    raw_images: List[Path],
    raw_labels: List[Path],
    test_size: Optional[float] = reproducibility_params.prepare.split,
) -> Datasets:
    """Creation of datasets.

    Create three image datasets (train, validation, and test) given `raw_images`
    and `raw_labels`.

    The first step is to gather `raw_images` and `raw_labels` in a single
    `dataset` entity, then shuffle it, this is to ensure that the dataset is
    already well shuffled before the before using the `scikit_learn` module
    `train_test_split` (for example `dataset` could be alphabetically sorted
    before the shuffling).

    Then `dataset` passes into `train_test_split` to first get the `images_train`
    and `labels_train` and an intermediate `images_val` and `labels_val`.

    The intermediate `images_val` and `labels_val` is then again split in half
    to get the actual `images_val`, `labels_val`, `images_test`, `labels_test`.

    Args:
        raw_images (List[Path]): Full list of the images used for the three
            datasets.
        raw_labels (List[str]): Full list of the labels used for the three
            datasets.
        test_size (Optional[float], optional): Ratio used in the first use of
            `train_test_split`. Defaults to split.

    Returns:
        The three datasets returned as lists of images, labels.
    """
    set_seed(reproducibility_params.prepare.seed)

    dataset = list(zip(raw_images, raw_labels))
    random.shuffle(dataset)
    shuffled_images, shuffled_labels = zip(*dataset)

    images_train, images_val, labels_train, labels_val = train_test_split(
        shuffled_images,
        shuffled_labels,
        test_size=test_size,
        random_state=reproducibility_params.prepare.seed,
    )
    images_val, images_test, labels_val, labels_test = train_test_split(
        images_val,
        labels_val,
        test_size=0.5,
        random_state=reproducibility_params.prepare.seed,
    )

    return (
        images_train,
        labels_train,
        images_val,
        labels_val,
        images_test,
        labels_test,
    )


@app.command()
def main() -> None:
    """Main function."""
    images_paths = get_items_list(
        directory=datasets.raw_datasets.images,
        extension=".jpg",
    )
    masks_paths = get_items_list(
        directory=datasets.raw_datasets.masks,
        extension=".png",
    )

    datasets_components = create_train_val_test_datasets(images_paths, masks_paths)

    save_as_csv(
        datasets_components[0],
        datasets_components[1],
        datasets.prepared_dataset.train,
    )
    save_as_csv(
        datasets_components[2],
        datasets_components[3],
        datasets.prepared_dataset.val,
    )
    save_as_csv(
        datasets_components[4],
        datasets_components[5],
        datasets.prepared_dataset.test,
    )


if __name__ == "__main__":
    app()
