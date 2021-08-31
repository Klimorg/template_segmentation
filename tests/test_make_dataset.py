from pathlib import Path

import pandas as pd
import pytest

from src.make_dataset import create_train_val_test_datasets


@pytest.fixture
def root_directory():
    """[summary].

    Returns:
        [type]: [description]
    """
    return Path("tests/test_datas")


@pytest.fixture
def df() -> pd.DataFrame:
    """Returns a test dataframe.

    Returns:
        pd.DataFrame: Test dataframe with manually crafted rows to check
        behavior during testing. Has 20 rows : 10 with 'Negative' elements
        followed by 10 with 'Positive' elements.
    """
    return pd.read_csv("tests/test_datas/test_datas.csv")


# def test_get_files_paths(root_directory) -> None:
#     """[summary].

#     Args:
#         root_directory ([type]): [description]
#     """
#     files_paths, subdirs = get_files_paths(root_directory)

#     for idx in range(20):
#         assert isinstance(files_paths[idx], Path)

#         image_path = Path(files_paths[idx])
#         assert image_path.is_file()

#     assert isinstance(subdirs, list)
#     assert len(subdirs) == 2


# def test_get_images_paths_and_labels(root_directory) -> None:
#     """[summary].

#     Args:
#         root_directory ([type]): [description]
#     """
#     files_paths, subdirs = get_files_paths(root_directory)

#     images, labels = get_images_paths_and_labels(files_paths, subdirs)

#     assert len(images) == 20
#     assert len(labels) == 20

#     for idx in range(10):
#         assert images[idx].parent.name == "Negative"
#         assert images[10 + idx].parent.name == "Positive"


def test_create_train_val_test_datasets(df) -> None:
    """[summary].

    Args:
        df ([type]): [description]
    """
    raw_images = df["filename"]
    raw_labels = df["label"]

    datasets_components = create_train_val_test_datasets(raw_images, raw_labels)

    assert len(datasets_components[0]) == len(datasets_components[1]) == 15
    assert 2 <= len(datasets_components[2]) <= 3
    assert 2 <= len(datasets_components[3]) <= 3
    assert 2 <= len(datasets_components[4]) <= 3
    assert 2 <= len(datasets_components[5]) <= 3
