from pathlib import Path

import pandas as pd
import pytest

from src.utils.make_datasets import save_as_csv


@pytest.fixture
def images_paths():
    return [
        "tests/test_datas/44e7dfd7_384_384.jpg",
        "tests/test_datas/cs1_2907_640_128.jpg",
    ]


@pytest.fixture
def masks_paths():
    return [
        "tests/test_datas/44e7dfd7_mask_384_384.png",
        "tests/test_datas/cs1_2907_mask_640_128.png",
    ]


def test_save_as_csv(tmp_path, images_paths, masks_paths):

    save_as_csv(images_paths, masks_paths, Path(f"{tmp_path}/test.csv"))

    with open(f"{tmp_path}/test.csv", "r") as outfile:
        df = pd.read_csv(outfile)
        assert isinstance(df, pd.DataFrame)
