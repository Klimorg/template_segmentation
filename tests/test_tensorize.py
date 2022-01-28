from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import tensorflow as tf

from src.pipelines.classic import BaseDataset


@pytest.fixture
def tensor() -> BaseDataset:
    """Returns a test class.

    Returns:
        Tensorize: The class we test here. Defined in `src.tensorize.py`
    """
    return BaseDataset(n_classes=4, img_shape=(256, 256, 3), random_seed=42)


@pytest.fixture
def df() -> pd.DataFrame:
    """Returns a test dataframe.

    Returns:
        pd.DataFrame: Test dataframe with manually crafted rows to check
        behavior during testing. Has 20 rows : 10 with 'Negative' elements
        followed by 10 with 'Positive' elements.
    """
    return pd.read_csv("tests/test_datas/test_datas.csv")


def test_constructor() -> None:
    """Test that the constructor is well defined.

    You should only need the three following parameters to initiate this
    class :

    1. The number of classes in the dataset.
    2. The dimensions of the images.
    3. The random seed for reproducibility.
    """
    ts = BaseDataset(n_classes=4, img_shape=(256, 256, 3), random_seed=42)

    assert isinstance(ts, BaseDataset)


def test_load_images_mask(tensor: BaseDataset, df: pd.DataFrame) -> None:
    """Test of the function `load_images`.

    The function should take the column 'filename' of the dataframe en return
    it as a list.

    Also checks that we have the right number of elements in the list.

    Args:
        tensor (BaseDataset): [description]
        df (pd.DataFrame): [description]
    """
    filenames = tensor.load_images(data_frame=df, column_name="filename")
    masks = tensor.load_images(data_frame=df, column_name="mask")

    assert isinstance(filenames, list)
    assert isinstance(masks, list)

    assert len(filenames) == 2
    assert len(masks) == 2

    for idx in range(2):
        assert isinstance(filenames[idx], str)
        assert isinstance(masks[idx], str)

        image_path = Path(filenames[idx])
        assert image_path.is_file()
        mask_path = Path(masks[idx])
        assert mask_path.is_file()


def test_parse_image_and_mask(tensor: BaseDataset, df: pd.DataFrame) -> None:
    """[summary].

    Args:
        tensor (BaseDataset): [description]
        df (pd.DataFrame): [description]
    """

    img1, mask1 = tensor.parse_image_and_mask(df["filename"][0], df["mask"][0])
    img2, mask2 = tensor.parse_image_and_mask(df["filename"][1], df["mask"][1])

    assert isinstance(img1, tf.Tensor)
    assert isinstance(img2, tf.Tensor)

    assert isinstance(mask1, tf.Tensor)
    assert isinstance(mask2, tf.Tensor)

    assert img1.numpy().shape == (256, 256, 3)
    assert mask1.numpy().shape == (256, 256, 1)


def test_train_prepocess(tensor: BaseDataset) -> None:
    """[summary].

    Args:
        tensor (BaseDataset): [description]
        df (pd.DataFrame): [description]
    """
    img_rdn = np.random.rand(256, 256, 3)
    mask_rdn = np.random.rand(256, 256, 1)
    img, mask = tensor.train_preprocess(img_rdn, mask_rdn)

    assert img.numpy().shape == (256, 256, 3)
    assert mask.numpy().shape == (256, 256, 1)


def test_create_dataset_without_augment(tensor):
    """[summary].

    Args:
        tensor ([type]): [description]
        df ([type]): [description]
    """
    ds = tensor.create_train_dataset(
        "tests/test_datas/test_datas.csv",
        batch=1,
        repet=1,
        prefetch=1,
        augment=False,
    )

    for img, mask in ds.take(1):
        assert img.numpy().shape == (1, 256, 256, 3)
        assert mask.numpy().shape == (1, 256, 256, 1)


def test_create_dataset_with_augment(tensor):
    """[summary].

    Args:
        tensor ([type]): [description]
        df ([type]): [description]
    """
    ds = tensor.create_train_dataset(
        "tests/test_datas/test_datas.csv",
        batch=1,
        repet=1,
        prefetch=1,
        augment=True,
    )

    for img, mask in ds.take(1):
        assert img.numpy().shape == (1, 256, 256, 3)
        assert mask.numpy().shape == (1, 256, 256, 1)
