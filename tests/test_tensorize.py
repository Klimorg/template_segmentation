from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import tensorflow as tf

from src.pipelines.classic import Tensorize


@pytest.fixture
def tensor() -> Tensorize:
    """Returns a test class.

    Returns:
        Tensorize: The class we test here. Defined in `src.tensorize.py`
    """
    return Tensorize(n_classes=2, img_shape=(224, 224, 3), random_seed=42)


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
    """Test that the constructor is weel defined.

    You should only need the three following parameters to initiate this
    class :

    1. The number of classes in the dataset.
    2. The dimensions of the images.
    3. The random seed for reproducibility.
    """
    ts = Tensorize(n_classes=2, img_shape=(224, 224, 3), random_seed=42)

    assert isinstance(ts, Tensorize)


def test_load_images(tensor: Tensorize, df: pd.DataFrame) -> None:
    """Test of the function `load_images`.

    The function should take the column 'filename' of the dataframe en return
    it as a list.

    Also checks that we have the right number of elements in the list.

    Args:
        tensor (Tensorize): [description]
        df (pd.DataFrame): [description]
    """
    filenames = tensor.load_images(data_frame=df, column_name="filename")

    assert isinstance(filenames, list)

    assert len(filenames) == 20

    for idx in range(20):
        assert isinstance(filenames[idx], str)

        image_path = Path(filenames[idx])
        assert image_path.is_file()


def test_load_labels(tensor: Tensorize, df: pd.DataFrame) -> None:
    """Test load_labels function.

    Args:
        tensor (Tensorize): [description]
        df (pd.DataFrame): [description]
    """
    zeros = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    ones = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]

    labels_test = zeros + ones

    labels_list = tensor.load_labels(data_frame=df, column_name="label")

    assert len(labels_list) == 20

    for idx in range(20):
        assert labels_list[idx] == labels_test[idx]


def test_parse_image_and_label(tensor: Tensorize, df: pd.DataFrame) -> None:
    """[summary].

    Args:
        tensor (Tensorize): [description]
        df (pd.DataFrame): [description]
    """
    label1 = 0
    label2 = 1

    img1, oh_label1 = tensor.parse_image_and_label(df["filename"][0], label1)
    _, oh_label2 = tensor.parse_image_and_label(df["filename"][10], label2)

    assert isinstance(img1, tf.Tensor)
    assert isinstance(oh_label1, tf.Tensor)

    assert img1.numpy().shape == (224, 224, 3)
    oh_label10 = oh_label1.numpy()[0]
    oh_label11 = oh_label1.numpy()[1]

    oh_label20 = oh_label2.numpy()[0]
    oh_label21 = oh_label2.numpy()[1]
    assert (oh_label10, oh_label11) == (1, 0)
    assert (oh_label20, oh_label21) == (0, 1)


def test_train_prepocess(tensor: Tensorize) -> None:
    """[summary].

    Args:
        tensor (Tensorize): [description]
        df (pd.DataFrame): [description]
    """
    oh_label = [0, 1]
    img_rdn = np.random.rand(224, 224, 3)
    img, _ = tensor.train_preprocess(img_rdn, oh_label)

    assert img.numpy().shape == (224, 224, 3)


def test_create_dataset_without_augment(tensor):
    """[summary].

    Args:
        tensor ([type]): [description]
        df ([type]): [description]
    """
    ds = tensor.create_dataset(
        "tests/test_datas/test_datas.csv",
        batch=5,
        repet=1,
        prefetch=1,
        augment=False,
    )

    for imgs, _ in ds.take(1):
        assert imgs.numpy().shape == (5, 224, 224, 3)


def test_create_dataset_with_augment(tensor):
    """[summary].

    Args:
        tensor ([type]): [description]
        df ([type]): [description]
    """
    ds = tensor.create_dataset(
        "tests/test_datas/test_datas.csv",
        batch=5,
        repet=1,
        prefetch=1,
        augment=True,
    )

    for imgs, _ in ds.take(1):
        assert imgs.numpy().shape == (5, 224, 224, 3)
