import os
from pathlib import Path

import pytest
from hydra.experimental import compose, initialize
from omegaconf import DictConfig

from src.utils.make_datasets import get_items_list

config_files = [
    filename.split(".")[0] for filename in os.listdir("configs") if "yaml" in filename
]


@pytest.fixture
def root_directory():
    """[summary].

    Returns:
        [type]: [description]
    """
    return Path("tests/test_datas")


# https://github.com/Erlemar/pytorch_tempest/blob/master/tests/test_cfg.py
@pytest.mark.parametrize("config_name", config_files)
def test_cfg(config_name: str) -> None:
    """[summary].

    Args:
        config_name (str): [description]
    """
    with initialize(config_path="../configs"):
        cfg = compose(config_name=config_name)
        assert isinstance(cfg, DictConfig)
        # check it isn't an empty dict
        assert bool(cfg)


def test_get_files_paths(root_directory) -> None:
    """[summary].

    Args:
        root_directory ([type]): [description]
    """
    files_paths = get_items_list(directory=root_directory, extension=".jpg")

    for idx in range(len(files_paths)):
        assert isinstance(files_paths[idx], Path)

        image_path = Path(files_paths[idx])
        assert image_path.is_file()
