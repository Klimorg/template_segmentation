import os

import pytest
from hydra.experimental import compose, initialize
from omegaconf import DictConfig

config_files = [
    filename.split(".")[0] for filename in os.listdir("configs") if "yaml" in filename
]


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
