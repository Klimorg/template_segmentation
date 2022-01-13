import os
import random
from pathlib import Path
from typing import List

import numpy as np
import tensorflow as tf
from loguru import logger
from omegaconf import DictConfig, OmegaConf

import hydra


def set_seed(random_seed: int) -> None:
    """(Try to) fix random behavior for reproducibility.

    Args:
        random_seed (int): The seed, the answer to life, the universe, and the rest.
    """
    os.environ["PYTHONHASHSEED"] = str(random_seed)
    random.seed(random_seed)
    np.random.seed(random_seed)
    tf.random.set_seed(random_seed)
    os.environ["TF_DETERMINISTIC_OPS"] = "1"
    # https://github.com/tensorflow/tensorflow/issues/39751
    # needed until a deterministic fix of tf.gather is implemented
    os.environ["TF_DISABLE_SEGMENT_REDUCTION_OP_DETERMINISM_EXCEPTIONS"] = "1"


def set_log_infos(cfg: DictConfig) -> str:
    """[summary].

    Args:
        cfg (DictConfig): [description]

    Returns:
        str: [description]
    """
    timestamp = cfg.log.timestamp
    ml_config = OmegaConf.to_container(cfg, resolve=True)

    logger.add(f"logs_train_{timestamp}.log")
    logger.info(f"Training started at {timestamp}")
    logger.info(f"Experiment configuration: {ml_config}")

    return hydra.utils.get_original_cwd()


def get_items_list(directory: str, extension: str) -> List[Path]:
    return sorted(
        Path(file).absolute()
        for file in Path(directory).glob(f"**/*{extension}")
        if file.is_file()
    )
