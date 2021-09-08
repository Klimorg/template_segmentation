import os
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import mlflow
import numpy as np
import pandas as pd
import tensorflow as tf
from loguru import logger
from omegaconf import DictConfig, OmegaConf

import hydra


# https://github.com/Erlemar/pytorch_tempest/blob/master/src/utils/technical_utils.py
def config_to_hydra_dict(cfg: DictConfig) -> Dict[str, str]:
    """
    Convert config into dict with lists of values.

    Key is full name of parameter this function
    is used to get key names which can be used in hydra.

    Args:
        cfg (DictConfig) : Hydra config file.

    Returns:
        converted dict
    """
    experiment_dict = {}
    for key, attributed_value in cfg.items():
        for sub_key, sub_value in attributed_value.items():
            experiment_dict[f"{key}.{sub_key}"] = sub_value

    return experiment_dict


# https://github.com/Erlemar/pytorch_tempest/blob/master/src/utils/technical_utils.py
def flatten_omegaconf(cfg: Any) -> Dict[Any, Any]:
    """Recursively flatten a nested Dict into a simple one.

    The difference between this function and `recurse` is that the dictionnary produced
    by this one doesn't have the hydra variables "$" anymore, and that the keys are
    alphabetically sorted.

    Used to store the parameters of the experiment in MLFlow.

    Args:
        cfg (Any): Hydra config files.

    Returns:
        The flattened dictionnary with all the parameters of the experiment.
    """
    cfg = OmegaConf.to_container(cfg)

    flattened_dict = {}

    def recurse(
        datas: Union[List[Any], Dict[str, str], str, None],
        parent_key="",
        sep: str = "_",
    ):
        """Recursively flatten a nested Dict into a simple one.

        Only used in `flatten_omegaconf`.

        Args:
            datas (Union[List, Dict]): Parts of the nested dictionnary to flatten.
            parent_key (str, optional): Parent key in a nested dictionnary, if
                necessary. Defaults to "".
            sep (str): Separator used between keys. Defaults to "_".
        """
        if isinstance(datas, list):
            for idx, _ in enumerate(datas):
                recurse(
                    datas[idx], parent_key + sep + str(idx) if parent_key else str(idx)
                )
        elif isinstance(datas, dict):
            for key, attributed_value in datas.items():
                recurse(attributed_value, parent_key + sep + key if parent_key else key)
        else:
            flattened_dict[parent_key] = datas

    recurse(cfg)

    obj_txt = {
        key: attributed_value
        for key, attributed_value in flattened_dict.items()
        if isinstance(attributed_value, str) and not attributed_value.startswith("$")
    }
    obj_num = {
        key: attributed_num
        for key, attributed_num in flattened_dict.items()
        if isinstance(attributed_num, (int, float))  # type: ignore
    }

    obj_txt.update(obj_num)

    res = dict(sorted(obj_txt.items()))
    return {key: attributed_value for key, attributed_value in res.items()}


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


# https://github.com/GokuMohandas/applied-ml/blob/main/tagifai/utils.py
def get_sorted_runs(
    experiment_name: str, order_by: List[str], top_k: Optional[int] = 10
) -> pd.DataFrame:
    """Get top_k best runs for a given experiment_name according to given metrics.

    Usage:
    ```python
    runs = get_sorted_runs(experiment_name="best", order_by=["metrics.val_loss ASC"])
    ```

    Args:
        experiment_name (str): [description]
        order_by (List): [description]
        top_k (Optional[int], optional): [description]. Defaults to 10.

    Returns:
        A dataframe of top_k best runs sorted by given metrics.
    """
    experiment_id = mlflow.get_experiment_by_name(experiment_name).experiment_id

    return mlflow.search_runs(
        experiment_ids=experiment_id,
        order_by=order_by,
    )[:top_k]


def set_log_infos(cfg: DictConfig) -> Tuple[Dict[str, str], str]:
    """[summary].

    Args:
        cfg (DictConfig): [description]

    Returns:
        Tuple[Dict, str]: [description]
    """
    timestamp = cfg.log.timestamp
    ml_config = OmegaConf.to_yaml(cfg)

    logger.add(f"logs_train_{timestamp}.log")
    logger.info(f"Training started at {timestamp}")
    logger.info(f"{ml_config}")

    conf_dict = config_to_hydra_dict(cfg)
    repo_path = hydra.utils.get_original_cwd()

    return conf_dict, repo_path


def get_items_list(directory: str, extension: str) -> List[Path]:
    return sorted(
        Path(item).absolute()
        for item in Path(directory).glob(f"**/*{extension}")
        if item.is_file()
    )
