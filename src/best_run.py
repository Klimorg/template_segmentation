from pathlib import Path

import mlflow
import tensorflow as tf
import yaml
from loguru import logger
from tensorflow.keras.models import load_model

from utils import get_sorted_runs

with open("configs/params.yaml") as reproducibility_params:
    mlflow_config = yaml.safe_load(reproducibility_params)["mlflow"]

experiment_name = mlflow_config["experiment_name"]


def load_model_artifact() -> tf.keras.Model:
    """Load artifacts for a particular `run_id`.
    Args:
        run_id (str): ID of the run to load model artifacts from.
    Returns:
        Artifacts needed for inference.
    """

    all_runs = get_sorted_runs(
        experiment_name=experiment_name,
        order_by=["metrics.val_loss ASC"],
    )

    print(
        all_runs[
            [
                "run_id",
                "tags.mlflow.runName",
                "metrics.val_categorical_accuracy",
                "metrics.val_loss",
            ]
        ],
    )

    best_run = all_runs.iloc[0]["run_id"]

    logger.info(f"Best run id is : {best_run}")

    # Load model
    run = mlflow.get_run(run_id=best_run)
    homedir = Path(run.info.artifact_uri).parent.parent.parent.parent

    root = Path(run.info.artifact_uri).relative_to(homedir)
    model_url = Path(root) / Path("model/data") / "model.h5"

    model = load_model(model_url)
    logger.info(f"Model loaded from {run.info}")

    return model


if __name__ == "__main__":
    load_model_artifact()
