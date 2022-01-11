from pathlib import Path

import mlflow
import tensorflow as tf
from hydra.utils import instantiate
from loguru import logger
from mlflow import tensorflow as mltensorflow
from omegaconf import DictConfig

import hydra
from utils.utils import flatten_omegaconf, set_log_infos, set_seed


@logger.catch()
@hydra.main(config_path="../configs/", config_name="params.yaml")
def train(config: DictConfig) -> tf.keras.Model:
    """Training loop.

    When you wrok with Hydra, all the logic of the funtion has to be contained in
    the function decorated by `@hydra.main(...)`. You can't define another function in
    same script asthe one containing the `@hydra.main(...)` decorated one.

    You can import functions from other scripts and use it in the `@hydra.main(...)` decorated
    function, that's fine, but you can't write other functions in the same script.

    Also, be careful to the root file : Hydra modify the root file.

    `Path(__file__).parent.parent` will return `.`, but this root will be located in the
    `hydra` folder, not the real root of the folder. **This is to be expected**, as the job of
    `Hydra` is to monitor and record each iteration of the training loop for
    reproducibility, `hydra` create a new folder for each training loop.

    The name of the folder were the configuration of the training loop has been configured
    to be related to the run name :

    If we have :

    * `run_name: ${backbone.backbone_name}_${segmentation_model.name}_${now:%Y-%m-%d_%H-%M-%S}`

    The various files of this training loop will be stored in :

    * `dir: hydra/${now:%Y-%m-%d_%H-%M-%S}`.

    To access to the root path of the folder, we use the `hydra.utils.get_original_cwd()` command.

    Args:
        config (DictConfig): The hydra configuration file used for the training loop.

    Returns:
        A trained tf.keras model.
    """
    _, repo_path = set_log_infos(config)

    if config.mixed_precision.activate:
        logger.info("Setting training policy.")
        policy = tf.keras.mixed_precision.Policy("mixed_float16")
        tf.keras.mixed_precision.set_global_policy(policy)
        logger.info(f"Layers computations dtype : {policy.compute_dtype}")
        logger.info(f"Layers variables dtype : {policy.variable_dtype}")

    mlflow.set_tracking_uri(f"file://{repo_path}/mlruns")
    mlflow.set_experiment(config.mlflow.experiment_name)

    set_seed(config.prepare.seed)

    logger.info("Data loading")

    logger.info(f"Root path of the folder : {repo_path}")
    logger.info(f"MLFlow uri : {mlflow.get_tracking_uri()}")
    with mlflow.start_run(
        run_name=config.mlflow.run_name,
    ) as run:

        logger.info(f"Run infos : {run.info}")

        mltensorflow.autolog(every_n_iter=1)
        mlflow.log_params(flatten_omegaconf(config))

        logger.info("Instantiate data pipeline.")
        pipeline = instantiate(config.pipeline)

        ds = pipeline.create_train_dataset(
            Path(repo_path) / config.datasets.prepared_dataset.train,
            config.datasets.params.batch_size,
            config.datasets.params.repetitions,
            config.datasets.params.augment,
            config.datasets.params.prefetch,
        )

        ds_val = pipeline.create_test_dataset(
            Path(repo_path) / config.datasets.prepared_dataset.val,
            config.datasets.params.batch_size,
            config.datasets.params.repetitions,
            config.datasets.params.prefetch,
        )

        # logger.info("Instantiate weight sampling")
        # ds = ds.map(add_sample_weights)
        # ds_val = ds_val.map(add_sample_weights)

        logger.info("Instantiate model")

        if True:
            ...
        else:
            ...

        backbone = {"backbone": instantiate(config.backbone)}
        model = instantiate(config.segmentation_model, **backbone)

        if config.lrd.activate:
            logger.info("Found learning rate decay policy.")
            lr = {"learning_rate": instantiate(config.lr_decay)}
        else:
            lr = {"learning_rate": config.training.lr}

        logger.info("Instantiate optimizer")
        optimizer = instantiate(config.optimizer, **lr)

        logger.info("Instantiate loss")
        loss = instantiate(config.losses)

        logger.info("Instantiate metrics")
        metric = instantiate(config.metrics)

        logger.info("Compiling model")
        model.compile(
            optimizer=optimizer,
            loss=loss,
            metrics=[metric],
        )
        callbacks = [
            tf.keras.callbacks.ModelCheckpoint(
                f"callback_{config.mlflow.run_name}",
                monitor="val_mean_iou",
                mode="max",
                save_best_only=True,
                save_weights_only=False,
            ),
        ]

        logger.info("Start training")
        model.summary()
        model.fit(
            ds,
            epochs=config.training.epochs,
            validation_data=ds_val,
            callbacks=callbacks,
        )

        # tf.keras.models.save_model(model, f"{config.mlflow.run_name}")

        model.save(f"{config.mlflow.run_name}")


if __name__ == "__main__":
    train()
