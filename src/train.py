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
def train(config: DictConfig):
    """Training loop.

    Lorsque que l'on travaille avec Hydra, toute la logique de la fonction doit
    être contenu dans `main()`, on ne peut pas faire appel à des fonctions
    tierces extérieures à `main()`, il faut tout coder dedans.

    De même faire attention au dossier root : hydra modifie le dossier root :

    Path(__file__).parent.parent donnera bien `.` mais cette racine est située
    dans le dossier `outputs`, et non dans vrai dossier racine `cracks_defect`.

    Il faut donc ici utiliser `hydra.utils.get_original_cwd()` pour pouvoir
    avoir accès au dossier root `cracks_defect`.

    Pour changer le learning rate d'un optimiseur
    **après avoir compilé le modèle**, voir la question StackOverflow suivante.

    [Modifier de lr](https://stackoverflow.com/questions/
    59737875/keras-change-learning-rate)

    https://stackoverflow.com/questions/59635474/
    whats-difference-between-using-metrics-acc-and-tf-keras-metrics-accuracy

    I'll just add that as of tf v2.2 in training.py the docs say
    "When you pass the strings 'accuracy' or 'acc', we convert this to
    one of tf.keras.metrics.BinaryAccuracy,
    tf.keras.metrics.CategoricalAccuracy,
    tf.keras.metrics.SparseCategoricalAccuracy based on the loss function
    used and the model output shape. We do a similar conversion
    for the strings 'crossentropy' and 'ce' as well."

    Args:
        config (DictConfig): [description]
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
        # if False:
        #     pass
        # else:
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
