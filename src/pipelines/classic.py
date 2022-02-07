from typing import List, Tuple

import albumentations as A
import numpy as np
import pandas as pd
import tensorflow as tf

from src.pipelines.base_pipeline import BasePipeline

# class Tensorize(object):
#     """
#     Class used to create tensor datasets for TensorFlow.

#     Inheritance:
#         object: The base class of the class hierarchy, used only to enforce WPS306.
#         See https://wemake-python-stylegui.de/en/latest/pages/usage/violations/consistency.html#consistency.

#     Args:
#         n_classes (int): Number of classes in the dataset.
#         img_shape (Tuple[int,int,int]): Dimension of the image, format is (H,W,C).
#         random_seed (int): Fixed random seed for reproducibility.
#     """

#     def __init__(
#         self,
#         n_classes: int,
#         img_shape: Tuple[int, int, int],
#         random_seed: int,
#     ) -> None:
#         """Initialization of the class Tensorize.

#         Initialize the class, the number of classes in the datasets, the shape of the
#         images and the random seed.
#         """

#         self.n_classes = n_classes
#         self.img_shape = img_shape
#         self.random_seed = random_seed
#         self.AUTOTUNE = tf.data.AUTOTUNE

#     def load_images(self, data_frame: pd.DataFrame, column_name: str) -> List[str]:
#         """Load the images as a list.

#         Take the dataframe containing the observations and the masks and the return the
#         column containing the observations as a list.

#         Args:
#             data_frame (pd.DataFrame): Dataframe containing the dataset.
#             column_name (str): The name of the column containing the observations.

#         Returns:
#             The list of observations deduced from the dataframe.
#         """
#         return data_frame[column_name].tolist()

#     @tf.function
#     def parse_image_and_mask(
#         self,
#         image: str,
#         mask: str,
#     ) -> Tuple[np.ndarray, np.ndarray]:
#         """Transform image and mask.

#         Parse image and mask to go from path to a resized np.ndarray.

#         Args:
#             filename (str): The path of the image to parse.
#             mask (str): The mask of the image.

#         Returns:
#             A np.ndarray corresponding to the image and the corresponding one-hot mask.
#         """
#         resized_dims = [self.img_shape[0], self.img_shape[1]]
#         # convert the mask to one-hot encoding
#         # decode image
#         image = tf.io.read_file(image)
#         # Don't use tf.image.decode_image,
#         # or the output shape will be undefined
#         image = tf.image.decode_jpeg(image)
#         # This will convert to float values in [0, 1]
#         image = tf.image.convert_image_dtype(image, tf.float32)
#         image = tf.image.resize(
#             image,
#             resized_dims,
#             method=tf.image.ResizeMethod.NEAREST_NEIGHBOR,
#         )

#         mask = tf.io.read_file(mask)
#         # Don't use tf.image.decode_image,
#         # or the output shape will be undefined
#         mask = tf.io.decode_png(mask, channels=1)
#         mask = tf.image.resize(
#             mask,
#             resized_dims,
#             method=tf.image.ResizeMethod.NEAREST_NEIGHBOR,
#         )

#         return image, mask

#     def train_preprocess(
#         self,
#         image: np.ndarray,
#         mask: np.ndarray,
#     ) -> Tuple[np.ndarray, np.ndarray]:
#         """Augmentation preprocess, if needed.

#         Args:
#             image (np.ndarray): The image to augment.
#             mask (np.ndarray): The corresponding mask.

#         Returns:
#             The augmented pair.
#         """

#         aug = A.Compose(
#             [
#                 A.HorizontalFlip(p=0.5),
#                 A.VerticalFlip(p=0.5),
#                 A.RandomRotate90(p=0.5),
#                 A.Transpose(p=0.5),
#             ],
#         )

#         augmented = aug(image=image, mask=mask)

#         image = augmented["image"]
#         mask = augmented["mask"]

#         image = tf.cast(x=image, dtype=tf.float32)
#         mask = tf.cast(x=mask, dtype=tf.float32)

#         return image, mask

#     @tf.function
#     def apply_augments(
#         self,
#         image: np.ndarray,
#         mask: np.ndarray,
#     ) -> Tuple[np.ndarray, np.ndarray]:
#         """Apply augmentation (roations, transposition, flips), if needed.

#         Args:
#             image (np.ndarray): A numpy array representing an image of the dataset.
#             mask (np.ndarray): A numpy array representing a mask of the dataset.

#         Returns:
#             An augmented pair (image, mask).
#         """

#         image, mask = tf.numpy_function(
#             func=self.train_preprocess,
#             inp=[image, mask],
#             Tout=[tf.float32, tf.float32],
#         )

#         img_shape = [self.img_shape[0], self.img_shape[1], 3]
#         mask_shape = [self.img_shape[0], self.img_shape[1], 1]

#         image = tf.ensure_shape(image, shape=img_shape)
#         mask = tf.ensure_shape(mask, shape=mask_shape)

#         return image, mask

#     def create_train_dataset(
#         self,
#         data_path: str,
#         batch: int,
#         repet: int,
#         prefetch: int,
#         augment: bool,
#     ) -> tf.data.Dataset:
#         """Creation of a tensor dataset for TensorFlow.

#         Args:
#             data_path (str): Path where the csv file containing the dataframe is
#                 located.
#             batch (int): Batch size, usually 32.
#             repet (int): How many times the dataset has to be repeated.
#             prefetch (int): How many batch the CPU has to prepare in advance for the
#                 GPU.
#             augment (bool): Does the dataset has to be augmented or no.

#         Returns:
#             A batch of observations and masks.
#         """
#         df = pd.read_csv(data_path)
#         features = self.load_images(data_frame=df, column_name="filename")
#         masks = self.load_images(data_frame=df, column_name="mask")

#         dataset = tf.data.Dataset.from_tensor_slices((features, masks))
#         dataset = dataset.cache()
#         dataset = dataset.shuffle(len(features), seed=self.random_seed)
#         dataset = dataset.repeat(repet)
#         dataset = dataset.map(
#             self.parse_image_and_mask,
#             num_parallel_calls=self.AUTOTUNE,
#         )
#         if augment:
#             dataset = dataset.map(self.apply_augments, num_parallel_calls=self.AUTOTUNE)
#         dataset = dataset.batch(batch)
#         return dataset.prefetch(prefetch)

#     def create_test_dataset(
#         self,
#         data_path: str,
#         batch: int,
#         repet: int,
#         prefetch: int,
#     ) -> tf.data.Dataset:
#         """Creation of a tensor dataset for TensorFlow.

#         Args:
#             data_path (str): Path where the csv file containing the dataframe is
#                 located.
#             batch (int): Batch size, usually 32.
#             repet (int): How many times the dataset has to be repeated.
#             prefetch (int): How many batch the CPU has to prepare in advance for the
#                 GPU.
#             augment (bool): Does the dataset has to be augmented or no.

#         Returns:
#             A batch of observations and masks.
#         """
#         df = pd.read_csv(data_path)
#         features = self.load_images(data_frame=df, column_name="filename")
#         masks = self.load_images(data_frame=df, column_name="mask")

#         dataset = tf.data.Dataset.from_tensor_slices((features, masks))
#         dataset = dataset.cache()
#         dataset = dataset.shuffle(len(features), seed=self.random_seed)
#         dataset = dataset.repeat(repet)
#         dataset = dataset.map(
#             self.parse_image_and_mask,
#             num_parallel_calls=self.AUTOTUNE,
#         )
#         dataset = dataset.batch(batch)
#         return dataset.prefetch(prefetch)


class BaseDataset(BasePipeline):
    """
    Class used to create tensor datasets for TensorFlow.

    Inheritance:
        object: The base class of the class hierarchy, used only to enforce WPS306.
        See https://wemake-python-stylegui.de/en/latest/pages/usage/violations/consistency.html#consistency.

    Args:
        n_classes (int): Number of classes in the dataset.
        img_shape (Tuple[int,int,int]): Dimension of the image, format is (H,W,C).
        random_seed (int): Fixed random seed for reproducibility.
    """

    def __init__(
        self,
        *args,
        **kwargs,
    ) -> None:
        """Initialization of the class Tensorize.

        Initialize the class, the number of classes in the datasets, the shape of the
        images and the random seed.
        """
        super().__init__(
            *args,
            **kwargs,
        )

    def create_train_dataset(
        self,
        data_path: str,
        batch: int,
        repet: int,
        prefetch: int,
        augment: bool,
    ) -> tf.data.Dataset:
        """Creation of a tensor dataset for TensorFlow.

        Args:
            data_path (str): Path where the csv file containing the dataframe is
                located.
            batch (int): Batch size, usually 32.
            repet (int): How many times the dataset has to be repeated.
            prefetch (int): How many batch the CPU has to prepare in advance for the
                GPU.
            augment (bool): Does the dataset has to be augmented or no.

        Returns:
            A batch of observations and masks.
        """
        df = pd.read_csv(data_path)
        features = self.load_images(data_frame=df, column_name="filename")
        masks = self.load_images(data_frame=df, column_name="mask")

        dataset = tf.data.Dataset.from_tensor_slices((features, masks))
        dataset = dataset.cache()
        dataset = dataset.shuffle(len(features), seed=self.random_seed)
        dataset = dataset.repeat(repet)
        dataset = dataset.map(
            self.parse_image_and_mask,
            num_parallel_calls=self.AUTOTUNE,
        )
        if augment:
            dataset = dataset.map(self.apply_augments, num_parallel_calls=self.AUTOTUNE)
        dataset = dataset.batch(batch)
        return dataset.prefetch(prefetch)
