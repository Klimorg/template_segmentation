from typing import List, Tuple

import albumentations as A
import numpy as np
import pandas as pd
import tensorflow as tf
from loguru import logger


class CutMix(object):
    
    def __init__(
        self,
        n_classes: int,
        img_shape: List[int],
        random_seed: int,
        random_seed1: int,
        random_seed2: int,
    ) -> None:
        """Initialization of the class Featurize.

        Initialize the class the number of classes in the datasets, the shape of the
        images and the random seed.

        Args:
            n_classes (int): Number of classes in the dataset.
            img_shape (Tuple[int, int, int]): Dimension of the image, format is (H,W,C).
            random_seed (int): Fixed random seed for reproducibility.
        """
        self.n_classes = n_classes
        self.img_shape = img_shape
        self.random_seed = random_seed
        self.random_seed1 = random_seed1
        self.random_seed2 = random_seed2
        self.AUTOTUNE = tf.data.experimental.AUTOTUNE

        assert self.img_shape[0] == self.img_shape[1]
        self.image_size = self.img_shape[0]

    def load_images(self, data_frame: pd.DataFrame, column_name: str) -> List[str]:
        """Load the images as a list.

        Take the dataframe containing the observations and the labels and the return the
        column containing the observations as a list.

        Args:
            data_frame (pd.DataFrame): Dataframe containing the dataset.
            column_name (str): The name of the column containing the observations.

        Returns:
            The list of observations deduced from the dataframe.
        """
        return data_frame[column_name].tolist()

    @tf.function
    def parse_image_and_mask(
        self, image: str, mask: str
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Transform image and mask.

        Parse image and mask to go from path to a resized np.ndarray.

        Args:
            filename (str): The path of the image to parse.
            mask (str): The mask of the image.

        Returns:
            A np.ndarray corresponding to the image and the corresponding one-hot mask.
        """
        resized_dims = [self.img_shape[0], self.img_shape[1]]
        # decode image
        image = tf.io.read_file(image)
        # Don't use tf.image.decode_image,
        # or the output shape will be undefined
        image = tf.image.decode_jpeg(image)
        # This will convert to float values in [0, 1]
        image = tf.image.convert_image_dtype(image, tf.float32)
        image = tf.image.resize(
            image, resized_dims, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR
        )

        mask = tf.io.read_file(mask)
        # Don't use tf.image.decode_image,
        # or the output shape will be undefined
        mask = tf.io.decode_png(mask, channels=1)
        mask = tf.image.resize(
            mask, resized_dims, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR
        )

        return image, mask

    def create_double_dataset(
        self,
        data_path: str,
    ):

        df = pd.read_csv(data_path)
        features = self.load_images(data_frame=df, column_name="filename")
        masks = self.load_images(data_frame=df, column_name="mask")

        dataset_one = tf.data.Dataset.from_tensor_slices((features, masks))
        dataset_one = dataset_one.shuffle(len(features), seed=self.random_seed1)
        dataset_one = dataset_one.map(
            self.parse_image_and_mask, num_parallel_calls=self.AUTOTUNE
        )

        dataset_two = tf.data.Dataset.from_tensor_slices((features, masks))
        dataset_two = dataset_two.shuffle(len(features), seed=self.random_seed2)
        dataset_two = dataset_two.map(
            self.parse_image_and_mask, num_parallel_calls=self.AUTOTUNE
        )

        return tf.data.Dataset.zip((dataset_one, dataset_two))

    def sample_beta_distribution(
        self, size: int, concentration0: List[float], concentration1: List[float]
    ):

        gamma1sample = tf.random.gamma(shape=[size], alpha=concentration1)
        gamma2sample = tf.random.gamma(shape=[size], alpha=concentration0)

        return gamma1sample / (gamma1sample + gamma2sample)

    @tf.function
    def get_box(self, lambda_value):

        cut_rat = tf.math.sqrt(1.0 - lambda_value)

        cut_w = self.image_size * cut_rat  # rw
        cut_w = tf.cast(cut_w, tf.int32)

        cut_h = self.image_size * cut_rat  # rh
        cut_h = tf.cast(cut_h, tf.int32)

        cut_x = tf.random.uniform(
            (1,), minval=0, maxval=self.image_size, dtype=tf.int32
        )  # rx

        cut_y = tf.random.uniform(
            (1,), minval=0, maxval=self.image_size, dtype=tf.int32
        )  # ry

        boundaryx1 = tf.clip_by_value(cut_x[0] - cut_w // 2, 0, self.image_size)
        boundaryy1 = tf.clip_by_value(cut_y[0] - cut_h // 2, 0, self.image_size)
        bbx2 = tf.clip_by_value(cut_x[0] + cut_w // 2, 0, self.image_size)
        bby2 = tf.clip_by_value(cut_y[0] + cut_h // 2, 0, self.image_size)

        target_h = bby2 - boundaryy1
        if target_h == 0:
            target_h += 1

        target_w = bbx2 - boundaryx1
        if target_w == 0:
            target_w += 1

        return boundaryx1, boundaryy1, target_h, target_w

    @tf.function
    def cutmix(self, train_ds_one, train_ds_two):

        (image1, mask1), (image2, mask2) = train_ds_one, train_ds_two

        alpha = [0.25]
        beta = [0.25]

        # Get a sample from the Beta distribution
        lambda_value = self.sample_beta_distribution(1, alpha, beta)
        # Define Lambda
        lambda_value = lambda_value[0][0]
        # Get the bounding box offsets, heights and widths
        boundaryx1, boundaryy1, target_h, target_w = self.get_box(lambda_value)

        # Get a patch from the second image (`image2`)
        img_crop2 = tf.image.crop_to_bounding_box(
            image2, boundaryy1, boundaryx1, target_h, target_w
        )
        # Pad the `image2` patch (`crop2`) with the same offset
        image2 = tf.image.pad_to_bounding_box(
            img_crop2, boundaryy1, boundaryx1, self.image_size, self.image_size
        )
        # Get a patch from the first image (`image1`)
        img_crop1 = tf.image.crop_to_bounding_box(
            image1, boundaryy1, boundaryx1, target_h, target_w
        )
        # Pad the `image1` patch (`crop1`) with the same offset
        img1 = tf.image.pad_to_bounding_box(
            img_crop1, boundaryy1, boundaryx1, self.image_size, self.image_size
        )

        # Modify the first image by subtracting the patch from `image1`
        # (before applying the `image2` patch)
        image1 -= img1
        # Add the modified `image1` and `image2`  together to get the CutMix image
        image = image1 + image2

        # Get a patch from the second image (`image2`)
        mask_crop2 = tf.image.crop_to_bounding_box(
            mask2, boundaryy1, boundaryx1, target_h, target_w
        )
        # Pad the `image2` patch (`crop2`) with the same offset
        mask2 = tf.image.pad_to_bounding_box(
            mask_crop2, boundaryy1, boundaryx1, self.image_size, self.image_size
        )
        # Get a patch from the first image (`image1`)
        mask_crop1 = tf.image.crop_to_bounding_box(
            mask1, boundaryy1, boundaryx1, target_h, target_w
        )
        # Pad the `image1` patch (`crop1`) with the same offset
        msk1 = tf.image.pad_to_bounding_box(
            mask_crop1, boundaryy1, boundaryx1, self.image_size, self.image_size
        )

        # Modify the first image by subtracting the patch from `image1`
        # (before applying the `image2` patch)
        mask1 -= msk1
        # Add the modified `image1` and `image2`  together to get the CutMix image
        mask = mask1 + mask2

        return image, mask

    def train_preprocess(
        self, image: np.ndarray, mask: List[int]
    ) -> Tuple[np.ndarray, List[int]]:
        """Augmentation preprocess, if needed.

        Args:
            image (np.ndarray): The image to augment.
            mask (np.ndarray): The corresponding mask.

        Returns:
            The augmented pair.
        """

        aug = A.Compose(
            [
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                A.Transpose(p=0.5),
            ]
        )

        augmented = aug(image=image, mask=mask)

        image = augmented["image"]
        mask = augmented["mask"]

        image = tf.cast(x=image, dtype=tf.float32)
        mask = tf.cast(x=mask, dtype=tf.float32)

        return image, mask

    @tf.function
    def apply_augments(self, image, mask):

        image, mask = tf.numpy_function(
            func=self.train_preprocess, inp=[image, mask], Tout=[tf.float32, tf.float32]
        )

        img_shape = [self.img_shape[0], self.img_shape[1], 3]
        mask_shape = [self.img_shape[0], self.img_shape[1], 1]

        image = tf.ensure_shape(image, shape=img_shape)
        mask = tf.ensure_shape(mask, shape=mask_shape)

        return image, mask

    def create_train_dataset(
        self, data_path: str, batch: int, repet: int, augment: bool, prefecth: int
    ):

        dataset = self.create_double_dataset(data_path)
        dataset = dataset.shuffle(len(dataset), seed=self.random_seed)
        dataset = dataset.cache()
        dataset = dataset.repeat(repet)
        dataset = dataset.map(self.cutmix, num_parallel_calls=self.AUTOTUNE)
        if augment:
            dataset = dataset.map(self.apply_augments, num_parallel_calls=self.AUTOTUNE)
        dataset = dataset.batch(batch)

        return dataset.prefetch(prefecth)

    def create_test_dataset(
        self,
        data_path: str,
        batch: int,
        repet: int,
        prefetch: int,
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
            self.parse_image_and_mask, num_parallel_calls=self.AUTOTUNE
        )
        dataset = dataset.batch(batch)
        return dataset.prefetch(prefetch)
