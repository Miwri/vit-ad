"""A dataloader module to load train and validation data batch wise
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from os import walk
from os.path import join

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader
from tqdm import tqdm

from .GeneralDataset import GeneralDataset


@dataclass
class DataLoaderObject:
    """Dataclass to type data loader return"""

    train_loader: DataLoader
    valid_loader: DataLoader | None = None


class GeneralDataLoader:
    """A General Dataloader class to provide shared functions to subclasses

    Args:
        batch_size: size of batches in which data is loaded
        base_path: path to folder where data to load is
        img_size: pixel size to which images will be resized, default 512
    """

    def __init__(
        self,
        batch_size: int,
        base_path: str,
        data_path: str,
        valid_path: str | None = None,
        img_size: int = 512,
        validation_mode: bool = False,
    ) -> None:
        self.base_path = base_path

        self.validation_mode = validation_mode

        self.batch_size = batch_size

        self.img_size = img_size

        self.mean: np.ndarray | None = None
        self.std: np.ndarray | None = None

        self.train_file_names = join_to_file_list(
            base_path=base_path, suffix=data_path, shuffle=True
        )

        if valid_path is None and not validation_mode:
            train_index = round(len(self.train_file_names) * 0.8)
            intermediate_container = self.train_file_names[:train_index]
            self.valid_file_names = self.train_file_names[train_index:]
            self.train_file_names = intermediate_container
        elif not validation_mode:
            self.valid_file_names = join_to_file_list(
                base_path=base_path, suffix=valid_path, shuffle=True
            )

    def get_dataloader(
        self, amount_data: int = 0, centering: bool = False, only_labels: bool = False
    ) -> DataLoaderObject | DataLoader:
        """Function to get train and valid dataloader

        Args:
            amount_data: int, absolute amount of train data to be loaded, loads all available data when 0, validation data are always 20%
            centering: bool, indicator if data should be centered with mean and standard deviation of train set. mean and standard deviation will be computed automatically, default: False
            only_labels: bool, whether data_loader returns images or only image_labels, has only effect when validation mode is true, defaults to False

        Return: Data loader object which contains a train loader and a valid loader or just the test_loader if validation_mode = True
        """

        train_file_names = self.train_file_names

        if self.validation_mode:
            if amount_data > 0:
                train_file_names = train_file_names[:amount_data]

            if centering:
                return self.generate_standardized_dataloader(
                    train_file_names, only_labels=only_labels
                )

            return self.generate_dataloader(train_file_names, only_labels=only_labels)

        valid_file_names = self.valid_file_names

        if amount_data > 0:
            train_file_names = train_file_names[:amount_data]
            valid_file_names = self.valid_file_names[: round(amount_data * 0.25)]

        if not centering:
            return DataLoaderObject(
                train_loader=self.generate_dataloader(train_file_names),
                valid_loader=self.generate_dataloader(valid_file_names),
            )

        if self.mean is None or self.std is None:
            raw_dataloader = self.generate_dataloader(self.train_file_names)

            self.compute_mean_stdev(raw_dataloader)

        return DataLoaderObject(
            train_loader=self.generate_standardized_dataloader(train_file_names),
            valid_loader=self.generate_standardized_dataloader(valid_file_names),
        )

    # https://kozodoi.me/python/deep%20learning/pytorch/tutorial/2021/03/08/image-mean-std.html
    def compute_mean_stdev(self, dataloader: DataLoader) -> None:
        """Function to compute mean and standard deviation of a given dataset
        Args:
            dataloader: dataloader which delivers data to compute standard deviation and mean from
        """
        psum = Tensor([0.0, 0.0, 0.0])
        psum_sq = Tensor([0.0, 0.0, 0.0])

        for inputs in tqdm(dataloader):
            psum += inputs.sum(axis=[0, 2, 3])
            psum_sq += (inputs**2).sum(axis=[0, 2, 3])

        pixel_count = len(dataloader.dataset) * self.img_size * self.img_size
        self.mean = (psum / pixel_count).numpy()
        self.std = (psum_sq / pixel_count) - (self.mean**2)
        self.std = torch.sqrt(self.std).numpy()

    def generate_standardized_dataloader(
        self, file_names: list[str], only_labels: bool = False
    ):
        """Helper function to create a standardized dataset and return a dataloader with it"""

        standardized_dataset = GeneralDataset(
            file_names=file_names,
            img_size=self.img_size,
            transform=True,
            mean=self.mean,
            std=self.std,
            validation=self.validation_mode,
            only_labels=only_labels,
        )

        return DataLoader(
            dataset=standardized_dataset,
            batch_size=self.batch_size,
            shuffle=False,
        )

    def generate_dataloader(self, file_names: list[str], only_labels: bool = False):
        """Helper function to create a not standardized dataset and return a dataloader with it"""

        dataset = GeneralDataset(
            file_names=file_names,
            img_size=self.img_size,
            transform=False,
            validation=self.validation_mode,
            only_labels=only_labels,
        )

        return DataLoader(
            dataset=dataset,
            batch_size=self.batch_size,
            shuffle=False,
        )

    def load_all_data_at_once(
        self, centering: bool = False, only_labels: bool = False
    ) -> torch.Tensor:
        """Function to return all test data. Important: Batch size of object is changed to full dataset size."""
        self.batch_size = len(self.train_file_names)

        dataloader = self.get_dataloader(centering=centering, only_labels=only_labels)

        return next(iter(dataloader))


def join_to_file_list(base_path: str, suffix: str, shuffle: bool = True) -> list[str]:
    """Function to read all files from a given path and shuffles them with a given seed to ensure reproducibility

    Args:
        base_path: directory to look for data, all subdirectories will be scanned
        suffix: subdirectories from which data are taken
        shuffle: whether to shuffle the file names with a fixed seed or not, default: True

    Return: list of paths to each file as a string
    """

    paths = []
    base_paths = []

    for root, dirs, _ in walk(base_path):
        for name in dirs:
            path = join(root, name)
            if path.endswith(suffix):
                base_paths.append(path)

    for path in base_paths:
        for root, _, files in walk(path):
            for file in files:
                if file.lower().endswith((".png", ".jpg", ".jpeg", ".tiff", ".bmp")):
                    file_path = join(root, file)
                    paths.append(file_path)

    paths.sort()

    if shuffle:
        random.Random(24).shuffle(paths)

    return paths
