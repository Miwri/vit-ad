from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.data_loader.GeneralDataLoader import GeneralDataLoader

IMG_EXT = ".jpg"
MEAN_ARRAY = np.array([0, 0, 0])
STD_ARRAY = np.array([1, 1, 1])
COLORS = ["blue", "green", "red", "orange", "purple"]


"""
Helper module to plot images and loss curves.
"""


@dataclass
class LossPlotObject:
    loss: list[float]
    label: str
    color: str


class ImageHelper:
    def __init__(self, colour_restrict=True) -> None:
        self.colour_restrict = colour_restrict

    def get_current_day_of_month(self) -> str:
        return datetime.now().strftime("%Y-%m-%d")

    def get_current_time_of_day(self) -> str:
        return datetime.now().strftime("%H-%M-%S")

    def get_current_timestamp(self) -> str:
        return self.get_current_day_of_month() + "_" + self.get_current_time_of_day()

    def show_image(
        self,
        img_array: np.ndarray,
        std: np.ndarray = STD_ARRAY,
        mean: np.ndarray = MEAN_ARRAY,
        vmin: float = 0.0,
        vmax: float = 1.0,
    ) -> None:
        """function to display a single image with matplotlib
        args:
            img_array: numpy array of image value, should have the shape: [channel, length, width], channel is normally 1 or 3
            std: numpy array of standard deviation for each channel. Standard is [1, 1, 1]. Is used to reverse channel wise standardization.
            mean: numpy array of means for each channel. Standard is [0,0,0]. Is used to reverse channel wise standardization.
            vmin: min value of possible image values
            vmax: max value of possible image values
        """
        std = std if not std is None else STD_ARRAY
        mean = mean if not mean is None else MEAN_ARRAY

        return plt.imshow(img_array.transpose(1, 2, 0) * std + mean, vmin=vmin, vmax=vmax)

    def plot_recons(self, recons: np.ndarray):
        """Function to create a figure of nine reconstructions.
        Args:
            recons: numpy array of reconstructions shape [batch_size, channel, img_size, img_size]
        """

        fig_recons, ax_recons = plt.subplots(nrows=3, ncols=3)

        fig_recons.set_figheight(12)
        fig_recons.set_figwidth(12)

        for i, axis in enumerate(ax_recons.flat):
            axis.imshow(recons[i].transpose(1, 2, 0))
            axis.label_outer()

        return fig_recons

    def plot_heatmap(
        self,
        anomaly_score_maps: np.ndarray,
        ground_truth: np.ndarray,
        orig_images: np.ndarray | None = None,
        vmin: float = 0.0,
        vmax: float = 1.0,
    ):
        """Function to plot heat map and corresponding ground truth
        Args:
            anomaly_score_maps: numpy array of anomaly scores, shape: [batch_size, 1, img_size, img_size]
            ground_truth: numpy array of ground truth images
            orig_images: numpy array of original images
            vmin: float, min value of possible image values, default = 0.0
            vmax: float, max value of possible image values, default = 1.0
        Return:
        """

        fig_heat_map, ax_heat_map_list = plt.subplots(nrows=3, ncols=3)

        fig_heat_map.set_figheight(12)
        fig_heat_map.set_figwidth(15)

        for i, axis in enumerate(ax_heat_map_list.flat):
            img = axis.imshow(
                anomaly_score_maps[i].transpose(1, 2, 0),
                cmap="jet",
                vmin=vmin,
                vmax=vmax,
            )
            axis.label_outer()

        fig_heat_map.colorbar(mappable=img, ax=ax_heat_map_list, shrink=0.5)

        fig_ground_truth, ax_ground_truth_list = plt.subplots(nrows=3, ncols=3)

        fig_ground_truth.set_figheight(12)
        fig_ground_truth.set_figwidth(12)

        for i, axis in enumerate(ax_ground_truth_list.flat):
            axis.imshow(ground_truth[i].transpose(1, 2, 0))
            axis.label_outer()

        if orig_images is not None:
            fig_orig, ax_orig_list = plt.subplots(nrows=3, ncols=3)

            fig_orig.set_figheight(15)
            fig_orig.set_figwidth(22)

            for i, axis in enumerate(ax_orig_list.flat):
                axis.imshow(orig_images[i].transpose(1, 2, 0))
                axis.label_outer()

            for i, axis in enumerate(ax_orig_list.flat):
                img = axis.imshow(
                    anomaly_score_maps[i].transpose(1, 2, 0),
                    cmap="jet",
                    vmin=vmin,
                    vmax=vmax,
                    alpha=0.5,
                )
                axis.label_outer()

                fig_orig.colorbar(mappable=img, ax=axis, shrink=0.9)

            return fig_heat_map, fig_ground_truth, fig_orig

        return fig_heat_map, fig_ground_truth, None

    def plot_loss_in_one(
        self,
        loss_array: list[LossPlotObject],
        title: str = "validation loss",
        epochs: int = 300,
        size: tuple[int] = (18, 20),
        font_size: str = "16",
        path: str = "loss",
    ):
        """Function to plot several loss curves in one plot.
        Args:
            loss_array: array of data objects. Apart from loss data it should contain the desired label and color for each object.
            title: str, title for plot
            epochs: number, of trained epochs to set x-axis length, default=300
            size: tuple[int], to set size of figure, default=(18, 20)
            font_size: str number
            path: str, path to save loss to
        """
        fig = plt.figure(figsize=size)
        plt.rcParams["font.size"] = font_size

        ax = fig.add_subplot(title=title)
        ax.set_xlabel("Epochs")
        ax.set_ylabel("Loss")
        for loss_dict in loss_array:
            plt.xlim([0, epochs])
            ax.semilogy(loss_dict.loss, label=loss_dict.label, color=loss_dict.color)

        ax.legend()

        Path(path).mkdir(parents=True, exist_ok=True)

        plt.savefig(os.path.join(path, title))

    def load_json_data(self, datapath: str):
        """Function to load json data from file"""
        with open(datapath, "r", encoding="utf8") as f:
            data = json.load(f)

        return data

    def create_los_array(self, loss_object: dict) -> list[LossPlotObject]:
        loss_array = []

        for i, (key, value) in enumerate(loss_object.items()):
            loss_plot_object = LossPlotObject(loss=value, label=key, color=COLORS[i])
            loss_array.append(loss_plot_object)
            print(f"min_{key}: {min(value)}")

        return loss_array

    # helper class to plot loss curve from json file
    # args:
    #   path: path to folder which contains file
    #   name: file name without file type ending '.json'

    def load_and_plot_loss(
        self, path: str, file_name: str, epochs: int, size=(5, 5), font_size=10
    ):
        loss = self.load_json_data(os.path.join(path, file_name + ".json"))

        loss_array = self.create_los_array(loss_object=loss)

        self.plot_loss_in_one(
            loss_array=loss_array,
            epochs=epochs,
            size=size,
            font_size=font_size,
            path=path,
            title=file_name,
        )


def generate_data_distribution_plot(
    base_path: str,
    all_prods: list[str],
    train_path: str,
    test_path: str,
    dataset_name: str,
):
    """Function to generate bar plot to visualize ration between valid and invalid image data and train and test data.
    Args:
        base_path: str, path to dataset
        all_prods: list[str], list of all different classes in dataset
        train_path: str, suffix to load train data
        test_path: str, suffix to load test data
    """

    train_array = []
    test_valid_array = []
    test_invalid_array = []

    for prod in all_prods:
        prod_base = base_path + prod
        number_train_data = len(
            GeneralDataLoader(
                img_size=224,
                base_path=prod_base,
                batch_size=64,
                data_path=train_path,
                validation_mode=True,
            )
            .get_dataloader()
            .dataset
        )
        all_labels = GeneralDataLoader(
            img_size=224,
            base_path=prod_base,
            batch_size=16,
            data_path=test_path,
            validation_mode=True,
        ).load_all_data_at_once(only_labels=True)
        number_invalid_test_data = np.count_nonzero(all_labels)
        number_good_test_data = len(all_labels) - number_invalid_test_data

        train_array.append(number_train_data)
        test_invalid_array.append(number_invalid_test_data)
        test_valid_array.append(number_good_test_data)

    data = {
        "train": train_array,
        "test_defected": test_invalid_array,
        "test_normal": test_valid_array,
    }

    x = np.arange(len(train_array))
    width = 0.25
    multiplier = 0

    fig, ax = plt.subplots(layout="constrained")

    max_amount = 0

    for type_data, amount_data in data.items():
        if np.max(amount_data) > max_amount:
            max_amount = np.max(amount_data, axis=None)
        offset = width * multiplier
        rects = ax.bar(x + offset, amount_data, width=width, label=type_data)
        # TODO show here percentage instead of absolute values, only if time is left :)
        ax.bar_label(rects, padding=3)
        multiplier += 1

    fig_width = len(train_array) if len(train_array) > 5 else 5

    ax.set_xticks(x + width, all_prods)
    ax.set_ylabel("number of image samples")
    ax.legend(loc="upper left", ncols=3)
    ax.set_ylim(0, max_amount + (max_amount * 0.2))
    ax.set_title(f"Data distribution of {dataset_name} dataset")
    # fig.set_figheight(12)
    fig.set_figwidth(fig_width)

    plt.savefig(f"{dataset_name}_dataset_ratio")

    return fig
