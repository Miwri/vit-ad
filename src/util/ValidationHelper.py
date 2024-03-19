"""Helper module to calculate scores and thresholds"""

from __future__ import annotations

from typing import TypedDict

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn import metrics

import wandb
from src.util.ImageHelper import ImageHelper


class ValidationProps(TypedDict):
    """class to type validation props
    keys:
        num_gaussians: int, number of gaussians used
        dataclass: str, name used data class in dataset
        dataset: str, name of used dataset
        fp_thres: inf, threshold for false-positive rate
    """

    num_gaussians: int | None
    dataclass: str
    dataset: str
    fp_thres: int


class ValidLoopReturn(TypedDict):
    """Class to type return of Validation Loop for each model"""

    image_scores: torch.Tensor
    pixel_scores: torch.Tensor
    image_labels: torch.Tensor
    pixel_labels: torch.Tensor
    origs: torch.Tensor
    recons: torch.Tensor | None


def calc_auroc(
    anomaly_map: torch.Tensor,
    test_labels: torch.Tensor,
    dataset_name: str,
):
    """Function to calc elementwise AUROC and PRAUC score
    Args:
        anomaly_map: Tensor, shape [batch_size]
        test_labels: Tensor, shape [batch_size]
        dataset_name: str, name of dataset for labeling of curve
    Return: fig_auroc, fig_prauc
    """

    fig_auroc, ax_auroc = plt.subplots()
    fig_prauc, ax_prauc = plt.subplots()

    metrics.RocCurveDisplay.from_predictions(
        y_true=test_labels, y_pred=anomaly_map, ax=ax_auroc, name=f"AUROC-{dataset_name}"
    )
    metrics.PrecisionRecallDisplay.from_predictions(
        y_true=test_labels, y_pred=anomaly_map, ax=ax_prauc, name=f"PRAUC-{dataset_name}"
    )

    roc_auc_score = metrics.roc_auc_score(y_true=test_labels, y_score=anomaly_map)

    return fig_auroc, fig_prauc, roc_auc_score


def calc_threshold(
    anomaly_map: np.ndarray, test_labels: np.ndarray, fpr_threshold: float = 0.3
):
    """Function to calculate threshold for classification of pixels as normal or anormal. Calculation is oriented of the one described in Mishra et al. 2021.
    Calculation is orientend on the maximum PRO score with the condition that FPR cannot be higher then fpr_threshold
    Args:
        anomaly_map: Tensor, map of shape (1, IMG_SIZE, IMG_SIZE) which contains anomaly scores per pixel
        test_labels: Tensor, map of shape (1, IMG_SIZE, IMG_SIZE) which contains binary values. 1 in anomalous regions, 0 in normal regions.
        fpr_threshold: float, float value which limits threshold calculation to a given false positive rate
    """

    fpr, tpr, thresholds = metrics.roc_curve(y_true=test_labels, y_score=anomaly_map)
    indices_thresholded = np.where(fpr <= fpr_threshold)
    tp_thresholded = tpr[indices_thresholded]
    max_tpr = np.argmax(tp_thresholded)
    threshold = thresholds[max_tpr]

    return threshold


def predict_anomaly(
    anomaly_map: np.ndarray, threshold: float, classification_type: str = "binary"
):
    """Function to classify each image or pixel as normal or anormal depending on a given threshold
    Args:
        anomaly_map: ndarray,
        threshold: float,
        classification_type: str, whether classification should be made binary as 0 - normal and 1 - anormal or fluently with 0 - normal and anomaly value for anormal
    """
    if classification_type == "binary":
        return np.where(anomaly_map > threshold, 1, 0)

    return np.where(anomaly_map > threshold, anomaly_map, 0)


def create_heatmap_from_scores(
    anomaly_map: torch.Tensor, pixel_labels: torch.Tensor, fpr_threshold: float
) -> torch.Tensor:
    """Takes a given anomaly map, calculates a threshold with a given false-positive-rate and returns a thresholded anomaly map.
    Values below the threshold are set to zero while the values above remain.
    Args:
        anomaly_map: Tensor, map with anomaly scores
        pixel_labels: Tensor, map with binary pixel scores
        fpr_threshold: float, threshold for the maximum false-positive rate
    """

    threshold = calc_threshold(
        anomaly_map=anomaly_map.flatten(),
        test_labels=pixel_labels.flatten(),
        fpr_threshold=fpr_threshold,
    )

    anomalies = predict_anomaly(
        anomaly_map=anomaly_map,
        threshold=threshold,
        classification_type="fluently",
    )

    return anomalies


def calc_all_metrics(
    result: ValidLoopReturn,
    fp_thres: float,
    dataset_name: str,
    vmin: float = 0,
    vmax: float = 1,
    enable_wandb: bool = True,
):
    """Function to calculate pixel and image level metrics for anomaly detection and localization
    Args:
        result: ValidLoopReturn, heatmaps and labels derived from the model to evaluate
        fp_thres: float, threshold for false positive rate to calculate heatmaps and PRO score
        dataset_name: str, name and data class of used samples to set props
        vmin: float, min color value
        vmax: float, nmax color value
        enable_wandb: bool, whether to log the run on a existing weights & biases run or not, defaults to True
    """

    anomalies = create_heatmap_from_scores(
        anomaly_map=result["pixel_scores"],
        pixel_labels=result["pixel_labels"],
        fpr_threshold=fp_thres,
    )

    fig_img_auroc, fig_img_prauc, img_roc_auc_score = calc_auroc(
        anomaly_map=torch.Tensor(result["image_scores"]),
        test_labels=torch.Tensor(result["image_labels"]),
        dataset_name=dataset_name,
    )

    fig_pixel_auroc, _, pixel_roc_auc_score = calc_auroc(
        anomaly_map=torch.Tensor(result["pixel_scores"]).flatten(),
        test_labels=torch.Tensor(result["pixel_labels"]).flatten(),
        dataset_name=dataset_name,
    )

    fig_heat_map, fig_ground_truth, fig_map_origs = ImageHelper().plot_heatmap(
        anomaly_score_maps=anomalies,
        ground_truth=result["pixel_labels"],
        orig_images=result["origs"],
        vmin=vmin,
        vmax=vmax,
    )

    pro_score = metrics.roc_auc_score(
        y_true=result["pixel_labels"].flatten(),
        y_score=anomalies.flatten(),
    )

    precision, recall, _ = metrics.precision_recall_curve(
        y_true=result["image_labels"], probas_pred=result["image_scores"]
    )
    prauc_score = metrics.auc(y=precision, x=recall)

    if "recons" in result:
        fig_recons = ImageHelper().plot_recons(result["recons"])
        wandb.log({"reconstructions": wandb.Image(fig_recons)})

    print(
        f"detection AUROC: {img_roc_auc_score}, localization AUROC: {pixel_roc_auc_score}"
    )

    if enable_wandb:
        wandb.log(
            {
                "heat_maps": wandb.Image(fig_heat_map),
                "map_origs": wandb.Image(fig_map_origs),
                "ground_truth": wandb.Image(fig_ground_truth),
                "pixel_auroc": wandb.Image(fig_pixel_auroc),
                "image_auroc": wandb.Image(fig_img_auroc),
                "image_prauc": wandb.Image(fig_img_prauc),
                "image_auroc_score": img_roc_auc_score,
                "image_prauc_score": prauc_score,
                "pixel_auroc_score": pixel_roc_auc_score,
                f"pro_score_{fp_thres}fp": pro_score,
                "fp_thres": fp_thres,
                "stage": "eval",
            }
        )

    return fig_heat_map
