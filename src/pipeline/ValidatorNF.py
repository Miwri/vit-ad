"""Module to compute and save model metrics
"""
from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

import wandb
from src.classes.CnnEncoder import ResNetEncoder
from src.classes.NormalizingFlow import NormalizingFlow
from src.classes.transformer.TransformerEncoder import TransformerEncoder
from src.data_loader.GeneralDataLoader import GeneralDataLoader
from src.util.ValidationHelper import ValidationProps, ValidLoopReturn, calc_all_metrics

WEIGHTS = "weights"

BLOCK_INDEX_DEIT = 0


class ValidatorNF:
    """Class to compute scores for Normalizing Flow model on trained weights

    Args:
        nf_model: list[NormalizingFlow], list of nfs to be used for evaluation, only one when feature_extractor is a transformer
        feature_extractor: ResNetEncoder | TransformerEncoder, extractor which provides features, is always pre-trained
        dataloader: GeneralDataLoder, dataloader to load test data batch wise
        props: ValidationProps, dataset and metric configurations as defined in TypedDict
        weights_object: list[dict] | None, weights to be loaded in nf_model as dict, normally used when validation is used in test run
        weights_base_path: str, folder where weights to be loaded are saved, only used when weights_object is None
        weights_name: list[str], name of model weights to be loaded, only used when weights_object is None
    """

    def __init__(
        self,
        nf_model: list[NormalizingFlow],
        feature_extractor: ResNetEncoder | TransformerEncoder,
        dataloader: GeneralDataLoader,
        props: ValidationProps,
        weights_object: list[any] = None,
        weights_base_path: str = "",
        weights_name: list[str] = "",
    ) -> None:
        self.nf_model = nf_model
        self.dataloader = dataloader
        self.feature_extractor = feature_extractor
        self.dataset_name = f"{props['dataset']}_{props['dataclass']}"
        self.run_name = "nf"
        self.props = props

        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"

        if weights_object is not None:
            for i, model in enumerate(self.nf_model):
                model.load_state_dict(weights_object[i])
        else:
            for i, model in enumerate(self.nf_model):
                weights_path = os.path.join(weights_base_path, weights_name[i])
                model.load_state_dict(
                    torch.load(weights_path, map_location=torch.device("cpu"))
                )

    def calc_all_metrics(self, centering: bool = False, new_wandb_run: bool = True):
        """Function to calculate image wise area under roc curve for anomaly detection score from normalizing flow calculated likelihood

        Args:
            centering: bool, default=False, whether data will be used centered or not
            new_wandb_run: bool, default=True, whether a new wandb run should be initialized
        """

        if new_wandb_run:
            wandb.init(
                project="masterthesis",
                name=f"Eval-{self.run_name}-on-{self.dataset_name}",
                config=self.props,
            )
            wandb.log(
                {
                    "block_index": [1, 2, 3]
                    if isinstance(self.feature_extractor, ResNetEncoder)
                    else BLOCK_INDEX_DEIT
                }
            )
        try:
            test_loader = self.dataloader.get_dataloader(centering=centering)

            if isinstance(self.feature_extractor, ResNetEncoder):
                result = self.valid_loop_resnet_nf(test_loader)
            else:
                result = self.valid_loop_transformer_nf(test_loader)

            calc_all_metrics(
                result=result,
                fp_thres=self.props["fp_thres"],
                dataset_name=self.dataset_name,
            )

        finally:
            if new_wandb_run:
                wandb.finish()
            plt.close(fig="all")

    def valid_loop_transformer_nf(self, dataloader: DataLoader) -> ValidLoopReturn:
        """Helper function to calculate loss elementwise for a given dataset with a transformer as feature extractor
        Args:
            dataloader: DataLoader, loader for test data to be loaded batchwise
        """
        model = self.nf_model[0]
        model.to(self.device)
        self.feature_extractor.to(self.device)

        pixel_anomaly_score_list = []
        pixel_label_list = []
        image_anomaly_score_list = []
        image_label_list = []
        orig_list = []
        with tqdm(dataloader, unit="batch") as tepoch:
            model.eval()
            for images, pixel_labels, image_labels in tepoch:
                images = images.to(self.device)

                embedding = self.feature_extractor(
                    images, block_index=BLOCK_INDEX_DEIT
                ).patch_embedding

                feature_map_size = int(np.sqrt(embedding.shape[1]))
                channel_size = embedding.shape[2]
                embedding = embedding.transpose(2, 1).reshape(
                    (-1, channel_size, feature_map_size, feature_map_size)
                )
                result = model(embedding)

                image_anomaly_score = (
                    torch.amax(result.anomaly_score_map, dim=(1, 2, 3))
                    .detach()
                    .cpu()
                    .numpy()
                )

                pixel_anomaly_score_list.append(
                    result.anomaly_score_map.detach().cpu().numpy()
                )
                pixel_label_list.append(pixel_labels.detach().cpu().numpy())
                image_anomaly_score_list.append(image_anomaly_score)
                image_label_list.append(image_labels.detach().cpu().numpy())
                orig_list.append(images.detach().cpu().numpy())

        image_score_array = np.concatenate(image_anomaly_score_list, axis=0)
        pixel_score_array = np.concatenate(pixel_anomaly_score_list, axis=0)
        image_label_array = np.concatenate(image_label_list, axis=0)
        pixel_label_array = np.concatenate(pixel_label_list, axis=0)
        orig_array = np.concatenate(orig_list, axis=0)

        return {
            "image_scores": image_score_array,
            "pixel_scores": pixel_score_array,
            "image_labels": image_label_array,
            "pixel_labels": pixel_label_array,
            "origs": orig_array,
        }

    def valid_loop_resnet_nf(self, dataloader: DataLoader) -> ValidLoopReturn:
        """Helper function to calculate loss elementwise for a given dataset with a transformer as feature extractor
        Args:
            dataloader: DataLoader, loader for test data to be loaded batchwise
        """

        self.feature_extractor.to(self.device)

        pixel_anomaly_score_list = []
        pixel_label_list = []
        image_anomaly_score_list = []
        image_label_list = []
        orig_list = []
        with tqdm(dataloader, unit="batch") as tepoch:
            for images, pixel_labels, image_labels in tepoch:
                images = images.to(self.device)
                stage_anomaly_list = []
                feature_maps, _ = self.feature_extractor(images, separate_layer=True)

                for i, nf_model in enumerate(self.nf_model):
                    nf_model.to(self.device)
                    nf_model.eval()
                    feature_map = feature_maps[i + 1]

                    result = nf_model(feature_map)

                    stage_anomaly_list.append(result.anomaly_score_map)

                anomaly_list = torch.stack(stage_anomaly_list, dim=-1)
                anomaly_scores = torch.mean(anomaly_list, dim=-1)

                image_anomaly_score = (
                    torch.amax(anomaly_scores, dim=(1, 2, 3)).detach().cpu().numpy()
                )

                pixel_anomaly_score_list.append(anomaly_scores.detach().cpu().numpy())
                pixel_label_list.append(pixel_labels.detach().cpu().numpy())
                image_anomaly_score_list.append(image_anomaly_score)
                image_label_list.append(image_labels.detach().cpu().numpy())
                orig_list.append(images.detach().cpu().numpy())

        image_score_array = np.concatenate(image_anomaly_score_list, axis=0)
        pixel_score_array = np.concatenate(pixel_anomaly_score_list, axis=0)
        image_label_array = np.concatenate(image_label_list, axis=0)
        pixel_label_array = np.concatenate(pixel_label_list, axis=0)
        orig_array = np.concatenate(orig_list, axis=0)

        return {
            "image_scores": image_score_array,
            "pixel_scores": pixel_score_array,
            "image_labels": image_label_array,
            "pixel_labels": pixel_label_array,
            "origs": orig_array,
        }
