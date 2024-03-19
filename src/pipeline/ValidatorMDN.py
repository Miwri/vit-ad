"""Module to compute and save model metrics
"""
from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.nn.functional import interpolate
from torch.utils.data import DataLoader
from tqdm import tqdm

import wandb
from src.classes.CnnEncoder import ResNetEncoder
from src.classes.MixtureDensityNetwork import (
    GaussianMixtureDensityNetwork,
    get_probability_map,
)
from src.classes.transformer.TransformerEncoder import TransformerEncoder
from src.data_loader.GeneralDataLoader import GeneralDataLoader
from src.util.ValidationHelper import ValidationProps, ValidLoopReturn, calc_all_metrics

WEIGHTS = "weights"


class ValidatorMdn:
    """Class to compute scores on trained model weights

    Args:
        gmm_model: list[GaussianMixtureDensityNetwork], list of gmms to be used for evaluation, only one when feature_extractor is a transformer
        feature_extractor: ResNetEncoder | TransformerEncoder, extractor which provides features, is always pre-trained
        dataloader: GeneralDataLoder, dataloader to load test data batch wise
        props: ValidationProps, dataset and metric configurations as defined in TypedDict
        weights_object: list[dict] | None, weights to be loaded in gmm_model as dict, normally used when validation is used in test run
        weights_base_path: str, folder where weights to be loaded are saved, only used when weights_object is None
        weights_name: list[str], name of model weights to be loaded, only used when weights_object is None
    """

    def __init__(
        self,
        gmm_model: list[GaussianMixtureDensityNetwork],
        feature_extractor: ResNetEncoder | TransformerEncoder,
        dataloader: GeneralDataLoader,
        props: ValidationProps,
        weights_object: list[dict] = None,
        weights_base_path: str = "",
        weights_name: list[str] = "",
    ) -> None:
        self.gmm_model = gmm_model
        self.feature_extractor = feature_extractor
        self.dataloader = dataloader
        self.dataset_name = f"{props['dataset']}_{props['dataclass']}"
        self.run_name = f"gmm_{props['num_gaussians']}"
        self.props = props

        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"

        print(f"evaluating on {self.device}")

        if weights_object is not None:
            for i, model in enumerate(self.gmm_model):
                model.load_state_dict(weights_object[i])
        else:
            for i, model in enumerate(self.gmm_model):
                weights_path = os.path.join(weights_base_path, weights_name[i])
                model.load_state_dict(
                    torch.load(weights_path, map_location=torch.device("cpu"))
                )

    def calc_all_metrics(self, centering: bool = False, new_wandb_run: bool = True):
        """Fuction to calculate AUROC score on image and pixel level for MDN based approach
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

        try:
            test_loader = self.dataloader.get_dataloader(centering=centering)

            if isinstance(self.feature_extractor, ResNetEncoder):
                result = self.valid_loop_resnet(test_loader)
            else:
                result = self.valid_loop_transformer(test_loader)

            calc_all_metrics(
                result=result,
                fp_thres=self.props["fp_thres"],
                dataset_name=self.dataset_name,
            )

        finally:
            if new_wandb_run:
                wandb.finish()
            plt.close(fig="all")

    def valid_loop_transformer(self, dataloader: DataLoader) -> ValidLoopReturn:
        """Helper function to calculate loss elementwise for a given dataset with a transformer as encoder
        Args:
            dataloader: DataLoader, loader for test data to be loaded batchwise
        """

        model = self.gmm_model[0]

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

                features = self.feature_extractor(images)
                mdn_result = model(features.patch_embedding)
                probability_map = get_probability_map(
                    x=features.patch_embedding,
                    pi=mdn_result.pi,
                    sigma=mdn_result.sigma,
                    mu=mdn_result.mu,
                )
                image_scores = torch.amin(probability_map, (1)).detach().cpu().numpy()

                pixel_scores = [[] for el in probability_map]

                for i, el in enumerate(probability_map):
                    mask = el.reshape(
                        -1,
                        int(
                            self.feature_extractor.img_size
                            / self.feature_extractor.patch_size
                        ),
                        int(
                            self.feature_extractor.img_size
                            / self.feature_extractor.patch_size
                        ),
                    )
                    pixel_scores[i] = (
                        interpolate(
                            mask.unsqueeze(1),
                            size=(
                                self.feature_extractor.img_size,
                                self.feature_extractor.img_size,
                            ),
                            mode="bilinear",
                            align_corners=True,
                        )
                        .squeeze()
                        .cpu()
                        .numpy()
                    )

                image_anomaly_score_list.append(image_scores)
                pixel_anomaly_score_list.append(pixel_scores)
                image_label_list.append(image_labels.detach().cpu().numpy())
                pixel_label_list.append(pixel_labels.detach().cpu().numpy())
                orig_list.append(images.detach().cpu().numpy())

        image_score_array = (np.concatenate(image_anomaly_score_list, axis=0) * (-1)) + 1
        pixel_score_array = (np.concatenate(pixel_anomaly_score_list, axis=0) * (-1)) + 1
        pixel_score_array = np.expand_dims(pixel_score_array, axis=1)
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

    def valid_loop_resnet(self, dataloader: DataLoader) -> ValidLoopReturn:
        """Helper function to calculate loss elementwise for a given dataset with a resnet as encoder
        Args:
            dataloader: DataLoader, loader for test data to be loaded batchwise
        """
        self.feature_extractor.to(self.device)

        wandb.log(
            {
                "block_index": [2, 3],
            }
        )

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
                for i, mdn_model in enumerate(self.gmm_model):
                    mdn_model.to(self.device)
                    mdn_model.eval()
                    feature = feature_maps[i + 2]
                    num_features = feature.shape[2] ** 2
                    mdn_input = feature.reshape(
                        -1,
                        self.feature_extractor.res_net.in_channels[i + 2],
                        num_features,
                    ).transpose(2, 1)
                    mixture_result = mdn_model(mdn_input)

                    probability_map = get_probability_map(
                        x=mdn_input,
                        pi=mixture_result.pi,
                        sigma=mixture_result.sigma,
                        mu=mixture_result.mu,
                    )
                    feature_map_len = int(
                        self.feature_extractor.img_size
                        / self.feature_extractor.res_net.scales[i + 2]
                    )

                    stage_scores = probability_map.reshape(
                        probability_map.shape[0],
                        1,
                        feature_map_len,
                        feature_map_len,
                    )
                    stage_scores = interpolate(
                        stage_scores,
                        size=(
                            self.feature_extractor.img_size,
                            self.feature_extractor.img_size,
                        ),
                        mode="bilinear",
                        align_corners=True,
                    )

                    stage_anomalies = (stage_scores * (-1)) + 1
                    stage_anomaly_list.append(stage_anomalies)

                anomaly_list = torch.stack(stage_anomaly_list, dim=-1)
                anomaly_scores = torch.mean(anomaly_list, dim=-1)

                pixel_anomaly_score_list.append(anomaly_scores.detach().cpu().numpy())
                image_anomaly_score_list.append(
                    torch.amin(anomaly_scores, dim=(1, 2, 3)).detach().cpu().numpy()
                )
                pixel_label_list.append(pixel_labels.detach().cpu().numpy())
                image_label_list.append(image_labels.detach().cpu().numpy())
                orig_list.append(images.detach().cpu().cpu().numpy())

        image_score_array = (np.concatenate(image_anomaly_score_list, axis=0) * (-1)) + 1
        pixel_score_array = (np.concatenate(pixel_anomaly_score_list, axis=0) * (-1)) + 1
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
