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
from src.classes.CnnAutoEncoder import VanillaAutoEncoder
from src.data_loader.GeneralDataLoader import GeneralDataLoader
from src.util.ValidationHelper import ValidationProps, ValidLoopReturn, calc_all_metrics

WEIGHTS = "weights"


class ValidatorRecon:
    """Class to compute scores on trained model weights

    Args:
        model: AutoEncoder, model to use for predictions
        dataloader: GeneralDataLoder, dataloader to load test data batch wise
        props: ValidationProps, dataset and metric configurations as defined in TypedDict
        weights_object: dict | None, weights to be loaded as dict, normally used when validation is used in test run
        weights_base_path: str, folder where weights to be loaded are saved, only used when weights_object is None
        weights_name: str, name of model weights to be loaded, only used when weights_object is None
    """

    def __init__(
        self,
        model: VanillaAutoEncoder,
        dataloader: GeneralDataLoader,
        props: ValidationProps,
        weights_object: dict = None,
        weights_base_path: str = "",
        weights_name: str = "",
    ) -> None:
        self.model = model
        self.dataloader = dataloader
        self.dataset_name = f"{props['dataset']}_{props['dataclass']}"
        self.run_name = f"recon_{type(model.decoder).__name__}"
        self.props = props

        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"

        print(f"evaluating on {self.device}")

        if weights_object is not None:
            model.load_state_dict(weights_object)
        else:
            weights_path = os.path.join(weights_base_path, weights_name)
            self.model.load_state_dict(
                torch.load(weights_path, map_location=torch.device("cpu"))
            )

    def calc_all_metrics(self, centering: bool = False, new_wandb_run: bool = True):
        """Fuction to calculate AUROC score on image and pixel level for reconstruction based approach
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

            result = self.valid_loop_mse(test_loader)

            calc_all_metrics(
                result=result,
                fp_thres=self.props["fp_thres"],
                dataset_name=self.dataset_name,
                # vmax=np.median(result["pixel_scores"]) * 5,
                vmax=0.15,
            )

        finally:
            if new_wandb_run:
                wandb.finish()
            plt.close(fig="all")

    def valid_loop_mse(self, dataloader: DataLoader) -> ValidLoopReturn:
        """Helper function to calculate loss elementwise for a given dataset
        Args:
            dataloader: DataLoader, loader for test data to be loaded batchwise
        """
        self.model.to(self.device)

        pixel_anomaly_score_list = []
        pixel_label_list = []
        image_anomaly_score_list = []
        image_label_list = []
        orig_list = []
        recon_list = []
        with tqdm(dataloader, unit="batch") as tepoch:
            self.model.eval()
            for images, pixel_labels, image_labels in tepoch:
                images = images.to(self.device)
                output = self.model(images)
                mse = self.model.MSELoss(output=output.reconstruction, x=images)
                anomaly_score = torch.mean(input=mse, dim=1, keepdim=True)

                pixel_anomaly_score_list.append(anomaly_score.detach().cpu().numpy())
                pixel_label_list.append(pixel_labels.detach().cpu().numpy())
                image_anomaly_score_list.append(
                    torch.amax(anomaly_score, (1, 2, 3)).detach().cpu().numpy()
                )
                image_label_list.append(image_labels.detach().cpu().numpy())
                orig_list.append(images.detach().cpu().numpy())
                recon_list.append(output.reconstruction.detach().cpu().numpy())

        image_score_array = np.concatenate(image_anomaly_score_list, axis=0)
        pixel_score_array = np.concatenate(pixel_anomaly_score_list, axis=0)
        image_label_array = np.concatenate(image_label_list, axis=0)
        pixel_label_array = np.concatenate(pixel_label_list, axis=0)
        orig_array = np.concatenate(orig_list, axis=0)
        recon_array = np.concatenate(recon_list, axis=0)

        return {
            "image_scores": image_score_array,
            "pixel_scores": pixel_score_array,
            "image_labels": image_label_array,
            "pixel_labels": pixel_label_array,
            "origs": orig_array,
            "recons": recon_array,
        }
