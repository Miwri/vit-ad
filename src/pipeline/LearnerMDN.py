"""Module to start different training loops and save results in weights & biases"""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

import wandb
from src.classes.CnnEncoder import ResNetEncoder
from src.classes.MixtureDensityNetwork import GaussianMixtureDensityNetwork
from src.classes.MixtureDensityNetwork import mdn_loss as mdn_loss_fun
from src.classes.transformer.TransformerEncoder import TransformerEncoder
from src.data_loader.GeneralDataLoader import GeneralDataLoader
from src.pipeline.LearnerRecon import HyperParameterConfig
from src.pipeline.ValidatorMDN import ValidatorMdn
from src.util.ImageHelper import ImageHelper
from src.util.TrainingsHelper import early_stopping


class LearnerMDN:
    """Class which contains functions for different Gaussian Mixture Model learning pipelines.
    Args:
        model: Encoder, feature extractor which is used for training
        num_gaussians: int, number of gaussians for used MDN, default=10
        enable_wandb: bool, default=True, enables savon of model weights and training parameters in weights & biases
    """

    def __init__(
        self,
        feature_extractor: ResNetEncoder | TransformerEncoder,
        enable_wandb: bool = True,
    ) -> None:
        # save shared objects in instance
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.feature_extractor = feature_extractor

        self.enable_wandb = enable_wandb
        self.validator = None

        # calculate and print model size
        param_size = 0
        for param in feature_extractor.parameters():
            param_size += param.nelement() * param.element_size()
        buffer_size = 0
        for buffer in feature_extractor.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()
        size_all_mb = (param_size + buffer_size) / 1024**2
        print(f"model size: {size_all_mb}MB")

    def init_training(
        self,
        hyper_param_dict: HyperParameterConfig,
    ):
        """Function to register training run in weights & biases:
        Args:
            hyper_param_dict: typed dictionary, which contains the hyperparameter configuration
        """

        self.save_prefix = (
            str(hyper_param_dict["epochs"])
            + "-epochs_"
            + ImageHelper().get_current_timestamp()
            + "_"
            + hyper_param_dict["dataset"]
            + "_"
            + hyper_param_dict["dataclass"]
        )

        model_architecture = f"{hyper_param_dict['num_gaussians']}_{type(self.feature_extractor).__name__}_{hyper_param_dict['decoder']}_{hyper_param_dict['dataset']}_{hyper_param_dict['dataclass']}"

        if self.enable_wandb:
            wandb.init(
                project="masterthesis",
                name=f"{model_architecture}-{ImageHelper().get_current_timestamp()}",
                config={
                    "architecture": model_architecture,
                    "encoder": type(self.feature_extractor).__name__,
                    "encoder_type": self.feature_extractor.architecture,
                }
                | hyper_param_dict,
            )

        trainable_params = sum(
            p.numel() for p in self.feature_extractor.parameters() if p.requires_grad
        )
        non_trainable_params = sum(
            p.numel() for p in self.feature_extractor.parameters() if not p.requires_grad
        )
        total_params = trainable_params + non_trainable_params

        print(
            f"Trainable params: {trainable_params} \n Non trainable params: {non_trainable_params} \n Total params: {total_params}"
        )

    def learn_mdn_transformer(
        self,
        hyper_param_dict: HyperParameterConfig,
        train_loader: DataLoader,
        valid_loader: DataLoader,
        test_loader: GeneralDataLoader,
    ) -> None:
        """Function to train a Mixture Density Network only, requires a trained transformer Encoder which delivers an image embedding.
        Args:
            train_loader: DataLoader to deliver training data
            valid_loader: DataLoader to deliver validation data
            test_loader: GeneralDataloader to deliver test data for model evaluation
            hyper_param_dict: typed dictionary, which contains the hyperparameter configuration
        """

        if not isinstance(self.feature_extractor, TransformerEncoder):
            print(
                "Feature Extractor needs to be of type TransformerEncoder. Please preload and freeze weights. Training aborted."
            )
            return

        mdn = GaussianMixtureDensityNetwork(
            cluster_centers=None,
            input_dim=self.feature_extractor.size_patch_embedding,
            output_dim=self.feature_extractor.size_patch_embedding,
            num_gaussians=hyper_param_dict["num_gaussians"],
        )

        self.feature_extractor.to(self.device)
        mdn.to(self.device)

        self.init_training(hyper_param_dict=hyper_param_dict)

        optimizer = torch.optim.Adam(
            list(mdn.parameters()),
            lr=hyper_param_dict["learning_rate"],
            weight_decay=hyper_param_dict["weight_decay"],
        )

        min_valid_loss = np.inf
        best_weights = []
        not_improved = 0

        for epoch in range(hyper_param_dict["epochs"]):
            mdn_loss = 0.0
            valid_loss = 0.0

            with tqdm(train_loader, unit="batch") as tepoch:
                # training
                mdn.train()
                for images in tepoch:
                    images = images.to(self.device)

                    # train only mdn
                    output = self.feature_extractor(images)
                    mixture_result = mdn(output.patch_embedding)
                    loss_mdn = mdn_loss_fun(
                        x=output.patch_embedding,
                        pi=mixture_result.pi,
                        sigma=mixture_result.sigma,
                        mu=mixture_result.mu,
                    )

                    optimizer.zero_grad()
                    loss_mdn.backward()
                    optimizer.step()

                    tepoch.set_postfix({"loss_mdn": loss_mdn.item()})

                    mdn_loss += loss_mdn.item() * images.size(0)

                    del images
                    torch.cuda.empty_cache()

            # validation
            mdn.eval()
            for images in valid_loader:
                images = images.to(self.device)

                output = self.feature_extractor(images)
                mixture_result = mdn(output.patch_embedding)
                loss = mdn_loss_fun(
                    x=output.patch_embedding,
                    pi=mixture_result.pi,
                    sigma=mixture_result.sigma,
                    mu=mixture_result.mu,
                )
                valid_loss += loss.item() * images.size(0)

                del images
                torch.cuda.empty_cache()

            mdn_loss = mdn_loss / len(train_loader.dataset)
            valid_loss = valid_loss / len(valid_loader.dataset)

            print(
                f"Epoch: {epoch + 1} \tMDN Loss: {mdn_loss} \tValidation Loss: {valid_loss}"
            )

            (
                min_valid_loss,
                not_improved,
                continue_learning,
                best_weights,
            ) = early_stopping(
                min_valid_loss=min_valid_loss,
                valid_loss=valid_loss,
                not_improved=not_improved,
                epoch=epoch,
                patience=hyper_param_dict["patience"],
                models=[mdn],
                best_weights=best_weights,
                save_suffix=self.save_prefix,
            )

            if self.enable_wandb:
                wandb.log(
                    {
                        "mdn_loss": mdn_loss,
                        "valid_loss": min_valid_loss,
                        "epoch": epoch,
                        "stage": "train",
                    }
                )

            if not continue_learning:
                break

        if self.enable_wandb:
            validator = ValidatorMdn(
                gmm_model=[mdn],
                feature_extractor=self.feature_extractor,
                dataloader=test_loader,
                weights_object=best_weights,
                props={
                    "fp_thres": 0.3,
                    "num_gaussians": hyper_param_dict["num_gaussians"],
                    "dataset": hyper_param_dict["dataset"],
                    "dataclass": hyper_param_dict["dataclass"],
                },
            )
            validator.calc_all_metrics(new_wandb_run=False)

            wandb.finish()

    def learn_mdn_resnet(
        self,
        hyper_param_dict: HyperParameterConfig,
        train_loader: DataLoader,
        valid_loader: DataLoader,
        test_loader: GeneralDataLoader,
    ) -> None:
        """Function to train a Mixture Density Network only, requires a trained ResNet Encoder which delivers feature maps for different stages.
        Args:
            train_loader: DataLoader to deliver training data
            valid_loader: DataLoader to deliver validation data
            test_loader: DataLoader to deliver test data for model evaluation
            hyper_param_dict: typed dictionary, which contains the hyperparameter configuration
        """
        if not isinstance(self.feature_extractor, ResNetEncoder):
            print(
                "Feature Extractor needs to be of type ResNetEncoder. Please preload and freeze weights. Training aborted."
            )
            return

        # cluster_centers = init_cluster_centers(
        #     dataloader=train_loader,
        #     encoder=self.feature_extractor,
        #     num_clusters=hyper_param_dict["num_gaussians"],
        # )

        mdn_list: list[GaussianMixtureDensityNetwork] = []
        # hardcode training of two gmms
        for i in range(2, 4):
            mdn_list.append(
                GaussianMixtureDensityNetwork(
                    cluster_centers=None,
                    input_dim=self.feature_extractor.res_net.in_channels[i],
                    output_dim=self.feature_extractor.res_net.in_channels[i],
                    num_gaussians=hyper_param_dict["num_gaussians"],
                )
            )
            mdn_list[i - 2].to(self.device)

        self.feature_extractor.to(self.device)

        self.init_training(hyper_param_dict=hyper_param_dict)

        parameter_list = []

        for mdn in mdn_list:
            parameter_list += list(mdn.parameters())

        optimizer = torch.optim.Adam(
            parameter_list + list(self.feature_extractor.parameters()),
            lr=hyper_param_dict["learning_rate"],
            weight_decay=hyper_param_dict["weight_decay"],
        )

        min_valid_loss = np.inf
        best_weights = []
        not_improved = 0

        for epoch in range(hyper_param_dict["epochs"]):
            mdn_loss = 0.0
            valid_loss = 0.0

            with tqdm(train_loader, unit="batch") as tepoch:
                # training
                for images in tepoch:
                    images = images.to(self.device)
                    loss_mdn = 0

                    feature_maps, _ = self.feature_extractor(images, separate_layer=True)
                    for i, mdn_model in enumerate(mdn_list):
                        # start with the second feature because of memory issues
                        feature = feature_maps[i + 2]
                        num_features = feature.shape[2] ** 2
                        mdn_input = feature.reshape(
                            -1,
                            self.feature_extractor.res_net.in_channels[i + 2],
                            num_features,
                        ).transpose(2, 1)

                        mdn_model.train()
                        mixture_result = mdn_model(mdn_input)

                        loss_mdn += mdn_loss_fun(
                            x=mdn_input,
                            pi=mixture_result.pi,
                            sigma=mixture_result.sigma,
                            mu=mixture_result.mu,
                        )

                    optimizer.zero_grad()
                    loss_mdn.backward()
                    optimizer.step()

                    avg_loss = loss_mdn.item() / len(mdn_list)

                    tepoch.set_postfix({"loss_mdn": avg_loss})

                    mdn_loss += avg_loss * images.size(0)

            # validation
            for images in valid_loader:
                images = images.to(self.device)

                loss = 0
                feature_maps, _ = self.feature_extractor(images, separate_layer=True)
                for i, mdn_model in enumerate(mdn_list):
                    feature = feature_maps[i + 2]
                    mdn_model.eval()
                    num_features = feature.shape[2] ** 2
                    mdn_input = feature.reshape(
                        -1,
                        self.feature_extractor.res_net.in_channels[i + 2],
                        num_features,
                    ).transpose(2, 1)
                    mixture_result = mdn_model(mdn_input)

                    loss += mdn_loss_fun(
                        x=mdn_input,
                        pi=mixture_result.pi,
                        sigma=mixture_result.sigma,
                        mu=mixture_result.mu,
                    )

                valid_loss += (loss.item() * images.size(0)) / len(mdn_list)

            mdn_loss = mdn_loss / len(train_loader.dataset)
            valid_loss = valid_loss / len(valid_loader.dataset)

            print(
                f"Epoch: {epoch + 1} \tMDN Loss: {mdn_loss} \tValidation Loss: {valid_loss}"
            )

            if self.enable_wandb:
                (
                    min_valid_loss,
                    not_improved,
                    continue_learning,
                    best_weights,
                ) = early_stopping(
                    min_valid_loss=min_valid_loss,
                    valid_loss=valid_loss,
                    not_improved=not_improved,
                    epoch=epoch,
                    patience=hyper_param_dict["patience"],
                    best_weights=best_weights,
                    models=mdn_list,
                    save_suffix=self.save_prefix,
                )

                wandb.log(
                    {
                        "mdn_loss": mdn_loss,
                        "valid_loss": min_valid_loss,
                        "epoch": epoch,
                        "stage": "train",
                        "block_index": [2, 3],
                    }
                )

                if not continue_learning:
                    break

        if self.enable_wandb:
            validator = ValidatorMdn(
                gmm_model=mdn_list,
                feature_extractor=self.feature_extractor,
                dataloader=test_loader,
                weights_object=best_weights,
                props={
                    "fp_thres": 0.3,
                    "num_gaussians": hyper_param_dict["num_gaussians"],
                    "dataset": hyper_param_dict["dataset"],
                    "dataclass": hyper_param_dict["dataclass"],
                },
            )
            validator.calc_all_metrics(new_wandb_run=False)

            wandb.finish()
