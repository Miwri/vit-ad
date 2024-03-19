"""Module to start different training loops and save results in weights & biases"""

from __future__ import annotations

from typing import TypedDict

import numpy as np
import torch
from IPython.display import clear_output
from torch.utils.data import DataLoader
from tqdm import tqdm

import wandb
from src.classes.CnnAutoEncoder import VanillaAutoEncoder
from src.classes.VariationalAutoEncoder import VariationalAutoEncoder
from src.classes.VariationalAutoEncoder import kl_loss as kl_loss_fun
from src.data_loader.GeneralDataLoader import GeneralDataLoader
from src.pipeline.ValidatorRecon import ValidatorRecon
from src.util.ImageHelper import ImageHelper
from src.util.TrainingsHelper import early_stopping


class HyperParameterConfig(TypedDict):
    """Class to type dictionary which transports hyperparameter config.
    keys:
        amount_data: amount of training data,
        learning_rate: used learning rate
        weight_decay: used weight decay
        batch_size: used batch size in data loader
        img_size: size of input images
        patience: times loss can not improve until training is stopped,
        epochs: number of epochs trained
        centering: bool, image normalization (centered over the whole training set or not)
        dataset: name of the dataset used for training,
        num_gaussians: number of gaussians for mixture model,
        hidden_ratio: ratio for hidden dimension in one normalizing flow block
        decoder: string name of model used as decoder, either a CNN decoder or a likelihood estimator
    """

    amount_data: int
    learning_rate: float
    weight_decay: float
    batch_size: int
    img_size: int
    patience: int
    epochs: int
    centering: bool
    dataset: str
    dataclass: str
    num_gaussians: int
    decoder: str
    hidden_ratio: float


class LearnerRecon:
    """Class which contains functions for different learning pipelines.
    Args:
        model: AutoEncoder, model which is used for training
        num_gaussians: int, number of gaussians for used MDN, default=10
        enable_wandb: bool, default=True, enables savon of model weights and training parameters in weights & biases
    """

    def __init__(
        self,
        model: VanillaAutoEncoder,
        enable_wandb: bool = True,
    ) -> None:
        # save shared objects in instance
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.model = model

        self.enable_wandb = enable_wandb

        # calculate and print model size
        param_size = 0
        for param in model.parameters():
            param_size += param.nelement() * param.element_size()
        buffer_size = 0
        for buffer in model.buffers():
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

        model_architecture = type(self.model).__name__ + "_" + hyper_param_dict["decoder"]

        if self.enable_wandb:
            wandb.init(
                project="masterthesis",
                name=f"{model_architecture}-{ImageHelper().get_current_timestamp()}",
                config={
                    "architecture": model_architecture,
                    "encoder": type(self.model.encoder).__name__,
                    "decoder": type(self.model.decoder).__name__,
                    "encoder_type": self.model.architecture,
                }
                | hyper_param_dict,
            )

        self.save_prefix = (
            str(hyper_param_dict["epochs"])
            + "-epochs_"
            + ImageHelper().get_current_timestamp()
            + "_"
            + hyper_param_dict["dataset"]
            + "_"
            + hyper_param_dict["dataclass"]
        )

        trainable_params = sum(
            p.numel() for p in self.model.parameters() if p.requires_grad
        )
        non_trainable_params = sum(
            p.numel() for p in self.model.parameters() if not p.requires_grad
        )
        total_params = trainable_params + non_trainable_params

        print(
            f"Trainable params: {trainable_params} \n Non trainable params: {non_trainable_params} \n Total params: {total_params}"
        )

    def validation_loop(
        self,
        valid_loader: DataLoader,
        vae: bool,
    ) -> float:
        """Function to perform validation after trained epoch.
        Args:
            valid_loader: DataLoader to deliver validation data
            vae: bool, indicator wether training is performed with a Variational AutoEncoder or not, if True calculates KL-loss additionally to MSE
        """
        self.model.eval()
        epoch_valid_loss = 0.0
        for images in valid_loader:
            images = images.to(self.device)

            output = self.model(images)

            if vae:
                kl_loss = kl_loss_fun(
                    mu=output.latent_space.mu, log_var=output.latent_space.log_var
                )
                loss_mse = self.model.MSELoss(output=output.reconstruction, x=images)
                loss_mse_reduced = torch.mean(loss_mse)
                epoch_valid_loss += (
                    loss_mse_reduced.item() * images.size(0) + kl_loss.item()
                )
            else:
                loss_mse = self.model.MSELoss(output=output.reconstruction, x=images)
                loss_mse_reduced = torch.mean(loss_mse)
                epoch_valid_loss += loss_mse_reduced.item() * images.size(0)

        clear_output(wait=True)
        return epoch_valid_loss

    # TODO improve function, differenciate between kl loss and mse
    def learn_vae(
        self,
        train_loader: DataLoader,
        valid_loader: DataLoader,
        hyper_param_dict: HyperParameterConfig,
    ) -> None:
        """Function to train a Variational AutoEncoder model.
        Args:
            train_loader: DataLoader to deliver training data
            valid_loader: DataLoader to deliver validation data
            hyper_param_dict: typed dictionary, which contains the hyperparameter configuration
        """
        if not isinstance(self.model, VariationalAutoEncoder):
            print(
                "Can't train a Variational AutoEncoder with a vanilla AutoEncoder model."
            )
            return

        self.model.to(self.device)

        self.init_training(
            hyper_param_dict=hyper_param_dict,
        )

        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=hyper_param_dict["learning_rate"],
            weight_decay=hyper_param_dict["weight_decay"],
        )

        min_valid_loss = np.inf
        best_weights = []
        not_improved = 1

        for epoch in range(hyper_param_dict["epochs"]):
            train_loss = 0.0
            plain_mse = 0.0
            plain_kl = 0.0

            with tqdm(train_loader, unit="batch") as tepoch:
                # training
                self.model.train()
                for images in tepoch:
                    images = images.to(self.device)

                    output = self.model(images)
                    loss_kl = kl_loss_fun(
                        mu=output.latent_space.mu, log_var=output.latent_space.log_var
                    )

                    loss_mse = self.model.MSELoss(output=output.reconstruction, x=images)

                    # TODO https://stats.stackexchange.com/questions/332179/how-to-weight-kld-loss-vs-reconstruction-loss-in-variational-auto-encoder
                    loss = loss_mse + loss_kl
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    tepoch.set_postfix(loss=loss.item())

                    loss_mse = loss_mse.item() * images.size(0)

                    train_loss += loss_mse + loss_kl.item()

                    plain_mse += loss_mse
                    plain_kl += loss_kl

            # validation
            valid_loss = self.validation_loop(valid_loader=valid_loader, vae=True)

            train_loss /= len(train_loader.dataset)
            valid_loss /= len(valid_loader.dataset)
            plain_mse /= len(train_loader.dataset)
            plain_kl /= len(train_loader.dataset)

            print(
                f"Epoch: {epoch + 1} \tTraining Loss: {train_loss} \tValidation Loss: {valid_loss}"
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
                best_weights=best_weights,
                patience=hyper_param_dict["patience"],
                models=[self.model],
                save_suffix=self.save_prefix,
            )

            if self.enable_wandb:
                wandb.log(
                    {
                        "train_loss": train_loss,
                        "valid_loss": min_valid_loss,
                        "plain_mse_loss": plain_mse,
                        "plain_kl_loss": plain_kl,
                        "epoch": epoch,
                        "stage": "train",
                    }
                )

            if not continue_learning:
                break

        if self.enable_wandb:
            wandb.finish()

    def learn_ae_with_SSIM(
        self,
        hyper_param_dict: HyperParameterConfig,
        train_loader: DataLoader,
        valid_loader: DataLoader,
    ) -> None:
        """Function to train a AutoEncoder model, can have a CNN or a transformer Encoder.
        Training is performed with a combined MSE and SSIM loss.
        Args:
            train_loader: DataLoader to deliver training data
            valid_loader: DataLoader to deliver validation data
            hyper_param_dict: typed dictionary, which contains the hyperparameter configuration
        """
        if isinstance(self.model, VariationalAutoEncoder):
            print(
                "Can't train a Vanilla AutoEncoder or Transformer with a Variational AutoEncoder model."
            )
            return

        self.model.to(self.device)

        self.init_training(hyper_param_dict=hyper_param_dict)

        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=hyper_param_dict["learning_rate"],
            weight_decay=hyper_param_dict["weight_decay"],
        )

        min_valid_loss = np.inf
        best_weights = []
        not_improved = 0

        for epoch in range(hyper_param_dict["epochs"]):
            train_loss = 0.0
            ssim_loss = 0.0
            mse_loss = 0.0

            with tqdm(train_loader, unit="batch") as tepoch:
                # training
                self.model.train()
                for images in tepoch:
                    images = images.to(self.device)

                    # train AE
                    output = self.model(images)
                    loss_mse = self.model.MSELoss(output=output.reconstruction, x=images)
                    loss_ssim = self.model.SSIMLoss(
                        output=output.reconstruction, x=images
                    )

                    loss = 5 * loss_mse + 0.5 * loss_ssim

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    tepoch.set_postfix({"loss": loss.item()})

                    train_loss += loss.item() * images.size(0)
                    ssim_loss += loss_ssim.item() * images.size(0)
                    mse_loss += loss_mse.item() * images.size(0)

                    del images
                    torch.cuda.empty_cache()

            # validation only with mse
            valid_loss = self.validation_loop(valid_loader=valid_loader, vae=False)

            train_loss /= len(train_loader.dataset)
            valid_loss /= len(valid_loader.dataset)
            ssim_loss /= len(train_loader.dataset)
            mse_loss /= len(train_loader.dataset)

            print(
                f"Epoch: {epoch + 1} \tTraining Loss: {train_loss} "
                f"\tValidation Loss: {valid_loss} \tMSE Loss: {mse_loss} \tSSIM Loss: {ssim_loss}"
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
                best_weights=best_weights,
                models=[self.model],
                save_suffix=self.save_prefix,
            )
            if self.enable_wandb:
                wandb.log(
                    {
                        "train_loss": train_loss,
                        "valid_loss": min_valid_loss,
                        "mse_loss": mse_loss,
                        "ssim_loss": ssim_loss,
                        "epoch": epoch,
                        "stage": "train",
                    }
                )

            if not continue_learning:
                break

        if self.enable_wandb:
            wandb.finish()

    def learn_ae_with_MSE_only(
        self,
        hyper_param_dict: HyperParameterConfig,
        train_loader: DataLoader,
        valid_loader: DataLoader,
        test_loader: GeneralDataLoader,
    ) -> None:
        """Function to train a AutoEncoder model, can have a CNN or a transformer Encoder.
        MSE is the only loss used for training.
        Args:
            train_loader: DataLoader to deliver training data
            valid_loader: DataLoader to deliver validation data
            test_loader: GeneralDataloader to deliver test data for model evaluation
            hyper_param_dict: typed dictionary, which contains the hyperparameter configuration
        """
        if isinstance(self.model, VariationalAutoEncoder):
            print(
                "Can't train a Vanilla AutoEncoder or Transformer with a Variational AutoEncoder model."
            )
            return

        self.model.to(self.device)

        self.init_training(hyper_param_dict=hyper_param_dict)

        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=hyper_param_dict["learning_rate"],
            weight_decay=hyper_param_dict["weight_decay"],
        )

        min_valid_loss = np.inf
        best_weights = []
        not_improved = 0

        for epoch in range(hyper_param_dict["epochs"]):
            train_loss = 0.0

            with tqdm(train_loader, unit="batch") as tepoch:
                # training
                self.model.train()
                for images in tepoch:
                    images = images.to(self.device)

                    # train AE
                    output = self.model(images)
                    loss_mse = self.model.MSELoss(output=output.reconstruction, x=images)
                    loss_mse_reduced = torch.mean(loss_mse)

                    optimizer.zero_grad()
                    loss_mse_reduced.backward()
                    optimizer.step()

                    tepoch.set_postfix({"loss": loss_mse_reduced.item()})

                    train_loss += loss_mse_reduced.item() * images.size(0)

                    del images
                    torch.cuda.empty_cache()

            # validation only with mse
            valid_loss = self.validation_loop(valid_loader=valid_loader, vae=False)

            train_loss /= len(train_loader.dataset)
            valid_loss /= len(valid_loader.dataset)

            print(
                f"Epoch: {epoch + 1} \tTraining Loss: {train_loss} "
                f"\tValidation Loss: {valid_loss}"
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
                best_weights=best_weights,
                patience=hyper_param_dict["patience"],
                models=[self.model],
                save_suffix=self.save_prefix,
            )
            if self.enable_wandb:
                wandb.log(
                    {
                        "train_loss": train_loss,
                        "valid_loss": min_valid_loss,
                        "epoch": epoch,
                        "stage": "train",
                    }
                )

            if not continue_learning:
                break

        if self.enable_wandb:
            validator = ValidatorRecon(
                model=self.model,
                dataloader=test_loader,
                weights_object=best_weights[0],
                props={
                    "fp_thres": 0.3,
                    "dataset": hyper_param_dict["dataset"],
                    "dataclass": hyper_param_dict["dataclass"],
                },
            )
            validator.calc_all_metrics(new_wandb_run=False)

            wandb.finish()
