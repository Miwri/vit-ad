from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

import wandb
from src.classes.CnnEncoder import ResNetEncoder
from src.classes.NormalizingFlow import NormalizingFlow
from src.classes.transformer.TransformerEncoder import TransformerEncoder
from src.data_loader.GeneralDataLoader import GeneralDataLoader
from src.pipeline.LearnerRecon import HyperParameterConfig
from src.pipeline.ValidatorNF import ValidatorNF
from src.util.ImageHelper import ImageHelper
from src.util.TrainingsHelper import early_stopping

BLOCK_INDEX_DEIT = 0


class LearnerNF:
    """Class which contains functions for different learning pipelines.
    Args:
        encoder: Encoder, feature extractor model which is used to create a patch embedding, in our case always a transformer
        hidden_ratio: float, ratio of hidden dim in fast flow model
        flow_steps: int, number of flow steps in normalizing flow model, 20 recommended for transformer models
        enable_wandb: bool, default=True, enables save of model weights and training parameters in weights & biases
    """

    def __init__(
        self,
        encoder: ResNetEncoder | TransformerEncoder,
        hidden_ratio: float,
        flow_steps: int,
        enable_wandb: bool = True,
    ) -> None:
        # save shared objects in instance
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.encoder = encoder
        self.hidden_ratio = hidden_ratio
        self.flow_steps = flow_steps

        self.enable_wandb = enable_wandb
        self.save_prefix = ""

    def init_training(
        self,
        nf_model: NormalizingFlow,
        hyper_param_dict: HyperParameterConfig,
    ):
        """Function to register training run in weights & biases:
        Args:
            nf_model: NormalizingFlow, decoder model to determine architecture for logs
            hyper_param_dict: typed dictionary, which contains the hyperparameter configuration
        """

        model_architecture = (
            type(self.encoder).__name__
            + "_"
            + type(nf_model).__name__
            + "_"
            + nf_model.flow_type
        )

        if self.enable_wandb:
            wandb.init(
                project="masterthesis",
                name=f"{model_architecture}-{ImageHelper().get_current_timestamp()}",
                config={
                    "architecture": model_architecture,
                    "encoder": type(self.encoder).__name__,
                    "encoder_type": self.encoder.architecture,
                    "flow_type": nf_model.flow_type,
                }
                | hyper_param_dict,
            )

        self.save_prefix = (
            str(hyper_param_dict["epochs"])
            + "-epochs_"
            + "_img_size_"
            + str(hyper_param_dict["img_size"])
            + "_"
            + ImageHelper().get_current_timestamp()
            + "_"
            + hyper_param_dict["dataset"]
            + "_"
            + hyper_param_dict["dataclass"]
        )

    def train_with_transformer(
        self,
        train_loader: DataLoader,
        valid_loader: DataLoader,
        test_loader: GeneralDataLoader,
        hyper_param_dict: HyperParameterConfig,
    ):
        """function to train nf_model with transformer encoder
        Args:
            train_loader: DataLoader to deliver training data
            valid_loader: DataLoader to deliver validation data
            test_loader: GeneralDataloader to deliver test data for model evaluation
            hyper_param_dict: typed dictionary, which contains the hyperparameter configuration
        """

        nf_model = NormalizingFlow(
            num_channels=self.encoder.size_patch_embedding,
            img_size=self.encoder.img_size,
            num_patches=self.encoder.num_embedded_patches,
            hidden_ratio=self.hidden_ratio,
            flow_steps=self.flow_steps,
        )

        self.encoder.to(self.device)
        nf_model.to(self.device)

        self.init_training(hyper_param_dict=hyper_param_dict, nf_model=nf_model)

        optimizer = torch.optim.Adam(
            nf_model.parameters(),
            lr=hyper_param_dict["learning_rate"],
            weight_decay=hyper_param_dict["weight_decay"],
        )

        min_valid_loss = np.inf
        best_weights = []
        not_improved = 0

        for epoch in range(hyper_param_dict["epochs"]):
            train_loss = 0.0
            valid_loss = 0.0

            with tqdm(train_loader, unit="batch") as tepoch:
                nf_model.train()
                for images in tepoch:
                    images = images.to(self.device)
                    embedding = self.encoder(
                        images, block_index=BLOCK_INDEX_DEIT
                    ).patch_embedding
                    feature_map_size = int(np.sqrt(embedding.shape[1]))
                    channel_size = embedding.shape[2]
                    embedding = embedding.transpose(2, 1).reshape(
                        (-1, channel_size, feature_map_size, feature_map_size)
                    )
                    result = nf_model(embedding)

                    loss = result.loss

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    tepoch.set_postfix(loss=loss.item())

                    train_loss += loss.item() * images.size(0)

                    # anomaly_maps = result.anomaly_score_map.detach().tolist()

            nf_model.eval()
            for images in valid_loader:
                images = images.to(self.device)

                embedding = self.encoder(
                    images, block_index=BLOCK_INDEX_DEIT
                ).patch_embedding
                feature_map_size = int(np.sqrt(embedding.shape[1]))
                channel_size = embedding.shape[2]
                embedding = embedding.transpose(2, 1).reshape(
                    (-1, channel_size, feature_map_size, feature_map_size)
                )
                result = nf_model(embedding)

                loss = result.loss

                valid_loss += loss.item() * images.size(0)

            train_loss /= len(train_loader.dataset)
            valid_loss = valid_loss / len(valid_loader.dataset)

            print(
                f"Epoch: {epoch + 1} \tNF Loss: {train_loss} \tValidation Loss: {valid_loss}"
            )

            if self.enable_wandb:
                (
                    min_valid_loss,
                    not_improved,
                    continue_learning,
                    best_weights,
                ) = early_stopping(
                    save_suffix=self.save_prefix,
                    valid_loss=valid_loss,
                    min_valid_loss=min_valid_loss,
                    epoch=epoch,
                    best_weights=best_weights,
                    not_improved=not_improved,
                    patience=hyper_param_dict["patience"],
                    models=[nf_model],
                )

                wandb.log(
                    {
                        "nf_loss": train_loss,
                        "valid_loss": min_valid_loss,
                        "block_index": BLOCK_INDEX_DEIT,
                        "epoch": epoch,
                        "stage": "train",
                    }
                )

                # if epoch % 100 == 0:
                #     with open(
                #         os.path.join(wandb.run.dir, f"anomaly_maps_epoch_{epoch}.json"),
                #         "w",
                #         encoding="utf-8",
                #     ) as f:
                #         json.dump(anomaly_maps, f, ensure_ascii=False)

                if not continue_learning:
                    break

        if self.enable_wandb:
            validator = ValidatorNF(
                nf_model=[nf_model],
                feature_extractor=self.encoder,
                dataloader=test_loader,
                weights_object=best_weights,
                props={
                    "fp_thres": 0.3,
                    "dataset": hyper_param_dict["dataset"],
                    "dataclass": hyper_param_dict["dataclass"],
                },
            )
            validator.calc_all_metrics(new_wandb_run=False)
            wandb.finish()

    def train_with_resnet(
        self,
        train_loader: DataLoader,
        valid_loader: DataLoader,
        test_loader: GeneralDataLoader,
        hyper_param_dict: HyperParameterConfig,
    ):
        """function to train nf_model with resnet50 encoder
        Args:
            train_loader: DataLoader to deliver training data
            valid_loader: DataLoader to deliver validation data
            test_loader: GeneralDataloader to deliver test data for model evaluation
            hyper_param_dict: typed dictionary, which contains the hyperparameter configuration
        """

        nf_list: list[NormalizingFlow] = []

        # create nf model for last three blocks of resnet
        for i in range(1, 4):
            nf_list.append(
                NormalizingFlow(
                    num_channels=self.encoder.res_net.in_channels[i],
                    img_size=self.encoder.img_size,
                    num_patches=int(
                        (self.encoder.img_size / self.encoder.res_net.scales[i]) ** 2
                    ),
                    hidden_ratio=self.hidden_ratio,
                    flow_steps=self.flow_steps,
                )
            )
            nf_list[i - 1].to(self.device)

        self.encoder.to(self.device)

        self.init_training(hyper_param_dict=hyper_param_dict, nf_model=nf_list[0])

        parameter_list = []

        for nf in nf_list:
            parameter_list += list(nf.parameters())

        optimizer = torch.optim.Adam(
            parameter_list + list(self.encoder.parameters()),
            lr=hyper_param_dict["learning_rate"],
            weight_decay=hyper_param_dict["weight_decay"],
        )

        min_valid_loss = np.inf
        best_weights = []
        not_improved = 0

        for epoch in range(hyper_param_dict["epochs"]):
            train_loss = 0.0
            valid_loss = 0.0

            with tqdm(train_loader, unit="batch") as tepoch:
                for images in tepoch:
                    images = images.to(self.device)
                    loss_nf = 0

                    feature_maps, _ = self.encoder(images, separate_layer=True)
                    for i, model in enumerate(nf_list):
                        model.train()
                        feature_map = feature_maps[i + 1]
                        result = model(feature_map)

                        loss_nf += result.loss

                    optimizer.zero_grad()
                    loss_nf.backward()
                    optimizer.step()

                    avg_loss = loss_nf.item() / len(nf_list)

                    tepoch.set_postfix(loss=avg_loss)

                    train_loss += avg_loss * images.size(0)

                    # anomaly_maps = result.anomaly_score_map.detach().tolist()

            for images in valid_loader:
                images = images.to(self.device)
                loss_eval = 0

                feature_maps, _ = self.encoder(images, separate_layer=True)
                for i, model in enumerate(nf_list):
                    model.eval()
                    feature_map = feature_maps[i + 1]
                    result = model(feature_map)

                    loss_eval += result.loss

                valid_loss += (loss_eval.item() * images.size(0)) / len(nf_list)

            train_loss /= len(train_loader.dataset)
            valid_loss /= len(valid_loader.dataset)

            print(
                f"Epoch: {epoch + 1} \tNF Loss: {train_loss} \tValidation Loss: {valid_loss}"
            )

            if self.enable_wandb:
                (
                    min_valid_loss,
                    not_improved,
                    continue_learning,
                    best_weights,
                ) = early_stopping(
                    save_suffix=self.save_prefix,
                    valid_loss=valid_loss,
                    min_valid_loss=min_valid_loss,
                    epoch=epoch,
                    best_weights=best_weights,
                    not_improved=not_improved,
                    patience=hyper_param_dict["patience"],
                    models=nf_list,
                )

                wandb.log(
                    {
                        "nf_loss": train_loss,
                        "valid_loss": min_valid_loss,
                        "epoch": epoch,
                        "stage": "train",
                        "block_index": [1, 2, 3],
                    }
                )

                if not continue_learning:
                    break

        if self.enable_wandb:
            validator = ValidatorNF(
                nf_model=nf_list,
                feature_extractor=self.encoder,
                dataloader=test_loader,
                weights_object=best_weights,
                props={
                    "fp_thres": 0.3,
                    "dataset": hyper_param_dict["dataset"],
                    "dataclass": hyper_param_dict["dataclass"],
                },
            )
            validator.calc_all_metrics(new_wandb_run=False)
            wandb.finish()
