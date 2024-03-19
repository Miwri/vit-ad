"""Module to provide functions for trainings pipeline"""

from __future__ import annotations

import os

import numpy as np
from sklearn.cluster import KMeans
from torch import cuda, nn, save
from torch.utils.data import DataLoader
from tqdm import tqdm

import wandb
from src.classes.CnnEncoder import ResNetEncoder
from src.classes.transformer.TransformerEncoder import TransformerEncoder


def init_cluster_centers(
    dataloader: DataLoader, encoder: TransformerEncoder | ResNetEncoder, num_clusters: int
):
    """Function to precompute cluster centers for bias initialization of Mixture Density Network
    args:
        dataloader: DataLoader which delivers train images to compute embedding
        auto_encoder: transformer model which delivers as output a patch embedding of input images, required outputsize: [batch_size, number_patches, size_patch_embedding]
        num_clusters: int, number of clusters to find, should be the same as the used gaussians
    """

    device = "cuda:0" if cuda.is_available() else "cpu"

    if isinstance(encoder, ResNetEncoder):
        embeddings = tuple(([], [], []))
    else:
        embeddings = []

    encoder.to(device)

    print(f"Precomputing centers on {device}")

    with tqdm(dataloader, unit="batch") as tepoch:
        for images in tepoch:
            images = images.to(device)

            if isinstance(encoder, ResNetEncoder):
                feature_maps, _ = encoder(images, separate_layer=True)
                for i, feature_map in enumerate(feature_maps):
                    embeddings[i].append(feature_map.detach().cpu().numpy())
            else:
                output = encoder(images)
                embeddings.append(output.patch_embedding.detach().cpu().numpy())

    if isinstance(encoder, ResNetEncoder):
        embeddings_numpy: list[np.array] = []
        results = []
        for i, embedding in enumerate(embeddings):
            embeddings_numpy.append(np.concatenate(embedding))
            num_patches = embeddings_numpy[i].shape[2] ** 2
            embeddings_numpy[i] = (
                embeddings_numpy[i]
                .transpose(0, 2, 3, 1)
                .reshape(
                    (len(dataloader.dataset) * num_patches, -1),
                )
            )
            results.append(
                KMeans(n_clusters=num_clusters, random_state=42, n_init="auto")
                .fit(embeddings_numpy[i])
                .cluster_centers_.ravel()
            )
        return results

    embeddings_numpy = np.concatenate(embeddings)

    embeddings_numpy = np.reshape(
        embeddings_numpy, (len(dataloader.dataset) * embeddings_numpy.shape[1], -1)
    )

    return (
        KMeans(n_clusters=num_clusters, random_state=42, n_init="auto")
        .fit(embeddings_numpy)
        .cluster_centers_.ravel()
    )


def early_stopping(
    valid_loss: float,
    min_valid_loss: float,
    epoch: int,
    not_improved: int,
    patience: int,
    models: list[nn.Module],
    best_weights: list[dict],
    save_suffix: str,
):
    """Function to realize early stopping during the training loop. Saves model weights to weights & biases when validation loss has improved. Stopps training when validation loss did not improve {patience} times
    Args:
        valid_loss: float, current validation loss
        min_valid_loss: float, smallest achieved validation loss in training run
        epoch: int, current epoch
        not_improved: int, times loss did not improve
        patience: int, maximum times loss can not improve until training is stopped
        models: list[Module], list of trained models to save
        best_weights: list][tuple], list of best model weights
        save_suffix: str, suffix to append to model name
    """
    if valid_loss < min_valid_loss:
        print(
            f"Epoch: {epoch + 1} \tValidation Loss improved from {min_valid_loss} to {valid_loss} \tmodel state saved."
        )

        best_weights = []

        for i, model in enumerate(models):
            best_state_dict = model.state_dict()
            save(
                best_state_dict,
                os.path.join(
                    wandb.run.dir,
                    f"{type(model).__name__}_{i}_{save_suffix}.pth",
                    # f"{type(model).__name__}_{epoch+1}-epochs_{save_suffix}.pth",
                ),
            )
            best_weights.append(best_state_dict)

        new_min_loss = valid_loss
        new_not_improved = 0
    else:
        new_not_improved = not_improved + 1
        new_min_loss = min_valid_loss
        print(
            f"Epoch: {epoch + 1} \tValidation Loss did not improve the {new_not_improved}. time"
        )

    if new_not_improved > patience:
        print(
            f"Epoch: {epoch + 1} \tValidation Loss did not improve {new_not_improved} times. Training stopped."
        )

        return new_min_loss, new_not_improved, False, best_weights

    return new_min_loss, new_not_improved, True, best_weights
