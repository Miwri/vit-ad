from dataclasses import dataclass

import torch
from torch import Tensor, nn

from src.util.HelperFunctions import init_weights

from .CnnAutoEncoder import DecoderVanillaCNN, EncoderVanillaCNN, VanillaAutoEncoder

bias_fill = 0.001
# kernel = 3
stride_conv = 1
stride_pool = 2
padding = 1


@dataclass
class VaeLatentSpace:
    """Data class to type Variational AutoEncoder output"""

    mu: Tensor
    log_var: Tensor

    def size(self):
        """Returning only the size of the mean tensor to get torch_summary to work"""
        return self.mu.size()


@dataclass
class VariationalAutoEncoderOutput:
    latent_space: VaeLatentSpace
    reconstruction: Tensor


class VariationalEncoder(EncoderVanillaCNN):
    def __init__(self, img_size: int, flatten_dim: int, conv: bool = True) -> None:
        super().__init__(img_size=img_size)

        self.encoder_lin = nn.Sequential(
            nn.Linear(in_features=flatten_dim, out_features=1024),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=1024, out_features=2 * 768),
        )

        self.encoder_lin.apply(init_weights)

    def forward(self, x) -> VaeLatentSpace:
        x = self.encoder(x)
        x = self.flatten(x)
        x = self.encoder_lin(x)
        mu = x[:, :768]
        # same activation function as sigma for mdn
        log_var = nn.ELU()(x[:, 768:]) + 1 + 1e-15
        return VaeLatentSpace(mu=mu, log_var=log_var)


def kl_loss(mu, log_var):
    # devide by batch size to get average loss per item additionally devide by size of latent space to get average loss per feature
    return (
        (0.5 * torch.sum(-log_var - 1 + torch.square(mu) + torch.exp(log_var)))
        / log_var.shape[0]
    ) / log_var.shape[1]


class VariationalAutoEncoder(VanillaAutoEncoder):
    def __init__(self, img_size: int) -> None:
        super().__init__(img_size)

        self.encoder = VariationalEncoder(
            img_size=img_size, flatten_dim=self.flatten_size
        )
        self.decoder = DecoderVanillaCNN(
            z_space=self.z_space,
            first_feature_map_size=self.feature_map_size,
        )

    def forward(self, x) -> VariationalAutoEncoderOutput:
        vae_latent_vec = self.encoder(x)

        # log_var += 1e-4  # add a very small number to avoid NaN because of ln(0)
        sigma = torch.exp(0.5 * vae_latent_vec.log_var)
        # random numbers from gaussian with mean 0 and std 1 and shape from sigma tensor
        epsilon = torch.randn_like(sigma)
        # sampling z from mu and sigma
        z = vae_latent_vec.mu + epsilon * sigma
        x_recon = self.decoder(z)

        return VariationalAutoEncoderOutput(
            latent_space=vae_latent_vec, reconstruction=x_recon
        )
