"""
Implementation of a Gaussian Mixture Density Network to estimate the density of normal latent space features.
A distribution for the feature space of each patch is calculated separately.
Args:
    input_dim:      dimension of input vector, input vector should be the patch embedding of an image and therefore have the dimension [number_patches, size_embedding_space]
    output_dim:     length of output vectors, normally matches the input dimension
    num_gaussians:  number of mixture coefficients, also sets the number of learned Gaussian distributions per
                    feature. The distributions are weighted by their mixture coefficients. These weights are learned during training.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor, nn

from src.util.HelperFunctions import init_weights

# https://deep-and-shallow.com/2021/03/20/mixture-density-networks-probabilistic-regression-for-uncertainty-estimation/
# add L1 or L2 norm to weights?
# add different bias to mu depending on gaussian


@dataclass
class MdnReturn:
    """Dataclass to type the return of the mixture density network"""

    pi: Tensor
    sigma: Tensor
    mu: Tensor


def log_gaussian_density(sigma: Tensor, mu: Tensor, x: Tensor):
    """Calculates the logarithm of the gaussian density function to avoid numerical underflows
    args:
        sigma: Tensor of shape [batch_size, number_img_patches, size_patch_embedding, num_gaussians]
        mu: Tensor of shape [batch_size, number_img_patches, size_patch_embedding, num_gaussians]
        x: Tensor of shape [batch_size, number_img_patches, size_patch_embedding, num_gaussians]
    """
    return (
        -torch.log(sigma)
        - 0.5 * math.log(2 * math.pi)
        - 0.5 * torch.pow((x - mu) / sigma, 2)
    )


def log_likelihood(x: Tensor, pi: Tensor, sigma: Tensor, mu: Tensor) -> Tensor:
    """Calculates the log likelihood for each feature in each patch to belong to the learned distributions.
    args:
        x: Tensor of shape [batch_size, number_img_patches, size_patch_embedding]
        pi: Tensor of shape [batch_size, number_img_patches, num_gaussians]
        sigma: Tensor of shape [batch_size, number_img_patches, size_patch_embedding, num_gaussians]
        mu: Tensor of shape [batch_size, number_img_patches, size_patch_embedding, num_gaussians]
    return: Tensor of shape [batch_size, number_img_patches, size_patch_embedding]
    """
    x = x.unsqueeze(-1).expand_as(mu)
    # calculate weights for each gaussian and expand to size of feature vector. Each feature vector has the same weights for the individual gaussians
    # tau is a hyperparameter to be trained. It regulates the temperature of the gumble softmax. tau -> 0: sample vectors approach the one-hot vectors, tau -> inf: sample vectors become uniform
    # gumble softmax is a workaround to estimate the parameter of a discrete distribution. It mimics a continuous function, for which it is easier to estimate the parameters
    softmax_pi = nn.functional.gumbel_softmax(logits=pi, tau=1, dim=-1)

    log_pi = torch.log(torch.add(softmax_pi, 1e-15))

    # sum_pi = torch.sum(torch.exp(log_pi))  # = 1

    log_pi = log_pi.unsqueeze(2).repeat(1, 1, sigma.size()[2], 1)

    log_gaus_dens_per_feature = log_gaussian_density(sigma=sigma, mu=mu, x=x)

    return torch.logsumexp(log_pi + log_gaus_dens_per_feature, dim=-1)


def get_probability_map(x: Tensor, pi: Tensor, sigma: Tensor, mu: Tensor) -> Tensor:
    """Calculates the probability for each patch to belong to the learned distributions by returning the mean of the likelihood of each feature, normalized in [0,1].
    args:
        x: Tensor of shape [batch_size, number_img_patches, size_patch_embedding]
        pi: Tensor of shape [batch_size, number_img_patches, num_gaussians]
        sigma: Tensor of shape [batch_size, number_img_patches, size_patch_embedding, num_gaussians]
        mu: Tensor of shape [batch_size, number_img_patches, size_patch_embedding, num_gaussians]
    """
    log_likelihood_map = log_likelihood(
        x=x, pi=pi, sigma=sigma, mu=mu
    )  # [batch_size, num_img_patches, size_patch_embedding]
    likelihood_per_feature = torch.mean(
        log_likelihood_map.detach(), dim=2
    )  # [batch_size, num_img_patches]
    # TODO normalization is per batch, should be per feature?
    likelihood_per_feature -= torch.max(
        likelihood_per_feature
    )  # subtract max likelihood to get values in [-inf,0]
    probability_map = torch.exp(
        likelihood_per_feature
    )  # normalize values to be in range [0,1]

    return probability_map


def mdn_loss(x: Tensor, pi: Tensor, sigma: Tensor, mu: Tensor):
    """Computes the loss function for the Mixture Density Network as negative log likelihood"""
    return torch.mean(-log_likelihood(x=x, pi=pi, sigma=sigma, mu=mu))


class GaussianMixtureDensityNetwork(nn.Module):
    """
    Network which learns a defined number of gaussian distributions to represent a feature space.
    Computes des likelihood of each feature to belong to one of the gaussians.

    Args:
        cluster_centers: Tensor of shape [batch_size, input_dim], tensor of precomputed cluster centers on train set to init mu bias with it
        input_dim: int, dimension of input vector, in case of vit number of features per image patch
        output_dim: int, dimension of output vector, normally same es input_dim
        num_gaussians: int, number of gaussian distributions to be used for calculation
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        num_gaussians: int,
        cluster_centers: Tensor | None = None,
    ) -> None:
        super().__init__()

        self.elu = nn.ELU()

        self.pi = nn.Linear(in_features=input_dim, out_features=num_gaussians)
        nn.init.xavier_normal_(self.pi.weight)

        self.sigma = nn.Linear(
            in_features=input_dim, out_features=input_dim * num_gaussians
        )

        nn.init.xavier_normal_(self.sigma.weight)

        self.mu = nn.Linear(in_features=input_dim, out_features=input_dim * num_gaussians)

        if cluster_centers is not None:
            # set bias to cluster centers
            print("setting mu bias to cluster centers")
            for i, bias in enumerate(cluster_centers):
                nn.init.constant_(self.mu.bias[i], bias)
            nn.init.xavier_normal_(self.mu.weight)
        else:
            init_weights(self.mu.apply(init_weights))

        self.out_dim = output_dim
        self.num_gaussians = num_gaussians

    def forward(self, x: Tensor) -> MdnReturn:
        """function to calculate pi, sigma and mu of the gaussians
        Args:
            x: Tensor of size [batch_size, number_img_patches, size_patch_embedding]
        Output:
            pi:             Vector of mixture coefficients per distribution. length: number of coefficients
            sigma:          Vector of learned standard deviation per feature and per coefficient.
                            length: output dimension * number of coefficients, dim [batch_size, number_img_patches, size_patch_embedding, num_gaussians]
            mu:             Vector of learned mean values per feature and per coefficient.
                            length: output dimension * number of coefficients, dim [batch_size, number_img_patches, size_patch_embedding, num_gaussians]
        """

        # Mixing coefficient which are the weight parameters for each gaussian density
        pi = self.pi(x)
        # ELU activation with positive (+1), non null (+1e-15) output
        sigma = (self.elu(self.sigma(x)) + 1 + 1e-15).view(
            x.size(0), x.size(1), self.out_dim, self.num_gaussians
        )
        mu = self.mu(x).view(x.size(0), x.size(1), self.out_dim, self.num_gaussians)

        return MdnReturn(pi=pi, sigma=sigma, mu=mu)
