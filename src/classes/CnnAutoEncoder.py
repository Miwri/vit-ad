import math
from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torchmetrics import StructuralSimilarityIndexMeasure

from src.classes.CnnDecoder import DecoderCNNLinEnd, DecoderVanillaCNN
from src.classes.CnnEncoder import EfficientNetEncoder, EncoderVanillaCNN, ResNetEncoder
from src.classes.resnet.ReverseResNet import ReverseResNet

kernel = 3
stride_conv = 1
stride_pool = 2
padding = 1


@dataclass
class AutoEncoderOutput:
    """Dataclass to type standard AutoEncoder Output. Patch embedding only important for model with transformer as encoder."""

    latent_space: Tensor
    reconstruction: Tensor
    patch_embedding: Tensor = None


class VanillaAutoEncoder(nn.Module):
    """AutoEncoder class with a 5 stage CNN as Encoder and a 5 stage CNN as Decoder. Class is used as base class for extensions. Implements MSE and SSIM loss functions.
    Args:
        img_size: int, size of input images.
        red_mse: str, reduction which is used to get one mse value. Possible values are 'none', 'mean', 'sum'. default: 'mean'
        red_ssim: str, reduction which is used to get one ssim value. Possible values are 'none', 'elementwise_mean', 'sum'. default: 'elementwise_mean'
        latent_space: int, size of latent space. Has to be given if encoder output is a vector. Only important for decoder. default: 0
    """

    def __init__(
        self,
        img_size: int,
        red_mse="none",
        red_ssim="elementwise_mean",
        size_latent_space=0,
    ) -> None:
        super().__init__()

        self.img_size = img_size
        self.architecture = "convolution"

        self.ssim = StructuralSimilarityIndexMeasure(data_range=1.0, reduction=red_ssim)
        self.mse = nn.MSELoss(reduction=red_mse)

        self.z_space = size_latent_space
        self.feature_map_size = math.ceil(img_size / (2**5))

        self.encoder = EncoderVanillaCNN(img_size=img_size)
        self.decoder = DecoderVanillaCNN(
            z_space=self.z_space,
            first_feature_map_size=self.feature_map_size,
        )

    def forward(self, x) -> AutoEncoderOutput:
        """Forward pass which creates an embedding space and reconstructs the entire image out of it.
        Returns the embedding space and the reconstruction."""
        z = self.encoder(x)
        x_recon = self.decoder(z)

        return AutoEncoderOutput(latent_space=z, reconstruction=x_recon)

    def MSELoss(self, output: Tensor, x: Tensor) -> nn.MSELoss:
        """Measures the mean squared error between reconstruction and original image.
        args:
            output: Tensor, reconstructed image, mostly of shape [BATCH_SIZE, 3, IMG_SIZE, IMG_SIZE]
            x: Tensor, input image, mostly of shape [BATCH_SIZE, 3, IMG_SIZE, IMG_SIZE]
        """
        return self.mse(output, x)

    def SSIMLoss(self, output: Tensor, x: Tensor) -> StructuralSimilarityIndexMeasure:
        """Measures the structural similarity index between reconstruction and original image.
        Is used to capture features which are likely to be ignored by the MSE.
        args:
            output: Tensor, reconstructed image, mostly of shape [BATCH_SIZE, 3, IMG_SIZE, IMG_SIZE]
            x: Tensor, input image, mostly of shape [BATCH_SIZE, 3, IMG_SIZE, IMG_SIZE]
        """
        return (self.ssim(output, x) * (-1)) + 1


class AutoEncoderVanillaCNNLinEnd(VanillaAutoEncoder):
    """AutoEncoder class with a 5 stage CNN as Encoder and a 5 stage CNN as Decoder. Decoder uses a linear CNN layer as last layer instead of tanh activation function."""

    def __init__(
        self, img_size: int, red_mse="none", red_ssim="elementwise_mean"
    ) -> None:
        super().__init__(img_size=img_size, red_mse=red_mse, red_ssim=red_ssim)

        self.decoder = DecoderCNNLinEnd(
            z_space=self.z_space,
            first_feature_map_size=self.feature_map_size,
        )


class AutoEncoderEfficientNet(VanillaAutoEncoder):
    """AutoEncoder class with a EfficientNets as Encoder and a reverse ResNet as Decoder. Currently not in use."""

    def __init__(
        self, img_size: int, red_mse="none", red_ssim="elementwise_mean"
    ) -> None:
        super().__init__(img_size, red_mse, red_ssim)

        self.encoder = EfficientNetEncoder(img_size=img_size)


class AutoEncoderResNetSmallDecoder(VanillaAutoEncoder):
    """AutoEncoder class with a ResNet50 as encoder and a small CNN as Decoder"""

    def __init__(
        self,
        img_size: int,
        red_mse="none",
        red_ssim="elementwise_mean",
    ) -> None:
        super().__init__(img_size, red_mse, red_ssim, size_latent_space=2048)

        self.encoder = ResNetEncoder(img_size=img_size)

    def forward(self, x):
        z, _ = self.encoder(x)

        z = torch.reshape(z, (-1, z.shape[1]))

        x_recon = self.decoder(z)

        return AutoEncoderOutput(latent_space=z, reconstruction=x_recon)


class AutoEncoderResNet(VanillaAutoEncoder):
    """AutoEncoder class with a ResNet as Encoder and a reverse ResNet as Decoder."""

    def __init__(
        self, img_size: int, red_mse="none", red_ssim="elementwise_mean"
    ) -> None:
        super().__init__(
            img_size=img_size,
            red_mse=red_mse,
            red_ssim=red_ssim,
        )

        self.encoder = ResNetEncoder(img_size=img_size)
        self.decoder = ReverseResNet()

    def forward(self, x):
        z, indices = self.encoder(x)

        x_recon = self.decoder(z, indices)

        return AutoEncoderOutput(latent_space=z, reconstruction=x_recon)
