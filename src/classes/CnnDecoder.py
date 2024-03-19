"""Module to provide custom CNN Decoder"""


from torch import nn

from src.classes.resnet.ReverseResNet import ReverseResNet
from src.util.HelperFunctions import init_weights

bias_fill = 0.001
kernel = 3
stride_conv = 1
stride_pool = 2
padding = 1


class DecoderVanillaCNN(nn.Module):
    """Decoder which implements a 5 stage reverse CNN to reconstruct an image from either embedded feature maps or a flattened latent space.
    Args:
        z_space: int, size of flattened latent space. When set 2 linear layer are applied to get feature maps to desired size. default: 0
        first_feature_map_size: int, size of first feature map. Should be derived from image size, normally IMG_SIZE/ 2^5. Must be set when z_space is set. default: 0
    """

    def __init__(self, z_space: int = 0, first_feature_map_size: int = 0) -> None:
        super().__init__()

        self.use_linear = False

        if z_space != 0:
            self.use_linear = True
            unflatten_size = 768 * first_feature_map_size * first_feature_map_size
            hidden_size = 2 * z_space

            self.decoder_lin = nn.Sequential(
                nn.Linear(in_features=z_space, out_features=hidden_size),
                nn.ReLU(inplace=True),
                nn.Linear(in_features=hidden_size, out_features=unflatten_size),
                nn.ReLU(inplace=True),
            )

            self.unflatten = nn.Unflatten(
                dim=1,
                unflattened_size=(768, first_feature_map_size, first_feature_map_size),
            )

            self.decoder_lin.apply(init_weights)

        self.recon_conv1 = nn.ConvTranspose2d(
            in_channels=768,
            out_channels=384,
            kernel_size=kernel,
            stride=stride_pool,
            padding=padding,
            output_padding=padding,
        )

        self.recon_conv2 = nn.ConvTranspose2d(
            in_channels=384,
            out_channels=192,
            kernel_size=kernel,
            stride=stride_pool,
            padding=padding,
            output_padding=padding,
        )

        self.recon_conv3 = nn.ConvTranspose2d(
            in_channels=192,
            out_channels=96,
            kernel_size=kernel,
            stride=stride_pool,
            padding=padding,
            output_padding=padding,
        )

        self.recon_conv4 = nn.ConvTranspose2d(
            in_channels=96,
            out_channels=48,
            kernel_size=kernel,
            stride=stride_pool,
            padding=padding,
            output_padding=padding,
        )

        self.recon_conv5 = nn.ConvTranspose2d(
            in_channels=48,
            out_channels=3,
            kernel_size=kernel,
            stride=stride_pool,
            padding=padding,
            output_padding=padding,
        )

        self.decoder_cnn = nn.Sequential(
            self.recon_conv1,
            nn.BatchNorm2d(384),
            nn.ReLU(inplace=True),
            self.recon_conv2,
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True),
            self.recon_conv3,
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True),
            self.recon_conv4,
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True),
            self.recon_conv5,
            nn.BatchNorm2d(3),
            nn.Tanh(),
        )

        self.decoder_cnn.apply(init_weights)

    def forward(self, x) -> nn.Sequential:
        """Forward pass to reconstruct embedded image. Uses linear layer when z_space argument of class is set to a non zero value."""
        if self.use_linear:
            x = self.decoder_lin(x)
            x = self.unflatten(x)
        return self.decoder_cnn(x)


class DecoderCNNLinEnd(DecoderVanillaCNN):
    """Decoder which uses a linear CNN as last layer instead of tanh activation function. Currently not in use."""

    def __init__(self, z_space: int, first_feature_map_size: int) -> None:
        super().__init__(
            z_space=z_space,
            first_feature_map_size=first_feature_map_size,
        )

        # linear instead of Sigmoid
        self.linear_end = nn.Conv2d(
            in_channels=3,
            out_channels=3,
            kernel_size=kernel,
            stride=stride_conv,
            padding=padding,
        )

        self.decoder_cnn = nn.Sequential(
            self.recon_conv1,
            nn.BatchNorm2d(384),
            nn.ReLU(inplace=True),
            self.recon_conv2,
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True),
            self.recon_conv3,
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True),
            self.recon_conv4,
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True),
            self.recon_conv5,
            nn.BatchNorm2d(3),
            nn.ReLU(inplace=True),
            self.linear_end,
        )


class DecoderResNetVariableEmbeddingSize(ReverseResNet):
    """Decoder class which mirrors the architecture of resnet except that it adds one linear layer to get from variable embedding to 2048 features.
    Additionally unpooling is replaced by upsampling
    Args:
        embedding_size: integer which is used as input for first linear layer.
            768 for ViT, DeiT, EsViT.
            448 for EfficientFormer
            384 for NesT"""

    def __init__(self, embedding_size: int) -> None:
        super().__init__()

        hidden_size = 2 * embedding_size

        self.fc1 = nn.Sequential(
            nn.Linear(in_features=embedding_size, out_features=hidden_size),
            nn.ReLU(inplace=True),
        )
        # TODO maybe better 2048 * 7 * 7 instead of upsampling? No average pool in transformers for embedding space creation
        self.fc2 = nn.Sequential(
            nn.Linear(in_features=hidden_size, out_features=2048 * 1 * 1),
            nn.ReLU(inplace=True),
        )
        self.unflatten = nn.Unflatten(dim=1, unflattened_size=(2048, 1, 1))
        self.upsample2 = nn.Upsample(size=112, mode="nearest")

    def forward(self, x, indices=None):
        """Forward path which reconstructs image from embedding vector of variable sizes."""
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.unflatten(x)

        x = self._forward_cnns_only(x)
        x = self.upsample2(x)
        x = self.de_conv1(x)
        x = self.bn1(x)
        x = self.tanh(x)

        return x
