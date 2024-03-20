"""Module to provide pretrained CNN Encoder like EfficientNet and ResNet"""

from torch import Tensor, hub, nn
from torch.hub import load_state_dict_from_url

from src.classes.resnet.ResNetModel import ResNet
from src.util.HelperFunctions import init_weights

bias_fill = 0.001
kernel = 3
stride_conv = 1
stride_pool = 2
padding = 1


class EncoderVanillaCNN(nn.Module):
    """Shallow Encoder which mirrors the structure from the shallow decoder. Not used since pre-trained ResNet is used as CNN Encoder."""

    def __init__(self, img_size: int) -> None:
        super().__init__()

        self.architecture = "cnn_encoder"

        self.conv1 = nn.Conv2d(
            in_channels=3,
            out_channels=48,
            kernel_size=kernel,
            stride=stride_conv,
            padding=padding,
        )

        self.conv2 = nn.Conv2d(
            in_channels=48,
            out_channels=96,
            kernel_size=kernel,
            stride=stride_conv,
            padding=padding,
        )

        self.conv3 = nn.Conv2d(
            in_channels=96,
            out_channels=192,
            kernel_size=kernel,
            stride=stride_conv,
            padding=padding,
        )

        self.conv4 = nn.Conv2d(
            in_channels=192,
            out_channels=384,
            kernel_size=kernel,
            stride=stride_conv,
            padding=padding,
        )

        self.conv5 = nn.Conv2d(
            in_channels=384,
            out_channels=768,
            kernel_size=kernel,
            stride=stride_conv,
            padding=padding,
        )

        self.encoder = nn.Sequential(
            self.conv1,
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=kernel, stride=stride_pool, padding=padding),
            self.conv2,
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=kernel, stride=stride_pool, padding=padding),
            self.conv3,
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=kernel, stride=stride_pool, padding=padding),
            self.conv4,
            nn.BatchNorm2d(384),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=kernel, stride=stride_pool, padding=padding),
            self.conv5,
            nn.BatchNorm2d(768),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=kernel, stride=stride_pool, padding=padding),
        )

        self.encoder.apply(init_weights)

        # self.flatten = nn.Flatten(start_dim=1)

        # self.encoder_lin = nn.Sequential(
        #     nn.Linear(in_features=flatten_dim, out_features=1024),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(in_features=1024, out_features=z_space),
        # )

        # self.encoder_lin.apply(init_weights)

    def forward(self, x) -> nn.Sequential:
        x = self.encoder(x)
        # x = self.flatten(x)
        # x = self.encoder_lin(x)
        return x


class EfficientNetEncoder(nn.Module):
    """Encoder which capsules pre-trained EfficientNet B4, currently not in use"""

    def __init__(self, img_size: int) -> None:
        super().__init__()

        self.efficient_net = hub.load(
            "NVIDIA/DeepLearningExamples:torchhub",
            "nvidia_efficientnet_widese_b4",
            pretrained=True,
        )

        self.architecture = "cnn_encoder"

        for param in self.efficient_net.parameters():
            param.requires_grad = False

    def forward(self, x) -> nn.Sequential:
        """forward pass function to process input and return feature map of last layer"""
        output = self.efficient_net.extract_features(x)
        return output["features"]  # return feature map


class ResNetEncoder(nn.Module):
    """Encoder which capsules pre-trained ResNet50"""

    def __init__(self, img_size: int) -> None:
        super().__init__()

        self.res_net = ResNet()

        self.architecture = "cnn_encoder"

        self.img_size = img_size

        self.res_net.load_state_dict(
            load_state_dict_from_url(
                "https://download.pytorch.org/models/resnet50-11ad3fa6.pth", progress=True
            )
        )

        for param in self.res_net.parameters():
            param.requires_grad = False

        # add trainable layer norm as proposed here https://github.com/gathierry/FastFlow/
        self.norms = nn.ModuleList()
        for in_channels, scale in zip(self.res_net.in_channels, self.res_net.scales):
            self.norms.append(
                nn.LayerNorm(
                    [in_channels, int(img_size / scale), int(img_size / scale)],
                    elementwise_affine=True,
                )
            )

    def forward(self, x: Tensor, separate_layer: bool = False) -> nn.Sequential:
        """forward pass function to process input and return feature map of last layer or of all layer
        Args:
            x: Tensor, input tensor to process
            separate_layer: bool, whether to return feature map of last layer or of each layer separately
        """
        features, indices = self.res_net(x, separate_layer=separate_layer)
        if separate_layer:
            features = [self.norms[i](feature) for i, feature in enumerate(features)]
        return features, indices  # return feature map
