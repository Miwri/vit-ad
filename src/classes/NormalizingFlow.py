"""Module which provides the implementation of a 2D normalizing flow model"""

import math
from dataclasses import dataclass

import FrEIA.framework as Ff
import FrEIA.modules as Fm
import torch
from torch import nn

# _GCONST_ = -0.9189385332046727  # ln(sqrt(2*pi))


@dataclass
class NormalizingFlowReturn:
    """Dataclass to type return of normalizing flow forward function"""

    loss: torch.Tensor
    anomaly_score_map: torch.Tensor


class NormalizingFlow(nn.Module):
    """Normalizing flow model as proposed by Yu et al. for FastFlow
    Args:
        num_channels: int, number of channels the input feature map has. Is used to define channel size of conv layers for splitted feature maps.
    """

    def __init__(
        self,
        num_channels: int,
        img_size: int,
        num_patches: int,
        hidden_ratio: float = 1.0,
        flow_steps: int = 8,
    ) -> None:
        super().__init__()

        self.norms = nn.ModuleList()
        self.img_size = img_size

        embedding_width_height = int(math.sqrt(num_patches))

        self.layer_norm = nn.LayerNorm(
            (num_channels, embedding_width_height, embedding_width_height)
        )

        self.num_channels = num_channels

        self.flow_type = "AllInOneBlock"

        self.fast_flow_decoder = self.fast_flow_steps(
            input_dims=[
                num_channels,
                embedding_width_height,
                embedding_width_height,
            ],
            hidden_ratio=hidden_ratio,
            flow_steps=flow_steps,
        )

    def subnet_conv_fun(self, kernel: int, hidden_ratio: float):
        """Convolutions which are used in a fast flow step."""

        def subnet_conv(in_channels: int, out_channels: int):
            hidden_channels = int(in_channels * hidden_ratio)
            return nn.Sequential(
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=hidden_channels,
                    padding="same",
                    kernel_size=kernel,
                ),
                nn.ReLU(inplace=False),
                nn.Conv2d(
                    in_channels=hidden_channels,
                    out_channels=out_channels,
                    padding="same",
                    kernel_size=kernel,
                ),
            )

        return subnet_conv

    def fast_flow_steps(
        self, input_dims: list[int], hidden_ratio: float, flow_steps: int
    ):
        """Implementation of several fast flow steps with the help of bilinear invertible functions from freia framework.
        Convolutions are stacked alternating between 3x3 and 1x1
        Args:
            input_dims: list[int]
            hidden_ratio: float, ratio for the hidden dimension of the conv block
            flow_steps: int, number of flow steps
        """

        steps = Ff.SequenceINN(*input_dims)
        for i in range(flow_steps):
            if i % 2 == 1:
                kernel = 1
            else:
                kernel = 3

            # steps.append(Fm.ActNorm)
            # steps.append(Fm.PermuteRandom)
            steps.append(
                module_class=Fm.AllInOneBlock,  # maybe works better because has global affine transformation (ActNorm)
                # module_class=Fm.RNVPCouplingBlock,  # this makes what the paper describes
                # module_class=Fm.GLOWCouplingBlock,  # this makes what the paper describes but optimized
                permute_soft=False,
                subnet_constructor=self.subnet_conv_fun(
                    kernel=kernel, hidden_ratio=hidden_ratio
                ),
                # clamp=2.0,
                affine_clamping=2.0,  # as seen here https://github.com/gathierry/FastFlow/blob/2cf1f2f4c562a7f13cfb1959e3afe5df2f2d2565/fastflow.py#L23
            )

        return steps

    def forward(self, x: torch.Tensor):
        """Function for forward pass, alternating stacks 3x3 and 1x1 layer
        Args:
            x: Tensor, input tensor of size [batch_size, num_channels, feature_map_len, feature_map_width], should have the number of channels assigned to the model at creation
        Return: computed loss and generated anomaly maps with the same size as image input
        """
        # if cnn:
        # x = self.layer_norm(x)

        output, log_jac_det = self.fast_flow_decoder(x)

        # negative log likelihood with standard normal prior
        loss = torch.mean(
            (0.5 * torch.sum(output**2, dim=(1, 2, 3), keepdim=False)) - log_jac_det
        )

        log_likelihood = -0.5 * torch.mean(
            output**2, dim=1, keepdim=True
        )  # log likelihood per image patch [batch_size, 1, patch_emb_width, patch_emb_height]
        prob = torch.exp(log_likelihood)  # normalize values to be in range [0,1]
        anomaly_score_map = nn.functional.interpolate(
            input=(prob * (-1)) + 1,
            size=[self.img_size, self.img_size],
            mode="bilinear",
            align_corners=False,
        )

        return NormalizingFlowReturn(loss=loss, anomaly_score_map=anomaly_score_map)
