"""Module which contains visual transformer backbones
"""

from __future__ import annotations

from dataclasses import dataclass

import timm
import torch
from torch import nn

from src.classes.transformer.SwinTransformerModule import SwinTransformer


@dataclass
class TransformerEncoderOutput:
    """Dataclass to type Transformer Encoder Output"""

    patch_embedding: torch.Tensor
    latent_space: torch.Tensor | None = None


class TransformerEncoder(nn.Module):
    """Base class for all Transformer encoder."""

    def __init__(self, img_size: int) -> None:
        super().__init__()

        self.img_size = img_size
        self.architecture = "transformer_encoder"

        self.size_patch_embedding = 0
        self.patch_size = 1
        self.num_embedded_patches = 0

    def calc_num_embedded_patches(self):
        """Function to calculate number of embedded patches"""
        return int((self.img_size / self.patch_size) ** 2)

    def forward(self, x: torch.Tensor):
        """Placeholder implementation for forward function"""
        print("Yet not implemented")
        return x


class EncoderNest(TransformerEncoder):
    """EncoderNest
    Encoder class which uses the hierarchical vision transformer NesT proposed by Zhang et al. published in timm library
    Image Size has to be 224 to keep position embedding
    Args:
        img_size: int, width/ height of input image, only quadratic images are possible, used to determine embedding size and number of patches
        requires_grad: bool, default=False whether weights are trained or freezed during training, if True, no pretrained weights are loaded
    """

    def __init__(self, img_size: int, requires_grad: bool = False) -> None:
        super().__init__(img_size=img_size)
        self.nest = timm.models.jx_nest_tiny(pretrained=not requires_grad)

        self.size_patch_embedding = 384
        self.patch_size = 16
        self.num_embedded_patches = self.calc_num_embedded_patches()

        for param in self.nest.parameters():
            param.requires_grad = requires_grad

    def forward(self, x: torch.Tensor, block_index: int = 0) -> TransformerEncoderOutput:
        """Function which implements the forward pass and returns not flattened feature maps created by NesT transformer model.
        Return: Feature map with shape [batch_size, number_embedded_patches, size_feature_embedding]
        """
        patch_embedding = self.nest.forward_features(x).reshape(
            -1,
            self.num_embedded_patches,
            self.size_patch_embedding,
        )
        avg_pool = nn.AdaptiveAvgPool2d((1, patch_embedding.shape[2]))
        z = avg_pool(patch_embedding).reshape((-1, patch_embedding.shape[2]))

        return TransformerEncoderOutput(patch_embedding=patch_embedding, latent_space=z)


class EncoderEfficientFormer(TransformerEncoder):
    """EncoderEfficientFormer
    Encoder class which uses the hierarchical vision transformer EfficientFormer proposed by Li et al. published in timm library
    Image Size has to be 224 to keep position embedding
    Args:
        img_size: int, width/ height of input image, only quadratic images are possible, used to determine embedding size and number of patches
        requires_grad: bool, default=False whether weights are trained or freezed during training, if True, no pretrained weights are loaded
    """

    def __init__(self, img_size: int, requires_grad: bool = False) -> None:
        super().__init__(img_size=img_size)
        self.efficientformer = timm.models.efficientformer_l3(
            pretrained=not requires_grad
        )

        self.size_patch_embedding = 512
        self.patch_size = 32
        self.num_embedded_patches = self.calc_num_embedded_patches()

        for param in self.efficientformer.parameters():
            param.requires_grad = requires_grad

    def forward(self, x: torch.Tensor, block_index: int = 0) -> TransformerEncoderOutput:
        """Function which implements the forward pass and returns features created by EfficientFormer transformer model.
        Return: Features with shape [batch_size, number_embedded_patches, size_feature_embedding]
        """

        # TODO use output of stage3 instead of 4
        patch_embedding = self.efficientformer.forward_features(x)
        avg_pool = nn.AdaptiveAvgPool2d((1, patch_embedding.shape[2]))
        z = avg_pool(patch_embedding).reshape((-1, patch_embedding.shape[2]))

        return TransformerEncoderOutput(patch_embedding=patch_embedding, latent_space=z)


class EncoderDeit(TransformerEncoder):
    """EncoderDeit
    Encoder class that uses the monolithic vision transformer DeiT. DeiT is trained on ImageNet-21k and fine-tuned on ImageNet-1k as proposed by Touvron et al. (2022) and published in timm library
    Image Size has to be 224 to keep position embedding.
    Args:
        img_size: int, width/ height of input image, only quadratic images are possible, used to determine embedding size and number of patches
        requires_grad: bool, default=False whether weights are trained or freezed during training, if True, no pretrained weights are loaded
    """

    def __init__(
        self,
        img_size: int,
        requires_grad: bool = False,
    ) -> None:
        super().__init__(img_size=img_size)
        # self.deit = timm.models.deit_base_distilled_patch16_384(
        #     pretrained=not requires_grad
        # )
        self.deit = timm.models.deit_base_distilled_patch16_224(
            pretrained=not requires_grad
        )

        self.size_patch_embedding = 768
        self.patch_size = 16
        self.num_embedded_patches = self.calc_num_embedded_patches()

        for param in self.deit.parameters():
            param.requires_grad = requires_grad

    def forward(self, x: torch.Tensor, block_index: int = 0) -> TransformerEncoderOutput:
        """Function which implements the forward pass and returns features created by DeiT transformer model. The trained layer-norm is applied when another block than the last as proposed here https://github.com/gathierry/FastFlow.
        Return: Features with shape [batch_size, number_embedded_patches, size_feature_embedding]
        """
        if block_index != 0:
            x = self.deit.patch_embed(x)
            cls_token = self.deit.cls_token.expand(x.shape[0], -1, -1)
            x = torch.cat(
                (
                    cls_token,
                    self.deit.dist_token.expand(x.shape[0], -1, -1),
                    x,
                ),
                dim=1,
            )
            x = self.deit.pos_drop(x + self.deit.pos_embed)
            for i in range(block_index + 1):
                x = self.deit.blocks[i](x)
                x = self.deit.norm(x)
        else:
            x = self.deit.forward_features(x)

        # remove class token and distillation token at the beginning
        patch_embedding = x[:, 2:, :]
        cls_token = x[:, 0, :]

        return TransformerEncoderOutput(
            patch_embedding=patch_embedding, latent_space=cls_token
        )


class EncoderVit(TransformerEncoder):
    """EncoderVit
    Encoder class which uses a vision transformer (ViT) as proposed by Dosovitsky et al.
    Computation of different image sizes is possible due to an adaptable position embedding.

    Args:
        img_size: int, width/ height of input image, only quadratic images are possible, used to determine embedding size and number of patches
        requires_grad: bool, default=False whether weights are trained or freezed during training, if True, no pretrained weights are loaded
    """

    def __init__(self, img_size: int, requires_grad: bool = False) -> None:
        super().__init__(img_size=img_size)

        self.size_patch_embedding = 768
        self.patch_size = 16
        self.num_embedded_patches = self.calc_num_embedded_patches()

        self.vit = timm.models.vit_base_patch16_224(pretrained=not requires_grad)

        for param in self.vit.parameters():
            param.requires_grad = requires_grad

    def forward(self, x, block_index: int = 0) -> TransformerEncoderOutput:
        """Forward path through vision transformer model. Removes positional embedding before returning the patch embeddings.
        Returns patch embedding of shape [batch_size, num_embedded_patches, size_patch_emebdding].
        """
        x = self.vit.forward_features(x)
        # remove cls token from output
        patch_embedding = x[:, 1:, :]
        cls_token = x[:, 0, :]
        return TransformerEncoderOutput(
            patch_embedding=patch_embedding, latent_space=cls_token
        )


class EncoderEsVit(TransformerEncoder):
    """EsVitEncoder
    Encoder class which uses a distilled trained Swin Transformer as proposed by Touvron et al.
    Computation of different image sizes is possible due to an adaptable position embedding.

    Args:
        img_size: int, width/ height of input image, only quadratic images are possible, used to determine embedding size and number of patches
        requires_grad: bool, default=False whether weights are trained or freezed during training, if True, no pretrained weights are loaded
    """

    def __init__(self, img_size: int, requires_grad: bool = False) -> None:
        super().__init__(img_size=img_size)

        self.size_patch_embedding = 768
        self.patch_size = 32
        self.num_embedded_patches = self.calc_num_embedded_patches()

        esvit_pretrained = SwinTransformer(
            patch_size=4,
            img_size=img_size,
            # hidden_dim=z_space,
            layers=(2, 2, 6, 2),
            heads=(3, 6, 12, 24),
            channels=3,
            num_classes=3,
            window_size=14,
            downscaling_factors=(4, 2, 2, 2),
            relative_pos_embedding=True,
            use_dense_prediction=True,
        )

        if not requires_grad:
            checkpoint = torch.load(
                "pretrained_vit_weights/esvit-T/checkpoint_best.pth",
                map_location=torch.device("cpu"),
            )

            student_weights = checkpoint["student"]

            new_weights = {}

            for key in student_weights:
                if not key.startswith("module.head"):
                    new_weights[key[7:]] = student_weights[key]

            delattr(esvit_pretrained, "head")

            interpolated_weights = interpolate_position_encoding(
                weights=new_weights,
                model=esvit_pretrained,
            )

            esvit_pretrained.load_state_dict(interpolated_weights)

            esvit_pretrained.freeze_pretrained_layers(frozen_layers=["*"])

        self.esvit = esvit_pretrained

    def forward(self, x, block_index: int = 0) -> TransformerEncoderOutput:
        """adjusted forward function to get feature embedding of shape [batch_size, num_embedded_patches, size_patch_emebdding] from esvit encoder"""
        x, patch_embedding = self.esvit.forward_features(x)
        # x = self.encoder_lin(x)
        return TransformerEncoderOutput(latent_space=x, patch_embedding=patch_embedding)


def interpolate_position_encoding(weights: dict, model: torch.nn.Module) -> dict:
    """Function to interpolate position encoding when images size is different from pretrained weights.
    Copied from EsViT implementation from Touvron et al. https://github.com/microsoft/esvit and modified
    """
    model_dict = model.state_dict()
    interpolated_state_dict = {}
    for k, v in weights.items():
        if ("relative_position_bias_table") in k and v.size() != model_dict[k].size():
            relative_position_bias_table_pretrained = v
            relative_position_bias_table_current = model_dict[k]
            L1, nH1 = relative_position_bias_table_pretrained.size()
            L2, nH2 = relative_position_bias_table_current.size()
            if nH1 != nH2:
                print(f"Error in loading {k}, passing")
            elif L1 != L2:
                print(
                    f"=> load_pretrained: resized variant {k}: {(L1, nH1)} to {(L2, nH2)}"
                )
                S1 = int(L1**0.5)
                S2 = int(L2**0.5)
                relative_position_bias_table_pretrained_resized = (
                    torch.nn.functional.interpolate(
                        relative_position_bias_table_pretrained.permute(1, 0).view(
                            1, nH1, S1, S1
                        ),
                        size=(S2, S2),
                        mode="bicubic",
                    )
                )
                v = relative_position_bias_table_pretrained_resized.view(nH2, L2).permute(
                    1, 0
                )

        elif ("relative_position_index") in k and v.size() != model_dict[k].size():
            relative_position_index_pretrained = v
            relative_position_index_current = model_dict[k]
            L1, H1 = relative_position_index_pretrained.size()
            L2, H2 = relative_position_index_current.size()

            print(f"=> load_pretrained: resized variant {k}: {(L1, H1)} to {(L2, H2)}")
            relative_position_index_resized = torch.nn.functional.interpolate(
                input=relative_position_index_pretrained.view(1, 1, L1, H1).float(),
                size=(L2, H2),
                mode="nearest",
            )
            v = relative_position_index_resized.view(L2, H2)

        elif "absolute_pos_embed" in k and v.size() != model_dict[k].size():
            absolute_pos_embed_pretrained = v
            absolute_pos_embed_current = model_dict[k]
            _, L1, C1 = absolute_pos_embed_pretrained.size()
            _, L2, C2 = absolute_pos_embed_current.size()
            if C1 != C2:
                print(f"Error in loading {k}, passing")
            elif L1 != L2:
                print(
                    f"=> load_pretrained: resized variant {k}: {(1, L1, C1)} to {(1, L2, C2)}"
                )
                S1 = int(L1**0.5)
                S2 = int(L2**0.5)
                absolute_pos_embed_pretrained = absolute_pos_embed_pretrained.reshape(
                    -1, S1, S1, C1
                )
                absolute_pos_embed_pretrained = absolute_pos_embed_pretrained.permute(
                    0, 3, 1, 2
                )
                absolute_pos_embed_pretrained_resized = torch.nn.functional.interpolate(
                    absolute_pos_embed_pretrained, size=(S2, S2), mode="bicubic"
                )
                v = absolute_pos_embed_pretrained_resized.permute(0, 2, 3, 1).flatten(
                    1, 2
                )

        interpolated_state_dict[k] = v
    return interpolated_state_dict
