"""Module which contains AutoEncoder with a visual transformer as encoder
"""

from src.classes.CnnAutoEncoder import AutoEncoderOutput, VanillaAutoEncoder
from src.classes.CnnDecoder import DecoderResNetVariableEmbeddingSize
from src.classes.transformer.TransformerEncoder import (
    EncoderDeit,
    EncoderEfficientFormer,
    EncoderEsVit,
    EncoderNest,
    EncoderVit,
)


class AutoEncoderViT(VanillaAutoEncoder):
    """AutoEncoder class which uses ViT by Dosovitsky et al. as Encoder"""

    def __init__(
        self,
        img_size: int,
        requires_grad: bool = False,
        red_mse="mean",
        red_ssim="elementwise_mean",
        decoder="resnet",
    ) -> None:
        encoder = EncoderVit(img_size=img_size, requires_grad=requires_grad)

        super().__init__(
            img_size=img_size,
            red_ssim=red_ssim,
            red_mse=red_mse,
            size_latent_space=encoder.size_patch_embedding,
        )

        self.size_patch_embedding = encoder.size_patch_embedding
        self.num_embedded_patches = encoder.num_embedded_patches

        self.encoder = encoder

        if decoder == "resnet":
            self.decoder = DecoderResNetVariableEmbeddingSize(
                embedding_size=encoder.size_patch_embedding
            )

        self.architecture = "transformer"

    def forward(self, x) -> AutoEncoderOutput:
        """Forward function which computes the latent space by performing Global Average Pooling on patch embedding of transformer model"""

        output = self.encoder(x)

        x_recon = self.decoder(output.latent_space)

        return AutoEncoderOutput(
            latent_space=output.latent_space,
            reconstruction=x_recon,
            patch_embedding=output.patch_embedding,
        )


class AutoEncoderEsVit(VanillaAutoEncoder):
    """AutoEncoder class which uses EsViT by Touvron et al. as Encoder"""

    def __init__(
        self,
        img_size: int,
        requires_grad: bool = False,
        red_mse="mean",
        red_ssim="elementwise_mean",
        decoder="resnet",
    ) -> None:
        encoder = EncoderEsVit(img_size=img_size, requires_grad=requires_grad)

        super().__init__(
            img_size=img_size,
            red_mse=red_mse,
            red_ssim=red_ssim,
            size_latent_space=encoder.size_patch_embedding,
        )

        self.size_patch_embedding = encoder.size_patch_embedding
        self.num_embedded_patches = encoder.num_embedded_patches

        self.encoder = encoder

        if decoder == "resnet":
            self.decoder = DecoderResNetVariableEmbeddingSize(
                embedding_size=encoder.size_patch_embedding
            )

        self.architecture = "transformer"

    def forward(self, x) -> AutoEncoderOutput:
        """Forward function which returns patch_embedding additionally to latent_space"""

        output = self.encoder(x)

        x_recon = self.decoder(output.latent_space)

        return AutoEncoderOutput(
            latent_space=output.latent_space,
            reconstruction=x_recon,
            patch_embedding=output.patch_embedding,
        )


class AutoEncoderEfficientFormer(VanillaAutoEncoder):
    """AutoEncoder class which uses EfficientFormer by Li et al. 2022 as Encoder"""

    def __init__(
        self,
        img_size: int,
        requires_grad: bool = False,
        red_mse="mean",
        red_ssim="elementwise_mean",
        decoder="resnet",
    ) -> None:
        encoder = EncoderEfficientFormer(img_size=img_size, requires_grad=requires_grad)

        super().__init__(
            img_size=img_size,
            red_mse=red_mse,
            red_ssim=red_ssim,
            size_latent_space=encoder.size_patch_embedding,
        )
        self.size_patch_embedding = encoder.size_patch_embedding
        self.num_embedded_patches = encoder.num_embedded_patches

        self.encoder = encoder

        if decoder == "resnet":
            self.decoder = DecoderResNetVariableEmbeddingSize(
                embedding_size=encoder.size_patch_embedding
            )

        self.architecture = "transformer"

    def forward(self, x) -> AutoEncoderOutput:
        """Forward function which returns patch_embedding additionally to latent_space"""

        output = self.encoder(x)

        x_recon = self.decoder(output.latent_space)

        return AutoEncoderOutput(
            latent_space=output.latent_space,
            reconstruction=x_recon,
            patch_embedding=output.patch_embedding,
        )


class AutoEncoderDeit(VanillaAutoEncoder):
    """AutoEncoder class which uses DeiT by Touvron et al. 2022 as Encoder"""

    def __init__(
        self,
        img_size: int,
        requires_grad: bool = False,
        red_mse="mean",
        red_ssim="elementwise_mean",
        decoder="resnet",
    ) -> None:
        encoder = EncoderDeit(img_size=img_size, requires_grad=requires_grad)

        super().__init__(
            img_size=img_size,
            red_mse=red_mse,
            red_ssim=red_ssim,
            size_latent_space=encoder.size_patch_embedding,
        )
        self.size_patch_embedding = encoder.size_patch_embedding
        self.num_embedded_patches = encoder.num_embedded_patches

        self.encoder = encoder

        if decoder == "resnet":
            self.decoder = DecoderResNetVariableEmbeddingSize(
                embedding_size=encoder.size_patch_embedding
            )

        self.architecture = "transformer"

    def forward(self, x) -> AutoEncoderOutput:
        """Forward function which returns patch_embedding additionally to latent_space"""

        output = self.encoder(x)

        x_recon = self.decoder(output.latent_space)

        return AutoEncoderOutput(
            latent_space=output.latent_space,
            reconstruction=x_recon,
            patch_embedding=output.patch_embedding,
        )


class AutoEncoderNest(VanillaAutoEncoder):
    """AutoEncoder class which uses NesT by Zhang et al. 2021 as Encoder"""

    def __init__(
        self,
        img_size: int,
        requires_grad: bool = False,
        red_mse="mean",
        red_ssim="elementwise_mean",
        decoder="resnet",
    ) -> None:
        encoder = EncoderNest(img_size=img_size, requires_grad=requires_grad)

        super().__init__(
            img_size=img_size,
            red_mse=red_mse,
            red_ssim=red_ssim,
            size_latent_space=encoder.size_patch_embedding,
        )
        self.size_patch_embedding = encoder.size_patch_embedding
        self.num_embedded_patches = encoder.num_embedded_patches

        self.encoder = encoder

        if decoder == "resnet":
            self.decoder = DecoderResNetVariableEmbeddingSize(
                embedding_size=encoder.size_patch_embedding
            )

        self.architecture = "transformer"

    def forward(self, x) -> AutoEncoderOutput:
        """Forward function which returns patch_embedding additionally to latent_space"""

        output = self.encoder(x)

        x_recon = self.decoder(output.latent_space)

        return AutoEncoderOutput(
            latent_space=output.latent_space,
            reconstruction=x_recon,
            patch_embedding=output.patch_embedding,
        )
