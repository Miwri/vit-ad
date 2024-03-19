from src.classes import CnnAutoEncoder, CnnEncoder
from src.classes.transformer import TransformerAutoEncoder, TransformerEncoder

# https://pytorch.org/hub/pytorch_vision_resnet/
RES_NET_MEAN = [0.485, 0.456, 0.406]
RES_NET_STD = [0.229, 0.224, 0.225]

MODEL_DICT = {
    "enc_cnn": CnnEncoder.EncoderVanillaCNN,
    "enc_eff_net": CnnEncoder.EfficientNetEncoder,
    "enc_res_net": CnnEncoder.ResNetEncoder,
    "enc_nest": TransformerEncoder.EncoderNest,
    "enc_eff_former": TransformerEncoder.EncoderEfficientFormer,
    "enc_deit": TransformerEncoder.EncoderDeit,
    "enc_vit": TransformerEncoder.EncoderVit,
    "enc_esvit": TransformerEncoder.EncoderEsVit,
    "ae_cnn": CnnAutoEncoder.VanillaAutoEncoder,
    "ae_res_net": CnnAutoEncoder.AutoEncoderResNet,
    "ae_res_net_small": CnnAutoEncoder.AutoEncoderResNetSmallDecoder,
    "ae_nest": TransformerAutoEncoder.AutoEncoderNest,
    "ae_eff_former": TransformerAutoEncoder.AutoEncoderEfficientFormer,
    "ae_deit": TransformerAutoEncoder.AutoEncoderDeit,
    "ae_vit": TransformerAutoEncoder.AutoEncoderViT,
    "ae_esvit": TransformerAutoEncoder.AutoEncoderEsVit,
    "ae_nest_small": TransformerAutoEncoder.AutoEncoderNest,
    "ae_eff_former_small": TransformerAutoEncoder.AutoEncoderEfficientFormer,
    "ae_deit_small": TransformerAutoEncoder.AutoEncoderDeit,
    "ae_vit_small": TransformerAutoEncoder.AutoEncoderViT,
    "ae_esvit_small": TransformerAutoEncoder.AutoEncoderEsVit,
}


def get_model(name: str, img_size: int = 224, requires_grad: bool = False):
    """Model Factory which returns an autoencoder or encoder model, depending on the name
    Args:
        name: string name which specifies the model
        img_size: int, default=224, size of input images, have to be quadratic
        requires_grad: bool, whether the encoder model is trained or not, if not pretrained weights are loaded and freezed, option only valid for transformer based models
    """

    try:
        if ("cnn" in name) or ("res_net" in name) or ("eff_net" in name):
            if "ae" in name:
                return MODEL_DICT[name](img_size=img_size, red_mse="none")
            return MODEL_DICT[name](
                img_size=img_size,
            )
        if "ae" in name:
            if "small" in name:
                return MODEL_DICT[name](
                    img_size=img_size,
                    requires_grad=requires_grad,
                    red_mse="none",
                    decoder="cnn",
                )
            return MODEL_DICT[name](
                img_size=img_size, requires_grad=requires_grad, red_mse="none"
            )
        return MODEL_DICT[name](img_size=img_size, requires_grad=requires_grad)

    except KeyError:
        print(
            f"Defined model ${name} not known. Please specify one of the following model names: \n {get_possible_models()}"
        )
        return None


def get_possible_models():
    """Function returns all keys which are possible to create a model"""
    return list(MODEL_DICT.keys())
