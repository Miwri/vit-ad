from os import listdir

from src.classes.CnnAutoEncoder import (AutoEncoderResNet,
                                        AutoEncoderResNetSmallDecoder)
from src.classes.CnnEncoder import ResNetEncoder
from src.classes.MixtureDensityNetwork import GaussianMixtureDensityNetwork
from src.classes.NormalizingFlow import NormalizingFlow
from src.classes.transformer.TransformerEncoder import EncoderDeit
from src.data_loader.GeneralDataLoader import GeneralDataLoader
from src.pipeline.ValidatorMDN import ValidatorMdn
from src.pipeline.ValidatorNF import ValidatorNF
from src.pipeline.ValidatorRecon import ValidatorRecon
from src.util.ModelHelper import (RES_NET_MEAN, RES_NET_STD, get_model,
                                  get_possible_models)

MODEL_STRING = "enc_deit"

base_path_weights = f"trained_model_weights/evaluated_recon_resnetdecod/btad_all"

btad_base = "data/btad/BTech_Dataset_transformed/"
btad_train_pref = "train/ok"
btad_test_pref = "test"

mvtec_base = "data/mvtec_anomaly_detection/"
mvtec_train_pref = "train/good"
mvtec_test_pref = "test"

img_size = 224

BLOCK_INDEX_DEIT = 0

EXPERIMENT = "resnet_all_btad"


def validate_mdn(weights: list[str]):
    """Evaluate weights for Mixture Density Networks with transformer Encoder."""
    for i, weight in enumerate(weights):
        num_gaussians = int(weight.split("_")[0])
        dataclass = weight.split("_")[-1][:-4]
        if dataclass in ("nut", "metal"):
            dataclass = "metal_nut"

        feature_extractor = get_model(
            name=MODEL_STRING, img_size=img_size, requires_grad=False
        )

        dataloader = GeneralDataLoader(
            batch_size=32,
            base_path=f"{mvtec_base}{dataclass}",
            data_path=mvtec_test_pref,
            img_size=img_size,
            validation_mode=True,
        )

        gmm_1 = GaussianMixtureDensityNetwork(
            cluster_centers=None,
            input_dim=feature_extractor.size_patch_embedding,
            output_dim=feature_extractor.size_patch_embedding,
            num_gaussians=num_gaussians,
        )

        print(
            f"Evaluating {dataclass} for {type(feature_extractor).__name__} {type(gmm_1).__name__} "
        )

        validator = ValidatorMdn(
            gmm_model=[gmm_1],
            feature_extractor=feature_extractor,
            dataloader=dataloader,
            weights_base_path=base_path_weights,
            weights_name=[weight],
            props={
                "architecture": f"{type(feature_extractor).__name__}_{type(gmm_1).__name__}",
                "encoder_type": feature_extractor.architecture,
                "encoder": type(feature_extractor).__name__,
                "num_gaussians": num_gaussians,
                "dataclass": dataclass,
                "dataset": "mvtec",
                "experiment": EXPERIMENT,
                "fp_thres": 0.3,
            },
        )

        validator.calc_all_metrics()


def validate_mdn_resnet(weights: list[str]):
    """Evaluate weights for Mixture Density Networks with ResNet Encoder."""
    weight_names = []
    data_classes = []
    for i, weight in enumerate(weights):
        num_gaussians = int(weight.split("_")[0])
        dataclass = weight.split("_")[-1][:-4]
        if dataclass in ("nut", "metal"):
            dataclass = "metal_nut"
        data_classes.append(dataclass)
        weight_names.append(weight)

    feature_extractor = get_model(
        name=MODEL_STRING, img_size=img_size, requires_grad=False
    )

    dataloader = GeneralDataLoader(
        batch_size=8,
        base_path=f"{mvtec_base}{dataclass}",
        data_path=mvtec_test_pref,
        img_size=img_size,
        validation_mode=True,
    )

    gmm_1 = GaussianMixtureDensityNetwork(
        cluster_centers=None,
        input_dim=feature_extractor.res_net.in_channels[1],
        output_dim=feature_extractor.res_net.in_channels[1],
        num_gaussians=num_gaussians,
    )
    gmm_2 = GaussianMixtureDensityNetwork(
        cluster_centers=None,
        input_dim=feature_extractor.res_net.in_channels[2],
        output_dim=feature_extractor.res_net.in_channels[2],
        num_gaussians=num_gaussians,
    )

    unique_data_classes = set(data_classes)
    unique_data_classes = list(unique_data_classes)
    unique_data_classes.sort()

    num_classes = len(unique_data_classes)

    print(unique_data_classes)

    for i, name in enumerate(unique_data_classes):
        print(
            f"Evaluating {name} for {type(feature_extractor).__name__} {type(gmm_1).__name__} "
        )

        weight_0 = weight_names[i]
        weight_1 = weight_names[i + num_classes]

        validator = ValidatorMdn(
            gmm_model=[gmm_1, gmm_2],
            feature_extractor=feature_extractor,
            dataloader=dataloader,
            weights_base_path=base_path_weights,
            weights_name=[weight_0, weight_1],
            props={
                "architecture": f"{type(feature_extractor).__name__}_{type(gmm_1).__name__}",
                "encoder_type": feature_extractor.architecture,
                "encoder": type(feature_extractor).__name__,
                "num_gaussians": num_gaussians,
                "dataclass": name,
                "dataset": "mvtec",
                "experiment": EXPERIMENT,
                "fp_thres": 0.3,
            },
        )

        validator.calc_all_metrics()


def validate_nf(weights: list[str], model: str):
    """Evaluate weights for Normalizing Flow with transformer Encoder."""
    for i, weight in enumerate(weights):
        dataclass = weight.split("_")[-1][:-4]
        if dataclass in ("nut", "metal"):
            dataclass = "metal_nut"

        feature_extractor = get_model(name=model, img_size=img_size, requires_grad=False)

        dataloader = GeneralDataLoader(
            batch_size=32,
            base_path=f"{mvtec_base}{dataclass}",
            data_path=mvtec_test_pref,
            img_size=img_size,
            validation_mode=True,
        )

        nf_model_0 = NormalizingFlow(
            num_channels=feature_extractor.size_patch_embedding,
            num_patches=feature_extractor.num_embedded_patches,
            img_size=img_size,
            hidden_ratio=0.16,
            flow_steps=20,
        )

        print(
            f"Evaluating {dataclass} for {type(feature_extractor).__name__} {type(nf_model_0).__name__} "
        )

        validator = ValidatorNF(
            feature_extractor=feature_extractor,
            nf_model=[nf_model_0],
            dataloader=dataloader,
            weights_base_path=f"{base_path_weights}/{model}",
            weights_name=weight_paths,
            props={
                "architecture": f"{type(feature_extractor).__name__}_{type(nf_model_0).__name__}",
                "encoder_type": feature_extractor.architecture,
                "encoder": type(feature_extractor).__name__,
                "dataclass": dataclass,
                "dataset": "mvtec",
                "experiment": EXPERIMENT,
                "fp_thres": 0.3,
            },
        )

        validator.calc_all_metrics()


def validate_recon(weights: list[str], model_name: str):
    """Evaluate weights for Auto-Encoder"""
    for i, weight in enumerate(weights):
        dataclass = weight.split("_")[-1][:-4]
        if dataclass in ("nut", "metal"):
            dataclass = "metal_nut"

        model = get_model(name=model_name, img_size=img_size, requires_grad=False)

        dataloader = GeneralDataLoader(
            batch_size=32,
            base_path=f"{btad_base}{dataclass}",
            data_path=btad_test_pref,
            img_size=img_size,
            validation_mode=True,
        )

        print(f"Evaluating {dataclass} for {type(model).__name__} ")

        validator = ValidatorRecon(
            model=model,
            dataloader=dataloader,
            weights_base_path=base_path_weights,
            weights_name=weight,
            props={
                "architecture": f"{type(model).__name__}_{type(model.decoder).__name__}",
                "encoder_type": model.architecture,
                "encoder": type(model.encoder).__name__,
                "dataclass": dataclass,
                "dataset": "mvtec",
                "experiment": EXPERIMENT,
                "fp_thres": 0.3,
            },
        )

        validator.calc_all_metrics()


if __name__ == "__main__":
    print("Start validation loop")

    models = ["ae_res_net"]

    for model_str in models:
        weight_paths = [
            weight for weight in listdir(base_path_weights) if weight.endswith(".pth")
        ]
        # weight_paths = [
        #     weight
        #     for weight in listdir(f"{base_path_weights}/{model_str}")
        #     if weight.endswith(".pth")
        # ]

        weight_paths.sort()

        print(weight_paths)

        # if MODEL_STRING == "enc_res_net":
        #     validate_mdn_resnet(weight_paths)
        # else:
        #     validate_mdn(weight_paths)

        # validate_nf(weight_paths, model_str)

        validate_recon(weight_paths, model_str)

    # feature_extractor = EncoderDeit(img_size=img_size, block_index=BLOCK_INDEX_DEIT)

    # nf_model_1 = NormalizingFlow(
    #     num_channels=feature_extractor.res_net.in_channels[2],
    #     num_patches=int((img_size / feature_extractor.res_net.scales[2]) ** 2),
    #     img_size=img_size,
    #     hidden_ratio=0.16,
    #     flow_steps=20,
    # )
    # nf_model_2 = NormalizingFlow(
    #     num_channels=feature_extractor.res_net.in_channels[3],
    #     num_patches=int((img_size / feature_extractor.res_net.scales[3]) ** 2),
    #     img_size=img_size,
    #     hidden_ratio=0.16,
    #     flow_steps=20,
    # )
