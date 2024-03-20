from __future__ import annotations

import getopt
import os
import sys

from src.classes.CnnEncoder import ResNetEncoder
from src.data_loader.GeneralDataLoader import GeneralDataLoader
from src.pipeline.LearnerNF import LearnerNF
from src.pipeline.LearnerRecon import HyperParameterConfig
from src.util.ModelHelper import get_model, get_possible_models

btad_base = "data/btad/BTech_Dataset_transformed/01"
btad_train_pref = "train/ok"
btad_test_pref = "test"

mvtec_base = "data/mvtec_anomaly_detection/hazelnut"
mvtec_train_pref = "train/good"
mvtec_test_pref = "test"

d3_dataset = "data/3DPrinterDefectedDataset"
d3_train_pref = "no_defected"


def main(argv):
    model_string = "enc_deit"
    patience = 60
    epochs = 500
    amount_data = 0
    lr = 1e-3
    wd = 1e-5
    centering = False
    batch_size = 32
    data_path: str = mvtec_base
    train_pref: str = mvtec_train_pref
    test_pref: str = mvtec_test_pref
    img_size = 224
    hidden_ratio = 0.16
    flow_steps = 20

    opts, _ = getopt.getopt(argv, "hm:p:e:a:l:w:b:s:d:i:v:r:f:t:", ["centering"])

    for opt, arg in opts:
        if opt == "-h":
            print(
                "startTraining.py -m <modelType> -p <patience> -e <#epochs> -a <amountOfData> -l <learnRate> -w "
                "<weightDecay> -b <batchSize> -d <dataPath> -t <trainPref> -i <imageSize> -v <validPref> -r <hiddenRatio> -f <flowSteps> --centering"
            )
            print(f"Possible model values are: \n {get_possible_models()}")
            print(
                f"Default values are: model type: {model_string}, patience: {patience}, epochs: {epochs}, "
                f"amount of data: {amount_data}, learn rate: {lr}, weight decay: {wd}, batch size: {batch_size}, hidden ration: {hidden_ratio}, "
                f"centering: {centering}, data path: {data_path}, train pref: {train_pref}, valid pref: {test_pref}, image size: {img_size}, flow steps: {flow_steps}"
            )
            return

        if opt == "--centering":
            centering = True
        elif opt == "-p":
            patience = int(arg)
        elif opt == "-e":
            epochs = int(arg)
        elif opt == "-a":
            amount_data = int(arg)
        elif opt == "-l":
            lr = float(arg)
        elif opt == "-w":
            wd = float(arg)
        elif opt == "-b":
            batch_size = int(arg)
        elif opt == "-d":
            data_path = arg
        elif opt == "-t":
            train_pref = arg
        elif opt == "-i":
            img_size = int(arg)
        elif opt == "-m":
            model_string = arg.lower()
        elif opt == "-v":
            test_pref = arg
        elif opt == "-r":
            hidden_ratio = float(arg)
        elif opt == "-f":
            flow_steps = int(arg)

    model = get_model(name=model_string, img_size=img_size, requires_grad=False)

    if model is None:
        return "Please specify a valid model."

    dataloader = GeneralDataLoader(
        img_size=img_size,
        batch_size=batch_size,
        base_path=data_path,
        data_path=train_pref,
        # valid_path=test_pref,
    ).get_dataloader(amount_data=amount_data, centering=centering)

    train_loader = dataloader.train_loader
    valid_loader = dataloader.valid_loader
    test_loader_factory = GeneralDataLoader(
        img_size=img_size,
        batch_size=batch_size,
        base_path=data_path,
        data_path=test_pref,
        validation_mode=True,
    )

    learner = LearnerNF(
        encoder=model, enable_wandb=True, hidden_ratio=hidden_ratio, flow_steps=flow_steps
    )
    amount_data = len(train_loader.dataset)
    print(
        f"Training on {learner.device}, model type: {type(model).__name__}, patience: {patience}, epochs: {epochs}, "
        f"amount of data: {amount_data}, learn rate: {lr}, weight decay: {wd}, image size: {img_size}, hidden ratio: {hidden_ratio}, batch size: {batch_size}, flow steps: {flow_steps},  centering: "
        f"{centering}, data path: {os.path.join(data_path,train_pref)}"
    )

    dataset = data_path.split("/")[1]
    dataclass = data_path.split("/")[3] if dataset == "btad" else data_path.split("/")[2]

    hyper_param_dict: HyperParameterConfig = {
        "amount_data": amount_data,
        "ad_type": "nf",
        "learning_rate": lr,
        "weight_decay": wd,
        "batch_size": batch_size,
        "img_size": img_size,
        "patience": patience,
        "epochs": epochs,
        "centering": centering,
        "dataset": dataset,
        "dataclass": dataclass,
        "hidden_ratio": hidden_ratio,  # fix for training with transformers as done here https://github.com/gathierry/FastFlow/ and described in fast flow paper
        "flow_steps": flow_steps,  # fix for training with transformers as done here https://github.com/gathierry/FastFlow/
        "decoder": "NormalizingFlow",
    }

    if isinstance(model, ResNetEncoder):
        learner.train_with_resnet(
            train_loader=train_loader,
            valid_loader=valid_loader,
            hyper_param_dict=hyper_param_dict,
            test_loader=test_loader_factory,
        )
    else:
        learner.train_with_transformer(
            train_loader=train_loader,
            valid_loader=valid_loader,
            hyper_param_dict=hyper_param_dict,
            test_loader=test_loader_factory,
        )


if __name__ == "__main__":
    main(sys.argv[1:])
