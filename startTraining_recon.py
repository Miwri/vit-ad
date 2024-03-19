from __future__ import annotations

import getopt
import os
import sys

from torchsummary import summary

from src.classes.VariationalAutoEncoder import VariationalAutoEncoder
from src.data_loader.GeneralDataLoader import GeneralDataLoader
from src.pipeline.LearnerRecon import HyperParameterConfig, LearnerRecon
from src.util.ModelHelper import RES_NET_MEAN, RES_NET_STD, get_model, get_possible_models

btad_base = "data/btad/BTech_Dataset_transformed/01"
btad_train_pref = "train/ok"
btad_test_pref = "test"

mvtec_base = "data/mvtec_anomaly_detection/hazelnut"
mvtec_train_pref = "train/good"
mvtec_test_pref = "test"

d3_dataset = "data/3DPrinterDefectedDataset"
d3_train_pref = "no_defected"


def main(argv):
    model_string = "ae_cnn"
    patience = 30
    epochs = 50
    amount_data = 0
    lr = 1e-3
    wd = 1e-4
    centering = False
    batch_size = 64
    data_path: str = mvtec_base
    train_pref: str = mvtec_train_pref
    test_pref: str = mvtec_test_pref
    img_size = 224

    opts, _ = getopt.getopt(argv, "hm:p:e:a:l:w:b:s:d:i:v:t:", ["centering"])

    for opt, arg in opts:
        if opt == "-h":
            print(
                "startTraining.py -m <modelType> -p <patience> -e <#epochs> -a <amountOfData> -l <learnRate> -w "
                "<weightDecay> -b <batchSize> -d <dataPath> -t <trainPref> -i <imageSize> -v <validPref> --centering"
            )
            print(f"Possible model values are: \n {get_possible_models()}")
            print(
                f"Default values are: model type: {model_string}, patience: {patience}, epochs: {epochs}, "
                f"amount of data: {amount_data}, learn rate: {lr}, weight decay: {wd}, batch size: {batch_size}, "
                f"centering: {centering}, data path: {data_path}, train pref: {train_pref}, valid pref: {test_pref}, image size: {img_size}"
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

    model = get_model(name=model_string, img_size=img_size, requires_grad=False)

    if model is None:
        return "Please specify a valid model."

    if centering:
        dataloader_factory = GeneralDataLoader(
            img_size=img_size,
            batch_size=batch_size,
            base_path=data_path,
            data_path=train_pref,
            # valid_path=test_pref,
        )

        dataloader_factory.mean = RES_NET_MEAN
        dataloader_factory.std = RES_NET_STD

        dataloader = dataloader_factory.get_dataloader(
            amount_data=amount_data, centering=True
        )
    else:
        dataloader = GeneralDataLoader(
            img_size=img_size,
            batch_size=batch_size,
            base_path=data_path,
            data_path=train_pref,
            # valid_path=test_pref,
        ).get_dataloader(amount_data=amount_data, centering=False)

    train_loader = dataloader.train_loader
    valid_loader = dataloader.valid_loader
    test_loader_factory = GeneralDataLoader(
        img_size=img_size,
        batch_size=batch_size,
        base_path=data_path,
        data_path=test_pref,
        validation_mode=True,
    )

    learner = LearnerRecon(model=model)
    amount_data = len(train_loader.dataset)
    print(
        f"Training on {learner.device}, model type: {type(model).__name__}, patience: {patience}, epochs: {epochs}, "
        f"amount of data: {amount_data}, learn rate: {lr}, weight decay: {wd}, image size: {img_size}, batch size: {batch_size}, centering: "
        f"{centering}, data path: {os.path.join(data_path,train_pref)}"
    )

    dataset = data_path.split("/")[1]
    dataclass = data_path.split("/")[3] if dataset == "btad" else data_path.split("/")[2]

    hyper_param_dict: HyperParameterConfig = {
        "amount_data": amount_data,
        "ad_type": "recon",
        "learning_rate": lr,
        "weight_decay": wd,
        "batch_size": batch_size,
        "img_size": img_size,
        "patience": patience,
        "epochs": epochs,
        "centering": centering,
        "dataset": dataset,
        "dataclass": dataclass,
        "decoder": type(model.decoder).__name__,
    }

    if isinstance(model, VariationalAutoEncoder):
        learner.learn_vae(
            hyper_param_dict=hyper_param_dict,
            train_loader=train_loader,
            valid_loader=valid_loader,
        )
    else:
        learner.learn_ae_with_MSE_only(
            hyper_param_dict=hyper_param_dict,
            train_loader=train_loader,
            valid_loader=valid_loader,
            test_loader=test_loader_factory,
        )


if __name__ == "__main__":
    main(sys.argv[1:])
