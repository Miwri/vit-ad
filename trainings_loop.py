from startTraining_mdn import main as start_training_mdn
from startTraining_NF import main as start_training_nf
from startTraining_recon import main as start_training_recon

all_prods_mvtec = [
    "bottle",
    # "cable",
    # "capsule",
    "carpet",
    # "grid",
    "hazelnut",
    "leather",
    # "metal_nut",
    # "pill",
    "screw",
    # "tile",
    # "toothbrush",
    # "transistor",
    # "wood",
    # "zipper",
]

backbones = [
    "enc_res_net",
    # "enc_nest",
    # "enc_eff_former",
    # "enc_deit",
    # "enc_esvit",
]


all_prods_btad = [
    "01",
    "02",
    "03",
]

btad_base = "data/btad/BTech_Dataset_transformed"
btad_train_pref = "train/ok"
btad_test_pref = "test"

mvtec_base = "data/mvtec_anomaly_detection"
mvtec_train_pref = "train/good"
mvtec_test_pref = "test"

# prods_recon = [
#     "bottle",
#     "hazelnut",
#     "leather",
#     "screw",
#     "wood",
# ]

backbones_recon = [
    # "ae_cnn",
    # "ae_res_net",
    # "ae_res_net_small",
    # "ae_nest",
    # "ae_eff_former",
    "ae_deit",
    # "ae_esvit",
    # "ae_nest_small",
    # "ae_eff_former_small",
    # "ae_deit_small",
    # "ae_esvit_small",
]

params = [
    # "-a",
    # "32",
    "-e",
    "500",
    "-p",
    "30",
    "-l",
    "0.001",
    "-w",
    "0.0001",
    "-b",
    "2",
    "-m",
    "enc_deit",
    "-d",
    btad_base,
    "-t",
    btad_train_pref,
    "-v",
    btad_test_pref,
]

if __name__ == "__main__":
    # learning_rates_nf = [1e-3]
    # weight_decays_nf = [1e-5]

    # for prod in all_prods_mvtec:
    #     index_d = params.index("-d")
    #     params[index_d + 1] = mvtec_base + "/" + prod
    #     for m in backbones:
    #         index_m = params.index("-m")
    #         params[index_m + 1] = m

    #         index_b = params.index("-b")
    #         params[index_b + 1] = 8 if m == "enc_res_net" else 32
    #         for i, l in enumerate(learning_rates_nf):
    #             index_l = params.index("-l")
    #             params[index_l + 1] = l
    #             index_w = params.index("-w")
    #             params[index_w + 1] = weight_decays_nf[i]

    #             # try:
    #             start_training_nf(argv=params)
    #             # except Exception as err:
    #             #     print(
    #             #         f"Training with backbone {m} did not work due to the following exception. Continue training with the next backbone."
    #             #     )
    #             #     print(err)

    learning_rates_nf = [1e-3, 1e-4]
    weight_decays_nf = [1e-5, 1e-5]

    params_nf = [
        # "-a",
        # "32",
        "-e",
        "500",
        "-p",
        "30",
        "-l",
        "0.001",
        "-w",
        "0.0001",
        "-b",
        "2",
        "-m",
        "enc_deit",
        "-f",
        "8",
        "-d",
        mvtec_base,
        "-t",
        mvtec_train_pref,
        "-v",
        mvtec_test_pref,
    ]

    for prod in all_prods_mvtec:
        index_d = params_nf.index("-d")
        params_nf[index_d + 1] = mvtec_base + "/" + prod
        for m in backbones:
            index_m = params_nf.index("-m")
            params_nf[index_m + 1] = m

            index_f = params_nf.index("-f")
            params_nf[index_f + 1] = 8 if m == "enc_res_net" else 20

            index_b = params_nf.index("-b")
            params_nf[index_b + 1] = 32 if m == "enc_res_net" else 32
            for i, l in enumerate(learning_rates_nf):
                index_l = params_nf.index("-l")
                params_nf[index_l + 1] = l
                index_w = params_nf.index("-w")
                params_nf[index_w + 1] = weight_decays_nf[i]

                start_training_nf(argv=params_nf)

    learning_rates_recon = [5e-4]
    weight_decays_recon = [1e-5]

    for prod in all_prods_btad:
        index_d = params.index("-d")
        params[index_d + 1] = btad_base + "/" + prod
        for m in backbones_recon:
            index_m = params.index("-m")
            params[index_m + 1] = m

            index_b = params.index("-b")
            params[index_b + 1] = 32
            for i, l in enumerate(learning_rates_recon):
                index_l = params.index("-l")
                params[index_l + 1] = l
                index_w = params.index("-w")
                params[index_w + 1] = weight_decays_recon[i]

                # try:
                start_training_recon(argv=params)
                # except Exception as err:
                #     print(
                #         f"Training with backbone {m} did not work due to the following exception. Continue training with the next backbone."
                #     )
                #     print(err)

    # params_mdn = [
    #     # "-a",
    #     # "10",
    #     "-e",
    #     "500",
    #     "-p",
    #     "30",
    #     "-l",
    #     "0.001",
    #     "-w",
    #     "0.0001",
    #     "-b",
    #     "2",
    #     "-m",
    #     "enc_deit",
    #     "-n",
    #     "100",
    #     "-d",
    #     mvtec_base,
    #     "-t",
    #     mvtec_train_pref,
    #     "-v",
    #     mvtec_test_pref,
    # ]

    # all_prods_mvtec = [
    #     # "bottle",
    #     "cable",
    #     # "capsule",
    #     "carpet",
    #     "grid",
    #     "hazelnut",
    #     # "leather",
    #     # "metal_nut",
    #     # "pill",
    #     # "screw",
    #     "tile",
    #     # "toothbrush",
    #     # "transistor",
    #     # "wood",
    #     # "zipper",
    # ]

    # learning_rates_mdn = [1e-4]
    # weight_decays_mdn = [1e-4]

    # gaussians = [50]

    # for prod in all_prods_mvtec:
    #     index_d = params_mdn.index("-d")
    #     params_mdn[index_d + 1] = mvtec_base + "/" + prod
    #     for m in backbones:
    #         index_m = params_mdn.index("-m")
    #         params_mdn[index_m + 1] = m
    #         for n in gaussians:
    #             index_n = params_mdn.index("-n")
    #             # params_mdn[index_n + 1] = 50 if m == "enc_res_net" else n
    #             params_mdn[index_n + 1] = n
    #             index_b = params_mdn.index("-b")
    #             params_mdn[index_b + 1] = 4 if n > 110 else 8
    #             params_mdn[index_b + 1] = (
    #                 4 if m == "enc_res_net" else params_mdn[index_b + 1]
    #             )
    #             for i, l in enumerate(learning_rates_mdn):
    #                 index_l = params_mdn.index("-l")
    #                 params_mdn[index_l + 1] = l
    #                 index_w = params_mdn.index("-w")
    #                 params_mdn[index_w + 1] = weight_decays_mdn[i]

    #                 # try:
    #                 start_training_mdn(argv=params_mdn)
    #                 # except Exception as err:
    #                 #     print(
    #                 #         f"Training with backbone {m} did not work due to the following exception. Continue training with the next backbone."
    #                 #     )
    #                 #     print(err)

    # gaussians = [130]

    # for prod in all_prods_mvtec:
    #     index_d = params_mdn.index("-d")
    #     params_mdn[index_d + 1] = mvtec_base + "/" + prod
    #     for m in backbones:
    #         index_m = params_mdn.index("-m")
    #         params_mdn[index_m + 1] = m
    #         for n in gaussians:
    #             index_n = params_mdn.index("-n")
    #             # params_mdn[index_n + 1] = 50 if m == "enc_res_net" else n
    #             params_mdn[index_n + 1] = n
    #             index_b = params_mdn.index("-b")
    #             params_mdn[index_b + 1] = 4 if n > 110 else 8
    #             params_mdn[index_b + 1] = (
    #                 4 if m == "enc_res_net" else params_mdn[index_b + 1]
    #             )
    #             for i, l in enumerate(learning_rates_mdn):
    #                 index_l = params_mdn.index("-l")
    #                 params_mdn[index_l + 1] = l
    #                 index_w = params_mdn.index("-w")
    #                 params_mdn[index_w + 1] = weight_decays_mdn[i]

    #                 start_training_mdn(argv=params_mdn)
