# -*- coding: utf-8 -*-
import os
import sys
import re
import argparse

import pandas as pd

# naming convention
# key: config name
# value: ([version, num_gpus], rename, watt, price)
# version: 0 for pytorch:22.10-py3, and 1 for pytorch:24.10-py3 (currently irrelevant, when different versions of pytorch changes field positions this is relevant)
# num_gpus: sometimes num_gpus can't be inferred from config name (for example p3.16xlarge) or missing from the result log. So we ask for user to specify it here.
# rename: renaming the system so it is easier to read
# watt per gpu
# price per gpu

list_system_single = {
    "LambdaBM_ENG04_1xH200_140GB_SXM_h200_v2": ([1, 1],  "LambdaBM ENG04 1xH200 140GB SXM", 700, 39375),
    "LambdaBM_ENG04_HTOff_1xH200_140GB_SXM_h200_v2": ([1, 1],  "LambdaBM ENG04 HTOff 1xH200 140GB SXM", 700, 39375),
    "LambdaBM_Radiant_1xGH200_96GB_v2": ([1, 1],  "LambdaBM Radiant 1xGH200 96GB", 700, 45000),
    "LambdaOD_1x_1xGH200_80GB_192-222-58-35_v2": ([1, 1],  "LambdaOD 1x 1xGH200 80GB", 700, 45000),
    "LambdaOD_1x_1xGH200_96GB_192-222-57-0_v2": ([1, 1],  "LambdaOD 1x 1xGH200 96GB", 700, 45000),
    "LambdaOD_1x_1xH100_80GB_SXM5_192-222-52-83_v2": ([1, 1],  "LambdaOD 1x 1xH100 80GB SXM5", 700, 36718.75),
    "LambdaOD_1x_Test_1xGH200_96GB_192-222-56-184_v2": ([1, 1],  "LambdaOD 1x Test 1xGH200 96GB", 700, 45000),
    "LambdaOD_1x_Test_1xGH200_96GB_192-222-56-184_v2_bk": ([1, 1],  "LambdaOD 1x Test 1xGH200 96GB bk", 700, 45000),
    "LambdaOD_1x_Texas_1xH100_80GB_PCIe_209-20-158-50_v2": ([1, 1],  "LambdaOD 1x Texas 1xH100 80GB PCIe", 350, 30918),
    "LambdaOD_1x_Texas_1xH100_80GB_SXM5_192-222-52-129_v2": ([1, 1],  "LambdaOD 1x Texas 1xH100 80GB SXM5", 700, 36718.75),
    "LambdaOD_1x_Texas_1xH100_80GB_SXM5_192-222-52-154_v2": ([1, 1],  "LambdaOD 1x Texas 1xH100 80GB SXM5", 700, 36718.75),
    "LambdaOD_1x_Texas_1xH100_80GB_SXM5_192-222-52-179_v2": ([1, 1],  "LambdaOD 1x Texas 1xH100 80GB SXM5", 700, 36718.75),
    "LambdaOD_1x_Texas_1xH100_80GB_SXM5_192-222-52-249_v2": ([1, 1],  "LambdaOD 1x Texas 1xH100 80GB SXM5", 700, 36718.75),
    "LambdaOD_1x_Texas_1xH100_80GB_SXM5_192-222-52-60_v2": ([1, 1],  "LambdaOD 1x Texas 1xH100 80GB SXM5", 700, 36718.75),
    "LambdaOD_1x_Texas_1xH100_80GB_SXM5_192-222-52-74_v2": ([1, 1],  "LambdaOD 1x Texas 1xH100 80GB SXM5", 700, 36718.75),
    "LambdaOD_1x_Texas_1xH100_80GB_SXM5_192-222-52-77_v2": ([1, 1],  "LambdaOD 1x Texas 1xH100 80GB SXM5", 700, 36718.75),
    "LambdaOD_1x_Texas_1xH100_80GB_SXM5_192-222-52-92_v2": ([1, 1],  "LambdaOD 1x Texas 1xH100 80GB SXM5", 700, 36718.75),
    "LambdaOD_2x_Texas_1xH100_80GB_SXM5_192-222-52-120_v2": ([1, 1],  "LambdaOD 2x Texas 1xH100 80GB SXM5", 700, 36718.75),
    "LambdaOD_2x_Texas_1xH100_80GB_SXM5_192-222-52-176_v2": ([1, 1],  "LambdaOD 2x Texas 1xH100 80GB SXM5", 700, 36718.75),
    "LambdaOD_2x_Texas_1xH100_80GB_SXM5_192-222-52-211_v2": ([1, 1],  "LambdaOD 2x Texas 1xH100 80GB SXM5", 700, 36718.75),
    "LambdaOD_2x_Texas_1xH100_80GB_SXM5_192-222-52-65_v2": ([1, 1],  "LambdaOD 2x Texas 1xH100 80GB SXM5", 700, 36718.75),
    "LambdaOD_4x_Texas_1xH100_80GB_SXM5_192-222-52-139_v2": ([1, 1],  "LambdaOD 4x Texas 1xH100 80GB SXM5", 700, 36718.75),
    "LambdaOD_4x_Texas_1xH100_80GB_SXM5_192-222-52-178_v2": ([1, 1],  "LambdaOD 4x Texas 1xH100 80GB SXM5", 700, 36718.75),
    "LambdaOD_4x_Texas_1xH100_80GB_SXM5_192-222-54-151_v2": ([1, 1],  "LambdaOD 4x Texas 1xH100 80GB SXM5", 700, 36718.75),
    "LambdaOD_4x_Texas_1xH100_80GB_SXM5_192-222-54-254_v2": ([1, 1],  "LambdaOD 4x Texas 1xH100 80GB SXM5", 700, 36718.75),
    "LambdaOD_8x_Texas_1xH100_80GB_SXM5_192-222-52-149_v2": ([1, 1],  "LambdaOD 8x Texas 1xH100 80GB SXM5", 700, 36718.75),
    "LambdaOD_8x_Texas_1xH100_80GB_SXM5_192-222-52-89_v2": ([1, 1],  "LambdaOD 8x Texas 1xH100 80GB SXM5", 700, 36718.75),
    "LambdaOD_Texas_1xH100_80GB_SXM5_192-222-52-102_v2": ([1, 1],  "LambdaOD Texas 1xH100 80GB SXM5", 700, 36718.75),
    "LambdaOD_Texas_1xH100_80GB_SXM5_192-222-52-120_v2": ([1, 1],  "LambdaOD Texas 1xH100 80GB SXM5", 700, 36718.75),
    "LambdaOD_Texas_1xH100_80GB_SXM5_192-222-52-130_v2": ([1, 1],  "LambdaOD Texas 1xH100 80GB SXM5", 700, 36718.75),
    "LambdaOD_Texas_1xH100_80GB_SXM5_192-222-52-143_v2": ([1, 1],  "LambdaOD Texas 1xH100 80GB SXM5", 700, 36718.75),
    "LambdaOD_Texas_1xH100_80GB_SXM5_192-222-52-156_v2": ([1, 1],  "LambdaOD Texas 1xH100 80GB SXM5", 700, 36718.75),
    "LambdaOD_Texas_1xH100_80GB_SXM5_192-222-52-158_v2": ([1, 1],  "LambdaOD Texas 1xH100 80GB SXM5", 700, 36718.75),
    "LambdaOD_Texas_1xH100_80GB_SXM5_192-222-52-159_v2": ([1, 1],  "LambdaOD Texas 1xH100 80GB SXM5", 700, 36718.75),
    "LambdaOD_Texas_1xH100_80GB_SXM5_192-222-52-163_v2": ([1, 1],  "LambdaOD Texas 1xH100 80GB SXM5", 700, 36718.75),
    "LambdaOD_Texas_1xH100_80GB_SXM5_192-222-52-180_v2": ([1, 1],  "LambdaOD Texas 1xH100 80GB SXM5", 700, 36718.75),
    "LambdaOD_Texas_1xH100_80GB_SXM5_192-222-52-184_v2": ([1, 1],  "LambdaOD Texas 1xH100 80GB SXM5", 700, 36718.75),
    "LambdaOD_Texas_1xH100_80GB_SXM5_192-222-52-190_v2": ([1, 1],  "LambdaOD Texas 1xH100 80GB SXM5", 700, 36718.75),
    "LambdaOD_Texas_1xH100_80GB_SXM5_192-222-52-206_v2": ([1, 1],  "LambdaOD Texas 1xH100 80GB SXM5", 700, 36718.75),
    "LambdaOD_Texas_1xH100_80GB_SXM5_192-222-52-207_v2": ([1, 1],  "LambdaOD Texas 1xH100 80GB SXM5", 700, 36718.75),
    "LambdaOD_Texas_1xH100_80GB_SXM5_192-222-52-211_v2": ([1, 1],  "LambdaOD Texas 1xH100 80GB SXM5", 700, 36718.75),
    "LambdaOD_Texas_1xH100_80GB_SXM5_192-222-52-225_v2": ([1, 1],  "LambdaOD Texas 1xH100 80GB SXM5", 700, 36718.75),
    "LambdaOD_Texas_1xH100_80GB_SXM5_192-222-52-44_v2": ([1, 1],  "LambdaOD Texas 1xH100 80GB SXM5", 700, 36718.75),
    "LambdaOD_Texas_1xH100_80GB_SXM5_192-222-52-48_v2": ([1, 1],  "LambdaOD Texas 1xH100 80GB SXM5", 700, 36718.75),
    "LambdaOD_Texas_1xH100_80GB_SXM5_192-222-52-90_v2": ([1, 1],  "LambdaOD Texas 1xH100 80GB SXM5", 700, 36718.75),
    "LambdaOD_Texas_1xH100_80GB_SXM5_192-222-52-98_v2": ([1, 1],  "LambdaOD Texas 1xH100 80GB SXM5", 700, 36718.75),
    "LambdaOD_Texas_1xH100_80GB_SXM5_192-222-52-99_v2": ([1, 1],  "LambdaOD Texas 1xH100 80GB SXM5", 700, 36718.75),
}

list_system_multiple = {
    "LambdaBM_ENG04_8xH200_140GB_SXM_h200_v2": ([1, 8],  "LambdaBM ENG04 8xH200 140GB SXM", 700, 39375),
    "LambdaBM_ENG04_HTOff_8xH200_140GB_SXM_h200_v2": ([1, 8],  "LambdaBM ENG04 HTOff 8xH200 140GB SXM", 700, 39375),
    "LambdaOD_2x_Texas_2xH100_80GB_SXM5_192-222-52-120_v2": ([1, 2],  "LambdaOD 2x Texas 2xH100 80GB SXM5", 700, 36718.75),
    "LambdaOD_2x_Texas_2xH100_80GB_SXM5_192-222-52-176_v2": ([1, 2],  "LambdaOD 2x Texas 2xH100 80GB SXM5", 700, 36718.75),
    "LambdaOD_2x_Texas_2xH100_80GB_SXM5_192-222-52-211_v2": ([1, 2],  "LambdaOD 2x Texas 2xH100 80GB SXM5", 700, 36718.75),
    "LambdaOD_2x_Texas_2xH100_80GB_SXM5_192-222-52-65_v2": ([1, 2],  "LambdaOD 2x Texas 2xH100 80GB SXM5", 700, 36718.75),
    "LambdaOD_4x_Texas_2xH100_80GB_SXM5_192-222-52-139_v2": ([1, 2],  "LambdaOD 4x Texas 2xH100 80GB SXM5", 700, 36718.75),
    "LambdaOD_4x_Texas_2xH100_80GB_SXM5_192-222-52-178_v2": ([1, 2],  "LambdaOD 4x Texas 2xH100 80GB SXM5", 700, 36718.75),
    "LambdaOD_4x_Texas_2xH100_80GB_SXM5_192-222-54-151_v2": ([1, 2],  "LambdaOD 4x Texas 2xH100 80GB SXM5", 700, 36718.75),
    "LambdaOD_4x_Texas_2xH100_80GB_SXM5_192-222-54-254_v2": ([1, 2],  "LambdaOD 4x Texas 2xH100 80GB SXM5", 700, 36718.75),
    "LambdaOD_4x_Texas_4xH100_80GB_SXM5_192-222-52-139_v2": ([1, 4],  "LambdaOD 4x Texas 4xH100 80GB SXM5", 700, 36718.75),
    "LambdaOD_4x_Texas_4xH100_80GB_SXM5_192-222-52-178_v2": ([1, 4],  "LambdaOD 4x Texas 4xH100 80GB SXM5", 700, 36718.75),
    "LambdaOD_4x_Texas_4xH100_80GB_SXM5_192-222-54-151_v2": ([1, 4],  "LambdaOD 4x Texas 4xH100 80GB SXM5", 700, 36718.75),
    "LambdaOD_4x_Texas_4xH100_80GB_SXM5_192-222-54-254_v2": ([1, 4],  "LambdaOD 4x Texas 4xH100 80GB SXM5", 700, 36718.75),
    "LambdaOD_8x_Texas_2xH100_80GB_SXM5_192-222-52-149_v2": ([1, 2],  "LambdaOD 8x Texas 2xH100 80GB SXM5", 700, 36718.75),
    "LambdaOD_8x_Texas_2xH100_80GB_SXM5_192-222-52-89_v2": ([1, 2],  "LambdaOD 8x Texas 2xH100 80GB SXM5", 700, 36718.75),
    "LambdaOD_8x_Texas_4xH100_80GB_SXM5_192-222-52-149_v2": ([1, 4],  "LambdaOD 8x Texas 4xH100 80GB SXM5", 700, 36718.75),
    "LambdaOD_8x_Texas_4xH100_80GB_SXM5_192-222-52-89_v2": ([1, 4],  "LambdaOD 8x Texas 4xH100 80GB SXM5", 700, 36718.75),
    "LambdaOD_8x_Texas_8xH100_80GB_SXM5_192-222-52-149_v2": ([1, 8],  "LambdaOD 8x Texas 8xH100 80GB SXM5", 700, 36718.75),
    "LambdaOD_8x_Texas_8xH100_80GB_SXM5_192-222-52-89_v2": ([1, 8],  "LambdaOD 8x Texas 8xH100 80GB SXM5", 700, 36718.75),
    "LambdaOD_Texas_8xH100_80GB_SXM5_192-222-52-102_v2": ([1, 8],  "LambdaOD Texas 8xH100 80GB SXM5", 700, 36718.75),
    "LambdaOD_Texas_8xH100_80GB_SXM5_192-222-52-120_v2": ([1, 8],  "LambdaOD Texas 8xH100 80GB SXM5", 700, 36718.75),
    "LambdaOD_Texas_8xH100_80GB_SXM5_192-222-52-130_v2": ([1, 8],  "LambdaOD Texas 8xH100 80GB SXM5", 700, 36718.75),
    "LambdaOD_Texas_8xH100_80GB_SXM5_192-222-52-143_v2": ([1, 8],  "LambdaOD Texas 8xH100 80GB SXM5", 700, 36718.75),
    "LambdaOD_Texas_8xH100_80GB_SXM5_192-222-52-156_v2": ([1, 8],  "LambdaOD Texas 8xH100 80GB SXM5", 700, 36718.75),
    "LambdaOD_Texas_8xH100_80GB_SXM5_192-222-52-158_v2": ([1, 8],  "LambdaOD Texas 8xH100 80GB SXM5", 700, 36718.75),
    "LambdaOD_Texas_8xH100_80GB_SXM5_192-222-52-159_v2": ([1, 8],  "LambdaOD Texas 8xH100 80GB SXM5", 700, 36718.75),
    "LambdaOD_Texas_8xH100_80GB_SXM5_192-222-52-163_v2": ([1, 8],  "LambdaOD Texas 8xH100 80GB SXM5", 700, 36718.75),
    "LambdaOD_Texas_8xH100_80GB_SXM5_192-222-52-180_v2": ([1, 8],  "LambdaOD Texas 8xH100 80GB SXM5", 700, 36718.75),
    "LambdaOD_Texas_8xH100_80GB_SXM5_192-222-52-184_v2": ([1, 8],  "LambdaOD Texas 8xH100 80GB SXM5", 700, 36718.75),
    "LambdaOD_Texas_8xH100_80GB_SXM5_192-222-52-190_v2": ([1, 8],  "LambdaOD Texas 8xH100 80GB SXM5", 700, 36718.75),
    "LambdaOD_Texas_8xH100_80GB_SXM5_192-222-52-206_v2": ([1, 8],  "LambdaOD Texas 8xH100 80GB SXM5", 700, 36718.75),
    "LambdaOD_Texas_8xH100_80GB_SXM5_192-222-52-207_v2": ([1, 8],  "LambdaOD Texas 8xH100 80GB SXM5", 700, 36718.75),
    "LambdaOD_Texas_8xH100_80GB_SXM5_192-222-52-211_v2": ([1, 8],  "LambdaOD Texas 8xH100 80GB SXM5", 700, 36718.75),
    "LambdaOD_Texas_8xH100_80GB_SXM5_192-222-52-225_v2": ([1, 8],  "LambdaOD Texas 8xH100 80GB SXM5", 700, 36718.75),
    "LambdaOD_Texas_8xH100_80GB_SXM5_192-222-52-44_v2": ([1, 8],  "LambdaOD Texas 8xH100 80GB SXM5", 700, 36718.75),
    "LambdaOD_Texas_8xH100_80GB_SXM5_192-222-52-48_v2": ([1, 8],  "LambdaOD Texas 8xH100 80GB SXM5", 700, 36718.75),
    "LambdaOD_Texas_8xH100_80GB_SXM5_192-222-52-90_v2": ([1, 8],  "LambdaOD Texas 8xH100 80GB SXM5", 700, 36718.75),
    "LambdaOD_Texas_8xH100_80GB_SXM5_192-222-52-98_v2": ([1, 8],  "LambdaOD Texas 8xH100 80GB SXM5", 700, 36718.75),
    "LambdaOD_Texas_8xH100_80GB_SXM5_192-222-52-99_v2": ([1, 8],  "LambdaOD Texas 8xH100 80GB SXM5", 700, 36718.75),
}

list_test_fp32 = {
        "PyTorch_SSD_FP32": ("ssd", "^.*Average images/sec:.*$", -1),
        "PyTorch_resnet50_FP32": ("resnet50", "^.*Summary: train.loss.*$", 11),
        "PyTorch_gnmt_FP32": ("gnmt", "^.*Training:.*$", 4),
        "PyTorch_tacotron2_FP32": ("tacotron2", "^.*train_items_per_sec :.*$", -2),
        "PyTorch_waveglow_FP32": ("waveglow", "^.*train_items_per_sec :.*$", -2),
        "PyTorch_bert_large_squad_FP32": (
            "bert_large_squad",
            "^.*training_sequences_per_second :.*$",
            -6,
        ),
        "PyTorch_bert_base_squad_FP32": (
            "bert_base_squad",
            "^.*training_sequences_per_second :.*$",
            -6,
        ),
    }

list_test_fp16 = {
        "PyTorch_SSD_AMP": ("ssd", "^.*Average images/sec:.*$", -1),
        "PyTorch_resnet50_AMP": ("resnet50", "^.*Summary: train.loss.*$", 11),
        "PyTorch_gnmt_FP16": ("gnmt", "^.*Training:.*$", 4),
        "PyTorch_tacotron2_FP16": ("tacotron2", "^.*train_items_per_sec :.*$", -2),
        "PyTorch_waveglow_FP16": ("waveglow", "^.*train_items_per_sec :.*$", -2),
        "PyTorch_bert_large_squad_FP16": (
            "bert_large_squad",
            "^.*training_sequences_per_second :.*$",
            -6,
        ),
        "PyTorch_bert_base_squad_FP16": (
            "bert_base_squad",
            "^.*training_sequences_per_second :.*$",
            -6,
        ),
    }

def gather_throughput(
    list_test, list_system, name, system, config_name, df, version, path_result
):
    column_name, key, pos = list_test[name]
    pattern = re.compile(key)
    path = path_result + "/" + system + "/" + name
    count = 0.000000001
    total_throughput = 0.0

    if os.path.exists(path):
        for filename in os.listdir(path):
            if filename.endswith(".txt"):
                flag = False
                throughput = 0
                # Sift through all lines and only keep the last occurrence
                for i, line in enumerate(open(os.path.join(path, filename))):
                    for match in re.finditer(pattern, line):
                        if True:
                            print(match.group().split(' ')) # for debug
                        try:
                            throughput = float(match.group().split(" ")[pos])
                        except:
                            pass

                if throughput > 0:
                    count += 1
                    total_throughput += throughput
                    flag = True

                if not flag:
                    print(system + "/" + name + " " + filename + ": something wrong")
        df.at[config_name, column_name] = int(round(total_throughput / count, 2))
    else:
        df.at[config_name, column_name] = 0
    
    df.at[config_name, "num_gpu"] = list_system[system][0][1]
    df.at[config_name, "watt"] = list_system[system][2] * int(list_system[system][0][1])
    df.at[config_name, "price"] = list_system[system][3] * int(
        list_system[system][0][1]
    )

def gather_bs(
    list_test, list_system, name, system, config_name, df, version, path_result
):
    column_name, key, pos = list_test[name]
    path = path_result + "/" + system + "/" + name
    if os.path.exists(path):
        for filename in os.listdir(path):
            if filename.endswith(".para"):
                with open(os.path.join(path, filename)) as f:
                    first_line = f.readline()
                    df.at[config_name, column_name] = int(first_line.split(" ")[1])

    df.at[config_name, "num_gpu"] = list_system[system][0][1]
    df.at[config_name, "watt"] = list_system[system][2] * int(list_system[system][0][1])
    df.at[config_name, "price"] = list_system[system][3] * int(
        list_system[system][0][1]
    )

def main():
    parser = argparse.ArgumentParser(description="Gather benchmark results.")

    parser.add_argument(
        "--path", type=str, default="results_v2", help="path that has the results"
    )

    parser.add_argument(
        "--precision",
        type=str,
        default="fp32",
        choices=["fp32", "fp16"],
        help="Choose becnhmark precision",
    )

    parser.add_argument(
        "--system",
        type=str,
        default="all",
        choices=["single", "multiple", "all"],
        help="Choose system type (single or multiple GPUs)",
    )

    args = parser.parse_args()

    if args.precision == "fp32":
        list_test = list_test_fp32
    elif args.precision == "fp16":
        list_test = list_test_fp16
    else:
        sys.exit(
            "Wrong precision: " + args.precision + ", choose between fp32 and fp16"
        )
    
    if args.system == "single":
        list_system = list_system_single
    elif args.system == "multiple":
        list_system = list_system_multiple
    else:
        list_system = {}
        list_system.update(list_system_single)
        list_system.update(list_system_multiple)

    columns = []
    columns.append("num_gpu")
    columns.append("watt")
    columns.append("price")

    for test_name, value in sorted(list_test.items()):
        columns.append(list_test[test_name][0])
    list_configs = [list_system[key][1] for key in list_system]

    df_throughput = pd.DataFrame(index=list_configs, columns=columns)
    df_throughput = df_throughput.sort_index()

    df_throughput = df_throughput.fillna(-1.0)

    df_bs = pd.DataFrame(index=list_configs, columns=columns)

    for key in list_system:
        version = list_system[key][0][0]
        config_name = list_system[key][1]
        for test_name, value in sorted(list_test.items()):
            gather_throughput(
                list_test,
                list_system,
                test_name,
                key,
                config_name,
                df_throughput,
                version,
                args.path,
            )
            gather_bs(
                list_test,
                list_system,
                test_name,
                key,
                config_name,
                df_bs,
                version,
                args.path,
            )
    
    df_throughput.index.name = "name_gpu"
    df_throughput.to_csv("pytorch-train-throughput-v2-" + args.precision + ".csv")
    
    df_bs.index.name = "name_gpu"
    df_bs.to_csv("pytorch-train-bs-v2-" + args.precision + ".csv")



if __name__ == "__main__":
    main()
