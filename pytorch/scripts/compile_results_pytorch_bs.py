# -*- coding: utf-8 -*-
import os
import re
import argparse

import subprocess
from subprocess import Popen, PIPE
from os import environ

import pandas as pd

# You need add your own experiments here so it can be included in the generated csv files
# naming convention
# key: config name
# full name: renaming the system so it is easier to read
# watt per gpu
# price per gpu

list_system_single = [
   ('QuadroRTX8000_v1', 'Quadro RTX 8000 V1', 260, 6900, "v1"),
   ('4090_v1', 'RTX 4090', 450, 1599, "v1"),
   ('A100_40GB_PCIe_v1', 'A100 40GB PCIe', 250, 12785, "v1"),
   ('A6000_v1', 'RTX A6000', 300, 5785, "v1"),
]


list_system_multiple = [
]


# These are the rules to extract batch size from config files
list_test_fp32 = {
             'PyTorch_SSD_FP32': (4, -1, 1, 'ssd'),
             'PyTorch_resnet50_FP32': (7, -1, 1, 'resnet50'),
             'PyTorch_maskrcnn_FP32': (4, -1, 0, 'maskrcnn'),
             'PyTorch_gnmt_FP32': (4, -1, 1, 'gnmt'),
             'PyTorch_transformerxlbase_FP32': (5, -1, 0, 'transformerxlbase'),
             'PyTorch_transformerxllarge_FP32': (5, -1, 0, 'transformerxllarge'),
             'PyTorch_tacotron2_FP32': (7, -1, 1, 'tacotron2'),
             'PyTorch_waveglow_FP32': (8, -1, 1, 'waveglow'),
             'PyTorch_bert_large_squad_FP32': (5, -1, 1, 'bert_large_squad'),
             'PyTorch_bert_base_squad_FP32': (5, -1, 1, 'bert_base_squad'),
}

list_test_fp16 = {
             'PyTorch_SSD_AMP': (4, -1, 1, 'ssd'),
             'PyTorch_resnet50_AMP': (9, -1, 1, 'resnet50'),
             'PyTorch_gnmt_FP16': (4, -1, 1, 'gnmt'),
             'PyTorch_ncf_FP16': (5, -1, 0, 'ncf'),
             'PyTorch_transformerxlbase_FP16': (5, -1, 0, 'transformerxlbase'),
             'PyTorch_transformerxllarge_FP16': (5, -1, 0, 'transformerxllarge'),
             'PyTorch_tacotron2_FP16': (7, -1, 1, 'tacotron2'),
             'PyTorch_waveglow_FP16': (8, -1, 1, 'waveglow'),
             'PyTorch_bert_large_squad_FP16': (5, -1, 1, 'bert_large_squad'),
             'PyTorch_bert_base_squad_FP16': (5, -1, 1, 'bert_base_squad'),
}


def gather(list_test, key, name, df, path_config, watt, price, version):

    if version == "v0":
        f_name = os.path.join(path_config, 'config_v0/config_pytorch_' + key + '.sh')
        with open(f_name, 'r') as f:
            lines = f.readlines()

            idx_gpu = [i for i, s in enumerate(lines) if 'NUM_GPU=' in s]
            num_gpu = int(lines[idx_gpu[0]].rstrip().split("=")[1])

            for test_name, value in sorted(list_test.items()):
                idx = lines.index(test_name + "_PARAMS=(\n")
                line = lines[idx + value[0]].rstrip().split(" ")
                line = list(filter(lambda a: a != "", line))
                bs = int(line[value[1]][1:-1]) * (num_gpu if value[2] else 1)
                if bs == 1:
                    bs = 0
                df.at[name, value[3]] = bs        
    elif version == "v1": 
        f_name = os.path.join(path_config, 'config_v1/config_pytorch_' + key + '.sh')
        with open(f_name, 'r') as f:
            lines = f.readlines()
            
            f_template_name = [s for s in lines if "source" in s][0].split(" ")[1].strip()

            with open(os.path.join("scripts", f_template_name), 'r') as f_template:
                lines_template = f_template.readlines()
                
                idx_gpu = [i for i, s in enumerate(lines_template) if 'NUM_GPU=' in s]
                num_gpu = int(lines_template[idx_gpu[0]].rstrip().split("=")[1])

                for test_name, value in sorted(list_test.items()):

                    # check if lines has it
                    line = [s for s in lines if test_name in s]
                    if line:
                        bs = int(line[0].split("=")[1].rstrip())
                    else:
                        idx = lines_template.index(test_name + "_PARAMS=(\n")
                        line = lines_template[idx + value[0]].rstrip().split(" ")
                        line = list(filter(lambda a: a != "", line))
                        bs = int(line[value[1]][1:-1]) * (num_gpu if value[2] else 1)

                    if bs == 1:
                        bs = 0
                    df.at[name, value[3]] = bs

    df.at[name, 'num_gpu'] = num_gpu
    df.at[name, 'watt'] = watt * num_gpu
    df.at[name, 'price'] = price * num_gpu


def main():
    parser = argparse.ArgumentParser(description='Gather benchmark results.')

    parser.add_argument('--path', type=str, default='scripts',
                        help='path that has the results')    

    parser.add_argument('--precision', type=str, default='fp32',
                        choices=['fp32', 'fp16'],
                        help='Choose becnhmark precision')

    parser.add_argument('--system', type=str, default='all',
                        choices=['single', 'multiple', 'all'],
                        help='Choose system type (single or multiple GPUs)')

    args = parser.parse_args()

    list_test_all = list_test_fp32.copy()
    for key, value in list_test_fp16.items():
        list_test_all[key] = value

    if args.precision == 'fp32':
        list_test = list_test_fp32
    elif args.precision == 'fp16':
        list_test = list_test_fp16
    else:
        sys.exit("Wrong precision: " + args.precision + ', choose between fp32 and fp16')

    if args.system == 'single':
        list_system = list_system_single
    elif args.system == 'multiple':
        list_system = list_system_multiple
    else:
        list_system = list_system_single + list_system_multiple

    columns = []
    columns.append('num_gpu')
    columns.append('watt')
    columns.append('price')
    for test_name, value in sorted(list_test.items()):
        columns.append(value[3])

    df = pd.DataFrame(index=[i[1] for i in list_system], columns=columns)

    for s in list_system:
        key = s[0] 
        s_name = s[1]
        watt = s[2]
        price = s[3]
        version = s[4]
        gather(list_test, key, s_name, df, args.path, watt, price, version)

    df.index.name = 'name_gpu'

    df.to_csv('pytorch-train-bs-' + args.precision + '.csv')

if __name__ == "__main__":
    main()
