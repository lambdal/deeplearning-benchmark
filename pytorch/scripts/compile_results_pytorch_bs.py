# -*- coding: utf-8 -*-
import os
import re
import argparse


import pandas as pd


# You need add your own experiments here so it can be included in the generated csv files
# naming convention
# key: config name
# value: renaming the system so it is easier to read

# key, full name, Watt
list_system_single = [
   ('V100', 'V100 32GB', '250'),
   ('QuadroRTX8000', 'RTX 8000', '260'),
   ('QuadroRTX6000', 'RTX 6000', '260'),
   ('QuadroRTX5000', 'RTX 5000', '230'),
   ('TitanRTX', 'Titan RTX', '280'),
   ('2080Ti', 'RTX 2080Ti', '250'),
   ('1080Ti', 'GTX 1080Ti', '250'),
   ('2080SuperMaxQ', 'RTX 2080 SUPER MAX-Q', '80'),
   ('2080MaxQ', 'RTX 2080 MAX-Q', '90'),
   ('2070MaxQ', 'RTX 2070 MAX-Q', '90'),
   ('3070', 'RTX 3070', '220'),
   ('3080', 'RTX 3080', '320'),
   ('3090', 'RTX 3090', '350'),
   ('A100_PCIe', 'A100 40GB PCIe', '250'),
   ('A100_SXM4', 'A100 40GB SXM4', '400'),
   ('A6000', 'RTX A6000', '300'),
   ('A5000', 'RTX A5000', '230'),
   ('LambdaCloud_A6000', 'Lambda Cloud — RTX A6000', '300'),
   ('3080Max-Q', 'RTX 3080 Max-Q', '80'),
   ('A40', 'RTX A40', '300'),
   ('A4000', 'RTX A4000', '140'),
]


list_system_multiple = [
    ('2x2080TiNVlink_trt', '2x RTX 2080Ti NVLink', '500'),
    ('2x2080Ti_trt', '2x RTX 2080Ti', '500'),
    ('4x2080TiNVlink_trt', '4x RTX 2080Ti NVLink', '1000'),
    ('4x2080Ti_trt', '4x RTX 2080Ti', '1000'),
    ('8x2080TiNVlink_trt', '8x RTX 2080Ti NVLink', '2000'),
    ('8x2080Ti_trt', '8x RTX 2080Ti', '2000'),
    ('2xQuadroRTX8000NVlink_trt2', '2x RTX 8000 NVLink', '520'),
    ('2xQuadroRTX8000_trt2', '2x RTX 8000', '520'),
    ('4xQuadroRTX8000NVlink_trt2', '4x RTX 8000 NVLink', '1040'),
    ('4xQuadroRTX8000_trt2', '4x RTX 8000', '1040'),
    ('8xQuadroRTX8000NVlink_trt2', '8x RTX 8000 NVLink', '2080'),
    ('8xQuadroRTX8000_trt2', '8x RTX 8000', '2080'),
    ('2xV100', '2x V100 32GB', '500'),
    ('4xV100', '4x V100 32GB', '1000'),
    ('8xV100', '8x V100 32GB', '2000'),
    ('LambdaCloud_4x1080Ti', 'Lambda Cloud — 4x GTX 1080Ti', '1000'),
    ('LambdaCloud_2xQuadroRTX6000', 'Lambda Cloud — 2x RTX 6000', '520'),
    ('LambdaCloud_4xQuadroRTX6000', 'Lambda Cloud — 4x RTX 6000', '1040'),
    ('LambdaCloud_8xV10016G', 'Lambda Cloud — 8x V100 16GB', '2400'),
    ('Linode_2xQuadroRTX6000', 'Linode Cloud — 2x RTX 6000', '520'),
    ('p3.16xlarge', 'p3.16xlarge', '2400'),
    ('p3.8xlarge', 'p3.8xlarge', '1200'),
    ('2x3070', '2x RTX 3070', '440'),
    ('2x3080', '2x RTX 3080', '640'),
    ('2x3090', '2x RTX 3090', '700'),
    ('3x3090', '3x RTX 3090', '1050'),
    ('4x3070', '4x RTX 3070', '880'),
    ('4x3090', '4x RTX 3090', '1400'),
    ('8x3070', '8x RTX 3070', '1760'),
    ('8x3090', '8x RTX 3090', '2800'),
    ('2xA100_PCIe', '2x A100 40GB PCIe', '500'),
    ('4xA100_PCIe', '4x A100 40GB PCIe', '1000'),
    ('8xA100_PCIe', '8x A100 40GB PCIe', '2000'),
    ('2xA100_SXM4', '2x A100 40GB SXM4', '800'),
    ('4xA100_SXM4', '4x A100 40GB SXM4', '1600'),
    ('8xA100_SXM4', '8x A100 40GB SXM4', '3200'),
    ('8xA6000', '8x RTX A6000', '2400'),
    ('4xA6000', '4x RTX A6000', '1200'),
    ('2xA6000', '2x RTX A6000', '600'),
    ('4xA5000', '4x RTX A5000', '920'),
    ('2xA5000', '2x RTX A5000', '460'),
    ('8xA100_p4', 'p4d.24xlarge', '3200'),
    ('LambdaCloud_2xA6000', 'Lambda Cloud — 2x RTX A6000', '600'),
    ('LambdaCloud_4xA6000', 'Lambda Cloud — 4x RTX A6000', '1200'),
    ('8xA40', '8x RTX A40', '2400'),
    ('4xA40', '4x RTX A40', '1200'),
    ('2xA40', '2x RTX A40', '600'),
    ('8xA4000', '8x RTX A4000', '1120'),
    ('4xA4000', '4x RTX A4000', '560'),
    ('2xA4000', '2x RTX A4000', '280'),
]


# These are the rules to extract batch size from config files
list_test_fp32 = {
             'PyTorch_SSD_FP32': (4, -1, 1, 'ssd'),
             'PyTorch_resnet50_FP32': (7, -1, 1, 'resnet50'),
             'PyTorch_maskrcnn_FP32': (4, -1, 0, 'maskrcnn'),
             'PyTorch_gnmt_FP32': (4, -1, 1, 'gnmt'),
             'PyTorch_ncf_FP32': (5, -1, 0, 'ncf'),
             'PyTorch_transformerxlbase_FP32': (5, -1, 0, 'transformerxlbase'),
             'PyTorch_transformerxllarge_FP32': (5, -1, 0, 'transformerxllarge'),
             'PyTorch_tacotron2_FP32': (7, -1, 1, 'tacotron2'),
             'PyTorch_waveglow_FP32': (8, -1, 1, 'waveglow'),
             'PyTorch_bert_large_squad_FP32': (5, -1, 1, 'bert_large_squad'),
             'PyTorch_bert_base_squad_FP32': (5, -1, 1, 'bert_base_squad'),
}

list_test_fp16 = {
             'PyTorch_SSD_AMP': (4, -1, 1, 'ssd'),
             'PyTorch_resnet50_FP16': (9, -1, 1, 'resnet50'),
             'PyTorch_maskrcnn_FP16': (4, -1, 0, 'maskrcnn'),
             'PyTorch_gnmt_FP16': (4, -1, 1, 'gnmt'),
             'PyTorch_ncf_FP16': (5, -1, 0, 'ncf'),
             'PyTorch_transformerxlbase_FP16': (5, -1, 0, 'transformerxlbase'),
             'PyTorch_transformerxllarge_FP16': (5, -1, 0, 'transformerxllarge'),
             'PyTorch_tacotron2_FP16': (7, -1, 1, 'tacotron2'),
             'PyTorch_waveglow_FP16': (8, -1, 1, 'waveglow'),
             'PyTorch_bert_large_squad_FP16': (5, -1, 1, 'bert_large_squad'),
             'PyTorch_bert_base_squad_FP16': (5, -1, 1, 'bert_base_squad'),
}

def gather(list_test, key, name, df, path_config, watt):
    
    f_name = os.path.join(path_config, 'config_pytorch_' + key + '.sh')
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
    df.at[name, 'num_gpu'] = num_gpu
    df.at[name, 'watt'] = watt


def main():
    parser = argparse.ArgumentParser(description='Gather benchmark results.')

    parser.add_argument('--path', type=str, default='scripts/config',
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
    for test_name, value in sorted(list_test.items()):
        columns.append(value[3])

    df = pd.DataFrame(index=[i[1] for i in list_system], columns=columns)

    for s in list_system:
        key = s[0] 
        s_name = s[1]
        watt = s[2]
        gather(list_test, key, s_name, df, args.path, watt)

    df.index.name = 'name_gpu'

    df.to_csv('pytorch-train-bs-' + args.precision + '.csv')

if __name__ == "__main__":
    main()
