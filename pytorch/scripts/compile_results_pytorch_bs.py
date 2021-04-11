# -*- coding: utf-8 -*-
import os
import re
import argparse


import pandas as pd


# You need add your own experiments here so it can be included in the generated csv files
# naming convention
# key: config name
# value: renaming the system so it is easier to read

list_system_single = [
   ('V100', 'V100 32GB'),
   ('QuadroRTX8000', 'RTX 8000'),
   ('QuadroRTX6000', 'RTX 6000'),
   ('QuadroRTX5000', 'RTX 5000'),
   ('TitanRTX', 'Titan RTX'),
   ('2080Ti', 'RTX 2080Ti'),
   ('1080Ti', 'GTX 1080Ti'),
   ('2080SuperMaxQ', 'RTX 2080 SUPER MAX-Q'),
   ('2080MaxQ', 'RTX 2080 MAX-Q'),
   ('2070MaxQ', 'RTX 2070 MAX-Q'),
   ('3070', 'RTX 3070'),
   ('3080', 'RTX 3080'),
   ('3090', 'RTX 3090'),
   ('A100_PCIe', 'A100 40GB PCIe'),
   ('A100_SXM4', 'A100 40GB SXM4'),
   ('A6000', 'RTX A6000'),
   ('LambdaCloud_A6000', 'Lambda Cloud — RTX A6000')
]


list_system_multiple = [
    ('2x2080TiNVlink_trt', '2x RTX 2080Ti NVLink'),
    ('2x2080Ti_trt', '2x RTX 2080Ti'),
    ('4x2080TiNVlink_trt', '4x RTX 2080Ti NVLink'),
    ('4x2080Ti_trt', '4x RTX 2080Ti'),
    ('8x2080TiNVlink_trt', '8x RTX 2080Ti NVLink'),
    ('8x2080Ti_trt', '8x RTX 2080Ti'),
    ('2xQuadroRTX8000NVlink_trt2', '2x RTX 8000 NVLink'),
    ('2xQuadroRTX8000_trt2', '2x RTX 8000'),
    ('4xQuadroRTX8000NVlink_trt2', '4x RTX 8000 NVLink'),
    ('4xQuadroRTX8000_trt2', '4x RTX 8000'),
    ('8xQuadroRTX8000NVlink_trt2', '8x RTX 8000 NVLink'),
    ('8xQuadroRTX8000_trt2', '8x RTX 8000'),
    ('2xV100', '2x V100 32GB'),
    ('4xV100', '4x V100 32GB'),
    ('8xV100', '8x V100 32GB'),
    ('LambdaCloud_4x1080Ti', 'Lambda Cloud — 4x GTX 1080Ti'),
    ('LambdaCloud_2xQuadroRTX6000', 'Lambda Cloud — 2x RTX 6000'),
    ('LambdaCloud_4xQuadroRTX6000', 'Lambda Cloud — 4x RTX 6000'),
    ('LambdaCloud_8xV10016G', 'Lambda Cloud — 8x V100 16GB'),
    ('Linode_2xQuadroRTX6000', 'Linode Cloud — 2x RTX 6000'),
    ('p3.16xlarge', 'p3.16xlarge'),
    ('p3.8xlarge', 'p3.8xlarge'),
    ('2x3070', '2x RTX 3070'),
    ('2x3080', '2x RTX 3080'),
    ('2x3090', '2x RTX 3090'),
    ('3x3090', '3x RTX 3090'),
    ('4x3070', '4x RTX 3070'),
    ('4x3090', '4x RTX 3090'),
    ('8x3070', '8x RTX 3070'),
    ('8x3090', '8x RTX 3090'),
    ('2xA100_PCIe', '2x A100 40GB PCIe'),
    ('4xA100_PCIe', '4x A100 40GB PCIe'),
    ('8xA100_PCIe', '8x A100 40GB PCIe'),
    ('2xA100_SXM4', '2x A100 40GB SXM4'),
    ('4xA100_SXM4', '4x A100 40GB SXM4'),
    ('8xA100_SXM4', '8x A100 40GB SXM4'),
    ('8xA6000', '8x RTX A6000'),
    ('4xA6000', '4x RTX A6000'),
    ('2xA6000', '2x RTX A6000'),
    ('8xA100_p4', 'p4d.24xlarge'),
    ('LambdaCloud_2xA6000', 'Lambda Cloud — 2x RTX A6000'),
    ('LambdaCloud_4xA6000', 'Lambda Cloud — 4x RTX A6000')
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

def gather(list_test, key, name, df, path_config):
    
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
    for test_name, value in sorted(list_test.items()):
        columns.append(value[3])

    df = pd.DataFrame(index=[i[1] for i in list_system], columns=columns)

    for s in list_system:
        key = s[0] 
        s_name = s[1]
        gather(list_test, key, s_name, df, args.path)

    df.index.name = 'name_gpu'

    df.to_csv('pytorch-train-bs-' + args.precision + '.csv')

if __name__ == "__main__":
    main()
