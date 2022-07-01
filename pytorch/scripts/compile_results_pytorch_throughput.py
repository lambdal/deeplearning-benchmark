# -*- coding: utf-8 -*-
import os
import sys
import re
import argparse


import pandas as pd


# naming convention
# key: config name
# value: ([version, num_gpus], rename)
# version: 0 for pytorch:20.01-py3, and 1 for pytorch:20.10-py3
# num_gpus: sometimes num_gpus can't be inferred from config name (for example p3.16xlarge) or missing from the result log. So we ask for user to specify it here.
# rename: renaming the system so it is easier to read
# watt per gpu
# price per gpu

list_system_single = {
    'V100': ([0, 1], 'V100 32GB', 250, 11357),
    'QuadroRTX8000': ([0, 1], 'RTX 8000', 260, 6900),
    'QuadroRTX6000': ([0, 1], 'RTX 6000', 260, 4964),
    'QuadroRTX5000': ([0, 1], 'RTX 5000', 230, 2392),
    'TitanRTX': ([0, 1], 'Titan RTX', 280, 3500),
    '2080Ti': ([0, 1], 'RTX 2080Ti', 250, 1928),
    '1080Ti': ([0, 1], 'GTX 1080Ti', 250, 892),
    '2080SuperMaxQ': ([0, 1], 'RTX 2080 SUPER MAX-Q', 80, 1377),
    '2080MaxQ': ([0, 1], 'RTX 2080 MAX-Q', 90, 1032), 
    '2070MaxQ': ([0, 1], 'RTX 2070 MAX-Q', 90, 975),
    '3070': ([1, 1], 'RTX 3070', 220, 1035),
    '3080': ([1, 1], 'RTX 3080', 320, 1642),
    '3090': ([1, 1], 'RTX 3090', 350, 3142),
    '3090Ti': ([1, 1], 'RTX 3090Ti', 450, 1999),
    'A100_PCIe': ([1, 1], 'A100 40GB PCIe', 250, 12785),
    'A100_80GB_PCIe': ([1, 1], 'A100 80GB PCIe', 300, 15143),
    'A100_SXM4': ([1, 1], 'A100 40GB SXM4', 400, 14571),
    'A100_80GB_SXM4': ([1, 1], 'A100 80GB SXM4', 400, 18571),
    'A6000': ([1, 1], 'RTX A6000', 300, 5785),
    'A5000': ([1, 1], 'RTX A5000', 230, 2857),
    'LambdaCloud_A6000': ([1, 1], 'Lambda Cloud — RTX A6000', 300, 5785),
    '3080Max-Q': ([1, 1], 'RTX 3080 Max-Q', 80, 1600),
    'A40': ([1, 1], 'RTX A40', 300, 6464),
    'A4000': ([1, 1], 'RTX A4000', 140, 1321),
    'A4500': ([1, 1], 'RTX A4500', 200, 1928),
    'A5500': ([1, 1], 'RTX A5500', 230, 4500),
    }

list_system_multiple = {
    '2x2080TiNVlink_trt': ([0, 2], '2x RTX 2080Ti NVLink', 250, 1928),
    '2x2080Ti_trt': ([0, 2], '2x RTX 2080Ti', 250, 1928),
    '4x2080TiNVlink_trt': ([0, 4], '4x RTX 2080Ti NVLink', 250, 1928),
    '4x2080Ti_trt': ([0, 4], '4x RTX 2080Ti', 250, 1928),
    '8x2080TiNVlink_trt': ([0, 8], '8x RTX 2080Ti NVLink', 250, 1928),
    '8x2080Ti_trt': ([0, 8], '8x RTX 2080Ti', 250, 1928),
    '2xQuadroRTX8000NVlink_trt2': ([0, 2], '2x RTX 8000 NVLink', 260, 6900),
    '2xQuadroRTX8000_trt2': ([0, 2], '2x RTX 8000', 260, 6900),
    '4xQuadroRTX8000NVlink_trt2': ([0, 4], '4x RTX 8000 NVLink', 260, 6900),
    '4xQuadroRTX8000_trt2': ([0, 4], '4x RTX 8000', 260, 6900),
    '8xQuadroRTX8000NVlink_trt2': ([0, 8], '8x RTX 8000 NVLink', 260, 6900),
    '8xQuadroRTX8000_trt2': ([0, 8], '8x RTX 8000', 260, 6900),
    '2xV100': ([0, 2], '2x V100 32GB', 250, 11357),
    '4xV100': ([0, 4], '4x V100 32GB', 250, 11357),
    '8xV100': ([0, 8], '8x V100 32GB', 250, 11357),
    'p3.16xlarge': ([0, 8], 'p3.16xlarge', 300, 10664),
    'p3.8xlarge': ([0, 4], 'p3.8xlarge', 300, 10664),
    'LambdaCloud_8xV10016G': ([0, 8], 'Lambda Cloud — 8x V100 16GB', 300, 10664),
    'LambdaCloud_4x1080Ti': ([0, 4], 'Lambda Cloud — 4x GTX 1080Ti', 250, 892),
    'LambdaCloud_2xQuadroRTX6000': ([0, 2], 'Lambda Cloud — 2x RTX 6000', 260, 4964),
    'LambdaCloud_4xQuadroRTX6000': ([0, 4], 'Lambda Cloud — 4x RTX 6000', 260, 4964),
    'Linode_2xQuadroRTX6000': ([0, 2], 'Linode Cloud — 2x RTX 6000', 260, 4964),
    '2x3070': ([1, 2], '2x RTX 3070', 220, 1035),
    '4x3070': ([1, 4], '4x RTX 3070', 220, 1035),
    '8x3070': ([1, 8], '8x RTX 3070', 220, 1035),
    '2x3080': ([1, 2], '2x RTX 3080', 320, 1642),
    '2x3090': ([1, 2], '2x RTX 3090', 350, 3142),
    '3x3090': ([1, 3], '3x RTX 3090', 350, 3142),
    '4x3090': ([1, 4], '4x RTX 3090', 350, 3142),
    '8x3090': ([1, 8], '8x RTX 3090', 350, 3142),
    '2xA100_PCIe': ([1, 2], '2x A100 40GB PCIe', 250, 12785),
    '4xA100_PCIe': ([1, 4], '4x A100 40GB PCIe', 250, 12785),
    '8xA100_PCIe': ([1, 8], '8x A100 40GB PCIe', 250, 12785),
    '2xA100_80GB_PCIe': ([1, 2], '2x A100 80GB PCIe', 300, 15143),
    '2xA100_SXM4': ([1, 2], '2x A100 40GB SXM4', 400, 14571),
    '4xA100_SXM4': ([1, 4], '4x A100 40GB SXM4', 400, 14571),
    '8xA100_SXM4': ([1, 8], '8x A100 40GB SXM4', 400, 14571),
    '2xA100_80GB_SXM4': ([1, 2], '2x A100 80GB SXM4', 400, 18571),
    '4xA100_80GB_SXM4': ([1, 4], '4x A100 80GB SXM4', 400, 18571),
    '8xA100_80GB_SXM4': ([1, 8], '8x A100 80GB SXM4', 400, 18571),
    '8xA100_p4': ([1, 8], 'p4d.24xlarge', 400, 14571),
    '2xA6000': ([1, 2], '2x RTX A6000', 300, 5785),
    '4xA6000': ([1, 4], '4x RTX A6000', 300, 5785),
    '8xA6000': ([1, 8], '8x RTX A6000', 300, 5785),
    'LambdaCloud_2xA6000': ([1, 2], 'Lambda Cloud — 2x RTX A6000', 300, 5785),
    'LambdaCloud_4xA6000': ([1, 4], 'Lambda Cloud — 4x RTX A6000', 300, 5785),
    '4xA5000': ([1, 4], '4x RTX A5000', 230, 2857),
    '2xA5000': ([1, 2], '2x RTX A5000', 230, 2857),
    '2xA40': ([1, 2], '2x RTX A40', 300, 6464),
    '4xA40': ([1, 4], '4x RTX A40', 300, 6464),
    '8xA40': ([1, 8], '8x RTX A40', 300, 6464),
    '2xA4000': ([1, 2], '2x RTX A4000', 140, 1321),
    '4xA4000': ([1, 4], '4x RTX A4000', 140, 1321),
    '8xA4000': ([1, 8], '8x RTX A4000', 140, 1321),
    '16xA4000': ([1, 16], '16x RTX A4000', 140, 1321),
    '2xA4500': ([1, 2], '2x RTX A4500', 200, 1928),
    '4xA4500': ([1, 4], '4x RTX A4500', 200, 1928),
    '8xA4500': ([1, 8], '8x RTX A4500', 200, 1928),
    '2xA5500': ([1, 2], '2x RTX A5500', 230, 4500),
    '4xA5500': ([1, 4], '4x RTX A5500', 230, 4500),
    '8xA5500': ([1, 8], '8x RTX A5500', 230, 4500)
}

list_test_fp32 = [
            # nvcr.io/nvidia/pytorch:20.01-py3
            {
                'PyTorch_SSD_FP32': ('ssd', "^.*Training performance =.*$", -2),
                'PyTorch_resnet50_FP32': ('resnet50', "^.*Summary: train.loss.*$", -2),
                'PyTorch_maskrcnn_FP32': ('maskrcnn', "^.*Training perf is:.*$", -2),
                'PyTorch_gnmt_FP32': ('gnmt', "^.*Training:.*$", -4),
                'PyTorch_ncf_FP32': ('ncf', "^.*best_train_throughput:.*$", -1),
                'PyTorch_transformerxlbase_FP32': ('transformerxlbase', "^.*Training throughput:.*$", -2),
                'PyTorch_transformerxllarge_FP32': ('transformerxllarge', "^.*Training throughput:.*$", -2),
                'PyTorch_tacotron2_FP32': ('tacotron2', "^.*train_epoch_avg_items/sec:.*$", -1),
                'PyTorch_waveglow_FP32': ('waveglow', "^.*train_epoch_avg_items/sec:.*$", -1),
                'PyTorch_bert_large_squad_FP32': ('bert_large_squad', "^.*training throughput:.*$", -1),
                'PyTorch_bert_base_squad_FP32': ('bert_base_squad', "^.*training throughput:.*$", -1),
             },
            # nvcr.io/nvidia/pytorch:20.10-py3
            {
                'PyTorch_SSD_FP32': ('ssd', "^.*Training performance =.*$", -2),
                'PyTorch_resnet50_FP32': ('resnet50', "^.*Summary: train.loss.*$", -2),
                'PyTorch_maskrcnn_FP32': ('maskrcnn', "^.*Training perf is:.*$", -2),
                'PyTorch_gnmt_FP32': ('gnmt', "^.*Training:.*$", -4),
                'PyTorch_ncf_FP32': ('ncf', "^.*best_train_throughput:.*$", -1),
                'PyTorch_transformerxlbase_FP32': ('transformerxlbase', "^.*Training throughput:.*$", -2),
                'PyTorch_transformerxllarge_FP32': ('transformerxllarge', "^.*Training throughput:.*$", -2),
                'PyTorch_tacotron2_FP32': ('tacotron2', "^.*train_items_per_sec :.*$", -2),
                'PyTorch_waveglow_FP32': ('waveglow', "^.*train_items_per_sec :.*$", -2),
                'PyTorch_bert_large_squad_FP32': ('bert_large_squad', "^.*training_sequences_per_second :.*$", -6),
                'PyTorch_bert_base_squad_FP32': ('bert_base_squad', "^.*training_sequences_per_second :.*$", -6),
             }             
]

list_test_fp16 = [
        # version 0: nvcr.io/nvidia/pytorch:20.01-py3
        {
            'PyTorch_SSD_AMP': ('ssd', "^.*Training performance =.*$", -2),
            'PyTorch_resnet50_FP16': ('resnet50', "^.*Summary: train.loss.*$", -2),
            'PyTorch_maskrcnn_FP16': ('maskrcnn', "^.*Training perf is:.*$", -2),
            'PyTorch_gnmt_FP16': ('gnmt', "^.*Training:.*$", -4),
            'PyTorch_ncf_FP16': ('ncf', "^.*best_train_throughput:.*$", -1),
            'PyTorch_transformerxlbase_FP16': ('transformerxlbase', "^.*Training throughput:.*$", -2),
            'PyTorch_transformerxllarge_FP16': ('transformerxllarge', "^.*Training throughput:.*$", -2),
            'PyTorch_tacotron2_FP16': ('tacotron2', "^.*train_epoch_avg_items/sec:.*$", -1),
            'PyTorch_waveglow_FP16': ('waveglow', "^.*train_epoch_avg_items/sec:.*$", -1),
            'PyTorch_bert_large_squad_FP16': ('bert_large_squad', "^.*training throughput:.*$", -1),
            'PyTorch_bert_base_squad_FP16': ('bert_base_squad', "^.*training throughput:.*$", -1),
        },
        # version 1: nvcr.io/nvidia/pytorch:20.10-py3
        {
            'PyTorch_SSD_AMP': ('ssd', "^.*Training performance =.*$", 3),
            'PyTorch_resnet50_FP16': ('resnet50', "^.*Summary: train.loss.*$", -2),
            'PyTorch_maskrcnn_FP16': ('maskrcnn', "^.*Training perf is:.*$", -2),
            'PyTorch_gnmt_FP16': ('gnmt', "^.*Training:.*$", -4),
            'PyTorch_ncf_FP16': ('ncf', "^.*best_train_throughput:.*$", -1),
            'PyTorch_transformerxlbase_FP16': ('transformerxlbase', "^.*Training throughput:.*$", -2),
            'PyTorch_transformerxllarge_FP16': ('transformerxllarge', "^.*Training throughput:.*$", -2),
            'PyTorch_tacotron2_FP16': ('tacotron2', "^.*train_items_per_sec :.*$", -2),
            'PyTorch_waveglow_FP16': ('waveglow', "^.*train_items_per_sec :.*$", -2),
            'PyTorch_bert_large_squad_FP16': ('bert_large_squad', "^.*training_sequences_per_second :.*$", -6),
            'PyTorch_bert_base_squad_FP16': ('bert_base_squad', "^.*training_sequences_per_second :.*$", -6),
        }
]

def gather_last(list_test, list_system, name, system, config_name, df, version, path_result):
    column_name, key, pos = list_test[version][name]
    pattern = re.compile(key)

    path = path_result + '/' + system + '/' + name
    count = 0.000001
    total_throughput = 0.0

    if os.path.exists(path):
        for filename in os.listdir(path):
            if filename.endswith(".txt"):
                flag = False
                throughput = 0
                # Sift through all lines and only keep the last occurrence
                for i, line in enumerate(open(os.path.join(path, filename))):

                    for match in re.finditer(pattern, line):
                        try:
                            throughput = float(match.group().split(' ')[pos])
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

    df.at[config_name, 'num_gpu'] = list_system[system][0][1]
    df.at[config_name, 'watt'] = list_system[system][2] * int(list_system[system][0][1])
    df.at[config_name, 'price'] = list_system[system][3] * int(list_system[system][0][1])

def main():
    parser = argparse.ArgumentParser(description='Gather benchmark results.')

    parser.add_argument('--path', type=str, default='results',
                        help='path that has the results')    

    parser.add_argument('--precision', type=str, default='fp32',
                        choices=['fp32', 'fp16'],
                        help='Choose becnhmark precision')

    parser.add_argument('--system', type=str, default='all',
                        choices=['single', 'multiple', 'all'],
                        help='Choose system type (single or multiple GPUs)')

    args = parser.parse_args()

    if args.precision == 'fp32':
        list_test = list_test_fp32
    elif args.precision == 'fp16':
        list_test = list_test_fp16
    else:
        sys.exit("Wrong precision: " + precision + ', choose between fp32 and fp16')


    if args.system == 'single':
        list_system = list_system_single
    elif args.system == 'multiple':
        list_system = list_system_multiple
    else:
        list_system = {} 
        list_system.update(list_system_single)
        list_system.update(list_system_multiple)

    columns = []
    columns.append('num_gpu')
    columns.append('watt')
    columns.append('price')

    for test_name, value in sorted(list_test[0].items()):
        columns.append(list_test[0][test_name][0])
    list_configs = [list_system[key][1] for key in list_system]

    print(columns)
    print(list_configs)
    
    df = pd.DataFrame(index=list_configs, columns=columns)
    df = df.fillna(-1.0)

    for key in list_system:
        for test_name, value in sorted(list_test[0].items()):
            version = list_system[key][0][0]
            config_name = list_system[key][1]
            gather_last(list_test, list_system, test_name, key, config_name, df, version, args.path)

    df.index.name = 'name_gpu'

    df.to_csv('pytorch-train-throughput-' + args.precision + '.csv')

if __name__ == "__main__":
    main()

