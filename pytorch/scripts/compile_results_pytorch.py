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
    'QuadroRTX8000_v1': ([1, 1], 'Quadro RTX 8000', 260, 6037.5),
    'LambdaCloud_V100_16GB_v1': ([1, 1], 'LambdaCloud V100 16GB', 300, 9937.5),
    '3090_v1': ([1, 1], 'RTX 3090', 350, 2750),
    'LambdaCloud_A10_v1': ([1, 1], 'Lambda Cloud A10', 150, 3125),
    'LambdaCloud_A100_40GB_PCIe_v1': ([1, 1], 'Lambda Cloud A100 40GB PCIe', 250, 10437.50),
    # 'A100_40GB_PCIe_v1': ([1, 1], 'A100 40GB PCIe', 250, 12785),
    # 'A100_40GB_SXM4_v1': ([1, 1], 'A100 40GB SXM4', 250, 14571),
    'A100_80GB_PCIe_v1': ([1, 1], 'A100 80GB PCIe', 300, 13287.50),
    'A100_80GB_SXM4_v1': ([1, 1], 'A100 80GB SXM4', 400, 16250),
    'AdaA6000_v1': ([1, 1], 'Ada A6000', 300, 7680),
    'A6000_v1': ([1, 1], 'RTX A6000', 300, 4437.50),
    '4090_v1': ([1, 1], 'RTX 4090', 450, 2125),
    # 'H100_80GB_PCIe_v1': ([1, 1], 'H100 80GB PCIe Gen4', 310, 35714),
    'H100_80GB_PCIe5_v1': ([1, 1], 'H100 80GB PCIe Gen5', 310, 30918),
    }

list_system_multiple = {
    'LambdaCloud_2xV100_16GB_v1': ([1, 2], 'LambdaCloud 2x V100 16GB', 300, 9937.5),
    '2xAdaA6000_v1': ([1, 2], '2x Ada A6000', 300, 7680),
    '2xA6000_v1': ([1, 2], '2x RTX A6000', 300, 4437.50),
    '2x3090_v1': ([1, 2], '2x RTX 3090', 350, 2750),
    '2x4090_v1': ([1, 2], '2x RTX 4090', 450, 2125),
    # '2xA100_40GB_SXM4_v1': ([1, 2], '2xA100 40GB SXM4', 250, 14571),
    '2xA100_80GB_PCIe_v1': ([1, 2], '2x A100 80GB PCIe', 300, 13287.50),
    '2xA100_80GB_SXM4_v1': ([1, 2], '2x A100 80GB SXM4', 400, 16250),
    #'2xH100_80GB_PCIe_v1': ([1, 2], '2x H100 80GB PCIe Gen4', 310, 35714),
    '2xH100_80GB_PCIe5_v1': ([1, 2], '2x H100 80GB PCIe Gen5', 310, 30918),
    'LambdaCloud_4xV100_16GB_v1': ([1, 4], 'LambdaCloud 4xV100 16GB', 300, 9937.5),
    # '4xA100_40GB_SXM4_v1': ([1, 4], '4xA100 40GB SXM4', 250, 14571),
    #'4xA6000_v1': ([1, 4], '4x RTX A6000', 300, 5785),
    '4xA100_80GB_SXM4_v1': ([1, 4], '4x A100 80GB SXM4', 400, 16250),
    # '4xH100_80GB_PCIe_v1': ([1, 4], '4x H100 80GB PCIe Gen4', 310, 35714),
    '4xH100_80GB_PCIe5_v1': ([1, 4], '4x H100 80GB PCIe Gen5', 310, 30918),
    'LambdaCloud_8xV100_16GB_v1': ([1, 8], 'LambdaCloud 8xV100 16GB', 300, 9937.5),
    # '8xA100_40GB_SXM4_v1': ([1, 8], '8xA100 40GB SXM4', 250, 14571),
    '8xA100_80GB_SXM4_v1': ([1, 8], '8x A100 80GB SXM4', 400, 16250),
    # '8xH100_80GB_PCIe_v1': ([1, 8], '8x H100 80GB PCIe Gen4', 310, 35714),
    '8xH100_80GB_PCIe5_v1': ([1, 8], '8x H100 80GB PCIe Gen5', 310, 30918),
}
list_test_fp32 = [
        # version 0: nvcr.io/nvidia/pytorch:20.01-py3 and 20.10-py3
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
        # version 1: nvcr.io/nvidia/pytorch:22.09-py3
        {
            'PyTorch_SSD_FP32': ('ssd', "^.*Average images/sec:.*$", -1),
            'PyTorch_resnet50_FP32': ('resnet50', "^.*Summary: train.loss.*$", 11),
            'PyTorch_gnmt_FP32': ('gnmt', "^.*Training:.*$", 4),
            'PyTorch_ncf_FP32': ('ncf', "^.*best_train_throughput.*$", 7),
            'PyTorch_transformerxlbase_FP32': ('transformerxlbase', "^.*Training throughput:.*$", -2),
            'PyTorch_transformerxllarge_FP32': ('transformerxllarge', "^.*Training throughput:.*$", -2),
            'PyTorch_tacotron2_FP32': ('tacotron2', "^.*train_items_per_sec :.*$", -2),
            'PyTorch_waveglow_FP32': ('waveglow', "^.*train_items_per_sec :.*$", -2),
            'PyTorch_bert_large_squad_FP32': ('bert_large_squad', "^.*training_sequences_per_second :.*$", -6),
            'PyTorch_bert_base_squad_FP32': ('bert_base_squad', "^.*training_sequences_per_second :.*$", -6),
            }             
]

list_test_fp16 = [
        # version 0: nvcr.io/nvidia/pytorch:20.01-py3 and 20.10-py3
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
        # version 1: nvcr.io/nvidia/pytorch:22.09-py3
        {
            'PyTorch_SSD_AMP': ('ssd', "^.*Average images/sec:.*$", -1),
            'PyTorch_resnet50_AMP': ('resnet50', "^.*Summary: train.loss.*$", 11),
            'PyTorch_gnmt_FP16': ('gnmt', "^.*Training:.*$", 4),
            'PyTorch_ncf_FP16': ('ncf', "^.*best_train_throughput.*$", 7),
            'PyTorch_transformerxlbase_FP16': ('transformerxlbase', "^.*Training throughput:.*$", -2),
            'PyTorch_transformerxllarge_FP16': ('transformerxllarge', "^.*Training throughput:.*$", -2),
            'PyTorch_tacotron2_FP16': ('tacotron2', "^.*train_items_per_sec :.*$", -2),
            'PyTorch_waveglow_FP16': ('waveglow', "^.*train_items_per_sec :.*$", -2),
            'PyTorch_bert_large_squad_FP16': ('bert_large_squad', "^.*training_sequences_per_second :.*$", -6),
            'PyTorch_bert_base_squad_FP16': ('bert_base_squad', "^.*training_sequences_per_second :.*$", -6),
        }
]

def gather_throughput(list_test, list_system, name, system, config_name, df, version, path_result):
    column_name, key, pos = list_test[version][name]
    pattern = re.compile(key)
    path = path_result + '/' + system + '/' + name
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
                        # print(match.group().split(' ')) # for debug
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


def gather_bs(list_test, list_system, name, system, config_name, df, version, path_result):
    column_name, key, pos = list_test[version][name]
    path = path_result + '/' + system + '/' + name

    if os.path.exists(path):
        for filename in os.listdir(path):
            if filename.endswith(".para"):
                with open(os.path.join(path, filename)) as f:
                    first_line = f.readline()
                    df.at[config_name, column_name] = int(first_line.split(" ")[1])

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

    parser.add_argument('--version', type=int, default=1,
                        choices=[0, 1],
                        help='Choose benchmark version')

    args = parser.parse_args()

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
        list_system = {} 
        list_system.update(list_system_single)
        list_system.update(list_system_multiple)

    columns = []
    columns.append('num_gpu')
    columns.append('watt')
    columns.append('price')

    for test_name, value in sorted(list_test[args.version].items()):
        columns.append(list_test[args.version][test_name][0])
    list_configs = [list_system[key][1] for key in list_system]
    
    df_throughput = pd.DataFrame(index=list_configs, columns=columns)
    df_throughput = df_throughput.fillna(-1.0)

    df_bs = pd.DataFrame(index=list_configs, columns=columns)
    
    for key in list_system:
        version = list_system[key][0][0]
        config_name = list_system[key][1]
        for test_name, value in sorted(list_test[version].items()):
            gather_throughput(list_test, list_system, test_name, key, config_name, df_throughput, version, args.path)
            gather_bs(list_test, list_system, test_name, key, config_name, df_bs, version, args.path)

    df_throughput.index.name = 'name_gpu'
    df_throughput.to_csv('pytorch-train-throughput-' + args.precision + '.csv')

    df_bs.index.name = 'name_gpu'
    df_bs.to_csv('pytorch-train-bs-' + args.precision + '.csv')

if __name__ == "__main__":
    main()

