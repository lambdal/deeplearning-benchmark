# -*- coding: utf-8 -*-
import os
import sys
import re
import argparse


import pandas as pd


# naming convention
# key: config name
# value: ([version, num_gpus], rename)
# version: 0 for tensorflow:20.12-tf1-py3
# num_gpus: sometimes num_gpus can't be inferred from config name (for example p3.16xlarge) or missing from the result log. So we ask for user to specify it here.
# rename: renaming the system so it is easier to read

list_system_single = {
    'QuadroRTX8000': ([0, 1], 'RTX 8000'),
    }

list_system_multiple = {
    '2xTitanRTX': ([0, 2], '2x Titan RTX'),
}

list_test_fp32 = [
            # tensorflow:20.12-tf1-py3
            {
                'TensorFlow_resnet50_FP32': ('resnet50', "^.*train_throughput.*$", -2),
            },       
]

list_test_fp16 = [
        # version 0: tensorflow:20.12-tf1-py3
        {
            'TensorFlow_resnet50_FP16': ('resnet50', "^.*train_throughput.*$", -2),
        },
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
                            print(match.group().split(' '))
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
    for test_name, value in sorted(list_test[0].items()):
        columns.append(list_test[0][test_name][0])
    list_configs = [list_system[key][1] for key in list_system]

    df = pd.DataFrame(index=list_configs, columns=columns)
    df = df.fillna(-1.0)

    for key in list_system:
        for test_name, value in sorted(list_test[0].items()):
            version = list_system[key][0][0]
            config_name = list_system[key][1]
            gather_last(list_test, list_system, test_name, key, config_name, df, version, args.path)

    df.index.name = 'name_gpu'

    df.to_csv('tensorflow-train-throughput-' + args.precision + '.csv')

if __name__ == "__main__":
    main()

