# -*- coding: utf-8 -*-
import os
import re
import pandas as pd


# Choose between 'fp32', 'fp16'
precision = 'fp32'

# Choose between 'single', 'multiple', 'all'
system = 'all'

path_config = 'scripts/config'


list_system_single = [
   ('V100', (1, 'V100 32GB')),
   ('QuadroRTX8000', (1, 'RTX 8000')),
   ('QuadroRTX6000', (1, 'RTX 6000')),
   ('QuadroRTX5000', (1, 'RTX 5000')),
   ('TitanRTX', (1, 'Titan RTX')),
   ('2080Ti', (1, 'RTX 2080Ti')),
   ('1080Ti', (1, 'GTX 1080Ti')),
   ('2080SuperMaxQ', (1, 'RTX 2080 SUPER MAX-Q')),
   ('2080MaxQ', (1, 'RTX 2080 MAX-Q')),
   ('2070MaxQ', (1, 'RTX 2070 MAX-Q')),
   ('3080', (1, 'RTX 3080')),
   ('3090', (1, 'RTX 3090')),
   ('A100_PCIe', (1, 'A100 40GB PCIe')),
   ('A100_SXM4', (1, 'A100 40GB SXM4'))
   # ('A100_p4', 1)
]


list_system_multiple = [
    ('2x2080TiNVlink_trt', (2, '2x RTX 2080Ti NVLink')),
    # ('2x2080TiNVlink_trt2', 2),
    ('2x2080Ti_trt', (2, '2x RTX 2080Ti')),
    # ('2x2080Ti_trt2', 2),
    ('4x2080TiNVlink_trt', (4, '4x RTX 2080Ti NVLink')),
    # ('4x2080TiNVlink_trt2', 4),
    ('4x2080Ti_trt', (4, '4x RTX 2080Ti')),
    # ('4x2080Ti_trt2', 4),
    ('8x2080TiNVlink_trt', (8, '8x RTX 2080Ti NVLink')),
    # ('8x2080TiNVlink_trt2', 8),
    ('8x2080Ti_trt', (8, '8x RTX 2080Ti')),
    # ('8x2080Ti_trt2', 8),    
    # ('2xQuadroRTX8000NVlink_trt', 2),
    ('2xQuadroRTX8000NVlink_trt2', (2, '2x RTX 8000 NVLink')),
    # ('2xQuadroRTX8000_trt', 2),
    ('2xQuadroRTX8000_trt2', (2, '2x RTX 8000')),
    # ('4xQuadroRTX8000NVlink_trt', 4),
    ('4xQuadroRTX8000NVlink_trt2', (4, '4x RTX 8000 NVLink')),
    # ('4xQuadroRTX8000_trt', 4),
    ('4xQuadroRTX8000_trt2', (4, '4x RTX 8000')),
    # ('8xQuadroRTX8000NVlink_trt', 8),
    ('8xQuadroRTX8000NVlink_trt2', (8, '8x RTX 8000 NVLink')),
    # ('8xQuadroRTX8000_trt', 8),
    ('8xQuadroRTX8000_trt2', (8, '8x RTX 8000')),
    ('2xV100', (2, '2x V100 32GB')),
    ('4xV100', (4, '4x V100 32GB')),
    ('8xV100', (8, '8x V100 32GB')),
    ('LambdaCloud_4x1080Ti', (4, 'Lambda Cloud — 4x GTX 1080Ti')),
    ('LambdaCloud_2xQuadroRTX6000', (2, 'Lambda Cloud — 2x RTX 6000')),
    ('LambdaCloud_4xQuadroRTX6000', (4, 'Lambda Cloud — 4x RTX 6000')),
    ('LambdaCloud_8xV10016G', (8, 'Lambda Cloud — 8x V100 16GB')),
    ('Linode_2xQuadroRTX6000', (2, 'Linode Cloud — 2x RTX 6000')),
    ('p3.16xlarge', (8, 'p3.16xlarge')),
    ('p3.8xlarge', (4, 'p3.8xlarge')),
    ('2x3080', (2, '2x RTX 3080')),
    ('2x3090', (2, '2x RTX 3090')),
    ('3x3090', (3, '3x RTX 3090')),
    ('2xA100_PCIe', (2, '2x A100 40GB PCIe')),
    ('4xA100_PCIe', (4, '4x A100 40GB PCIe')),
    ('8xA100_PCIe', (8, '8x A100 40GB PCIe')),
    ('2xA100_SXM4', (2, '2x A100 40GB SXM4')),
    ('4xA100_SXM4', (4, '4x A100 40GB SXM4')),
    ('8xA100_SXM4', (8, '8x A100 40GB SXM4')),
    # ('2xA100_p4', 2),
    # ('4xA100_p4', 4),
    ('8xA100_p4', (8, 'p4d.24xlarge'))
]


# list_system_single = [
#    ('3080', 1)
# ]


# list_system_multiple = [
#     ('2x3080', 2)
# ]


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

list_test_all = list_test_fp32.copy()
for key, value in list_test_fp16.items():
    list_test_all[key] = value

if precision == 'fp32':
    list_test = list_test_fp32
elif precision == 'fp16':
    list_test = list_test_fp16
else:
    sys.exit("Wrong precision: " + precision + ', choose between fp32 and fp16')

if system == 'single':
    list_system = list_system_single
elif system == 'multiple':
    list_system = list_system_multiple
else:
    list_system = list_system_single + list_system_multiple

def gather(key, num_gpu, name, df):
    
    f_name = os.path.join(path_config, 'config_pytorch_' + key + '.sh')
    with open(f_name, 'r') as f:
        lines = f.readlines()

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
    columns = []
    columns.append('num_gpu')
    for test_name, value in sorted(list_test.items()):
        columns.append(value[3])

    df = pd.DataFrame(index=[i[1][1] for i in list_system], columns=columns)

    for s in list_system:
        key = s[0] 
        num_gpu = s[1][0]
        s_name = s[1][1]
        gather(key, num_gpu, s_name, df)
    df.index.name = 'name_gpu'

    df.to_csv('pytorch-train-bs-' + precision + '.csv')

if __name__ == "__main__":
    main()
