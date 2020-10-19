import os
import re
import pandas as pd


# Choose between 'fp32', 'fp16'
precision = 'fp16'

# Choose between 'single', 'multiple', 'all'
system = 'all'

path_config = 'scripts/config'


list_system_single = [
   ('V100', 1),
   ('QuadroRTX8000', 1),
   ('QuadroRTX6000', 1),
   ('QuadroRTX5000', 1),
   ('TitanRTX', 1),
   ('2080Ti', 1),
   ('1080Ti', 1),
   ('2080SuperMaxQ', 1),
   ('2080MaxQ', 1),
   ('2070MaxQ', 1)
]


list_system_multiple = [
    ('2x2080TiNVlink_trt', 2),
    ('2x2080TiNVlink_trt2', 2),
    ('2x2080Ti_trt', 2),
    ('2x2080Ti_trt2', 2),
    ('4x2080TiNVlink_trt', 4),
    ('4x2080TiNVlink_trt2', 4),
    ('4x2080Ti_trt', 4),
    ('4x2080Ti_trt2', 4),
    ('8x2080TiNVlink_trt', 8),
    ('8x2080TiNVlink_trt2', 8),
    ('8x2080Ti_trt', 8),
    ('8x2080Ti_trt2', 8),    
    ('2xQuadroRTX8000NVlink_trt', 2),
    ('2xQuadroRTX8000NVlink_trt2', 2),
    ('2xQuadroRTX8000_trt', 2),
    ('2xQuadroRTX8000_trt2', 2),
    ('4xQuadroRTX8000NVlink_trt', 4),
    ('4xQuadroRTX8000NVlink_trt2', 4),
    ('4xQuadroRTX8000_trt', 4),
    ('4xQuadroRTX8000_trt2', 4),
    ('8xQuadroRTX8000NVlink_trt', 8),
    ('8xQuadroRTX8000NVlink_trt2', 8),
    ('8xQuadroRTX8000_trt', 8),
    ('8xQuadroRTX8000_trt2', 8),
    ('2xV100', 2),
    ('4xV100', 4),
    ('8xV100', 8),
    ('LambdaCloud_4x1080Ti', 4),
    ('LambdaCloud_2xQuadroRTX6000', 2),
    ('LambdaCloud_4xQuadroRTX6000', 4),
    ('LambdaCloud_8xV10016G', 8),
    ('Linode_2xQuadroRTX6000', 2),
    ('p3.16xlarge',8),
    ('p3.8xlarge', 4)
]

# list_test_fp32 = {
#              'PyTorch_SSD_FP32': (4, -1, 1, 'ssd_FP32'),
#              'PyTorch_resnet50_FP32': (7, -1, 1, 'resnet50_FP32'),
#              'PyTorch_maskrcnn_FP32': (4, -1, 0, 'maskrcnn_FP32'),
#              'PyTorch_gnmt_FP32': (4, -1, 1, 'gnmt_FP32'),
#              'PyTorch_ncf_FP32': (5, -1, 0, 'ncf_FP32'),
#              'PyTorch_transformerxlbase_FP32': (5, -1, 0, 'transformerxlbase_FP32'),
#              'PyTorch_transformerxllarge_FP32': (5, -1, 0, 'transformerxllarge_FP32'),
#              'PyTorch_tacotron2_FP32': (7, -1, 1, 'tacotron2_FP32'),
#              'PyTorch_waveglow_FP32': (8, -1, 1, 'waveglow_FP32'),
#              'PyTorch_bert_large_squad_FP32': (5, -1, 1, 'bert_large_squad_FP32'),
#              'PyTorch_bert_base_squad_FP32': (5, -1, 1, 'bert_base_squad_FP32'),
# }

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

# list_test_fp16 = {
#              'PyTorch_SSD_AMP': (4, -1, 1, 'ssd_AMP'),
#              'PyTorch_resnet50_FP16': (9, -1, 1, 'resnet50_FP16'),
#              'PyTorch_resnet50_AMP': (9, -1, 1, 'resnet50_AMP'),
#              'PyTorch_maskrcnn_FP16': (4, -1, 0, 'maskrcnn_FP16'),
#              'PyTorch_gnmt_FP16': (4, -1, 1, 'gnmt_FP16'),
#              'PyTorch_ncf_FP16': (5, -1, 0, 'ncf_FP16'),
#              'PyTorch_transformerxlbase_FP16': (5, -1, 0, 'transformerxlbase_FP16'),
#              'PyTorch_transformerxllarge_FP16': (5, -1, 0, 'transformerxllarge_FP16'),
#              'PyTorch_tacotron2_FP16': (7, -1, 1, 'tacotron2_FP16'),
#              'PyTorch_waveglow_FP16': (8, -1, 1, 'waveglow_FP16'),
#              'PyTorch_bert_large_squad_FP16': (5, -1, 1, 'bert_large_squad_FP16'),
#              'PyTorch_bert_base_squad_FP16': (5, -1, 1, 'bert_base_squad_FP16'),
# }

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

def gather(system, num_gpu, df):
    
    f_name = os.path.join(path_config, 'config_pytorch_' + system + '.sh')
    with open(f_name, 'r') as f:
        lines = f.readlines()

        for test_name, value in sorted(list_test.items()):
            idx = lines.index(test_name + "_PARAMS=(\n")
            line = lines[idx + value[0]].rstrip().split(" ")
            line = list(filter(lambda a: a != "", line))
            bs = int(line[value[1]][1:-1]) * (num_gpu if value[2] else 1)
            if bs == 1:
                bs = 0
            df.at[system, value[3]] = bs


def main():
    columns = []
    for test_name, value in sorted(list_test.items()):
        columns.append(value[3])

    df = pd.DataFrame(index=[i[0] for i in list_system], columns=columns)

    for (system, num_gpu) in list_system:
        gather(system, num_gpu, df)
    df.index.name = 'name_gpu'

    df.to_csv('pytorch-train-bs-' + precision + '.csv')

if __name__ == "__main__":
    main()
