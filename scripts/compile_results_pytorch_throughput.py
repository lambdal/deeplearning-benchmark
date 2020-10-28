import os
import sys
import re
import pandas as pd

path_result = 'results'

# Choose between 'fp32', 'fp16'
precision = 'fp16'

# Choose between 'single', 'multiple', 'all'
system = 'all'


# list_test_fp32 = {
#             'PyTorch_SSD_FP32': ('ssd_FP32', "^.*Training performance =.*$", -2),
#             'PyTorch_resnet50_FP32': ('resnet50_FP32', "^.*Summary: train.loss.*$", -2),
#             'PyTorch_maskrcnn_FP32': ('maskrcnn_FP32', "^.*Training perf is:.*$", -2),
#             'PyTorch_gnmt_FP32': ('gnmt_FP32', "^.*Training:.*$", -4),
#             'PyTorch_ncf_FP32': ('ncf_FP32', "^.*best_train_throughput:.*$", -1),
#             'PyTorch_transformerxlbase_FP32': ('transformerxlbase_FP32', "^.*Training throughput:.*$", -2),
#             'PyTorch_transformerxllarge_FP32': ('transformerxllarge_FP32', "^.*Training throughput:.*$", -2),
#             'PyTorch_tacotron2_FP32': ('tacotron2_FP32', "^.*train_epoch_avg_items/sec:.*$", -1),
#             'PyTorch_waveglow_FP32': ('waveglow_FP32', "^.*train_epoch_avg_items/sec:.*$", -1),
#             'PyTorch_bert_large_squad_FP32': ('bert_large_squad_FP32', "^.*training throughput:.*$", -1),
#             'PyTorch_bert_base_squad_FP32': ('bert_base_squad_FP32', "^.*training throughput:.*$", -1),
#              }

list_test_fp32 = {
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
             }

# list_test_fp16 = {
#             'PyTorch_SSD_AMP': ('ssd_AMP', "^.*Training performance =.*$", -2),
#             'PyTorch_resnet50_FP16': ('resnet50_FP16', "^.*Summary: train.loss.*$", -2),
#             'PyTorch_resnet50_AMP': ('resnet50_AMP', "^.*Summary: train.loss.*$", -2),
#             'PyTorch_maskrcnn_FP16': ('maskrcnn_FP16', "^.*Training perf is:.*$", -2),
#             'PyTorch_gnmt_FP16': ('gnmt_FP16', "^.*Training:.*$", -4),
#             'PyTorch_ncf_FP16': ('ncf_FP16', "^.*best_train_throughput:.*$", -1),
#             'PyTorch_transformerxlbase_FP16': ('transformerxlbase_FP16', "^.*Training throughput:.*$", -2),
#             'PyTorch_transformerxllarge_FP16': ('transformerxllarge_FP16', "^.*Training throughput:.*$", -2),
#             'PyTorch_tacotron2_FP16': ('tacotron2_FP16', "^.*train_epoch_avg_items/sec:.*$", -1),
#             'PyTorch_waveglow_FP16': ('waveglow_FP16', "^.*train_epoch_avg_items/sec:.*$", -1),
#             'PyTorch_bert_large_squad_FP16': ('bert_large_squad_FP16', "^.*training throughput:.*$", -1),
#             'PyTorch_bert_base_squad_FP16': ('bert_base_squad_FP16', "^.*training throughput:.*$", -1),
#              }


list_test_fp16 = {
            # 'PyTorch_SSD_AMP': ('ssd', "^.*Training performance =.*$", -2),
            # 'PyTorch_resnet50_FP16': ('resnet50', "^.*Summary: train.loss.*$", -2),
            # 'PyTorch_maskrcnn_FP16': ('maskrcnn', "^.*Training perf is:.*$", -2),
            # 'PyTorch_gnmt_FP16': ('gnmt', "^.*Training:.*$", -4),
            # 'PyTorch_ncf_FP16': ('ncf', "^.*best_train_throughput:.*$", -1),
            # 'PyTorch_transformerxlbase_FP16': ('transformerxlbase', "^.*Training throughput:.*$", -2),
            # 'PyTorch_transformerxllarge_FP16': ('transformerxllarge', "^.*Training throughput:.*$", -2),
            # 'PyTorch_tacotron2_FP16': ('tacotron2', "^.*train_items_per_sec :.*$", -2),
            # 'PyTorch_waveglow_FP16': ('waveglow', "^.*train_items_per_sec :.*$", -2),
            # 'PyTorch_bert_large_squad_FP16': ('bert_large_squad', "^.*training_sequences_per_second :.*$", -6),
            'PyTorch_bert_base_squad_FP16': ('bert_base_squad', "^.*training_sequences_per_second :.*$", -6),
             }


list_test_all = list_test_fp32.copy()
for key, value in list_test_fp16.items():
    list_test_all[key] = value


# list_system_single = {
#     'V100': 1,
#     'QuadroRTX8000': 1,
#     'QuadroRTX6000': 1,
#     'QuadroRTX5000': 1,
#     'TitanRTX': 1,
#     '2080Ti': 1,
#     '1080Ti': 1,
#     '2080SuperMaxQ': 1,
#     '2080MaxQ': 1, 
#     '2070MaxQ': 1
#     }
    

# list_system_multiple = {
#     '2x2080TiNVlink_trt': 2,
#     '2x2080TiNVlink_trt2': 2,
#     '2x2080Ti_trt': 2,
#     '2x2080Ti_trt2': 2,
#     '4x2080TiNVlink_trt': 4,
#     '4x2080TiNVlink_trt2': 4,
#     '4x2080Ti_trt': 4,
#     '4x2080Ti_trt2': 4,
#     '8x2080TiNVlink_trt': 8,
#     '8x2080TiNVlink_trt2': 8,
#     '8x2080Ti_trt': 8,
#     '8x2080Ti_trt2': 8,    
#     '2xQuadroRTX8000NVlink_trt': 2,
#     '2xQuadroRTX8000NVlink_trt2': 2,
#     '2xQuadroRTX8000_trt': 2,
#     '2xQuadroRTX8000_trt2': 2,
#     '4xQuadroRTX8000NVlink_trt': 4,
#     '4xQuadroRTX8000NVlink_trt2': 4,
#     '4xQuadroRTX8000_trt': 4,
#     '4xQuadroRTX8000_trt2': 4,
#     '8xQuadroRTX8000NVlink_trt': 8,
#     '8xQuadroRTX8000NVlink_trt2': 8,
#     '8xQuadroRTX8000_trt': 8,
#     '8xQuadroRTX8000_trt2': 8,
#     '2xV100': 2,
#     '4xV100': 4,
#     '8xV100': 8,
#     'LambdaCloud_4x1080Ti': 4,
#     'LambdaCloud_2xQuadroRTX6000': 2,
#     'LambdaCloud_4xQuadroRTX6000': 4,
#     'LambdaCloud_8xV10016G': 8,
#     'Linode_2xQuadroRTX6000': 2,
#     'p3.16xlarge': 8,
#     'p3.8xlarge': 4
# }

list_system_single = {
    '3080': 1
    }
    

list_system_multiple = {
    '2x3080': 2
}

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
    list_system = {} 
    list_system.update(list_system_single)
    list_system.update(list_system_multiple)


def gather_avg(name, system, df):
    
    column_name, key, pos = list_test[name]
    pattern = re.compile(key)

    path = path_result + '/' + system + '/' + name
    count = 0.000001
    total_throughput = 0.0
    if os.path.exists(path):
        for filename in os.listdir(path):
            if filename.endswith(".txt"):
                flag = False
                for i, line in enumerate(open(os.path.join(path, filename))):

                    for match in re.finditer(pattern, line):

                        try:
                            throughput = float(match.group().split(' ')[pos])
                            
                            if throughput > 0:
                                count += 1
                                total_throughput += throughput
                                flag = True
                            else:
                                pass

                        except:
                            pass

                if not flag:
                    print(system + "/" + name + " " + filename + ": something wrong")
        df.at[system, column_name] = int(round(total_throughput / count, 2))
    else:
        df.at[system, column_name] = 0
    df.at[system, 'num_gpu'] = list_system[system]


def gather_last(name, system, df):
    
    column_name, key, pos = list_test[name]
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
        df.at[system, column_name] = int(round(total_throughput / count, 2))
    else:
        df.at[system, column_name] = 0

    df.at[system, 'num_gpu'] = list_system[system]

def main():

    columns = []
    columns.append('num_gpu')
    for test_name, value in sorted(list_test.items()):
        columns.append(list_test[test_name][0])
    list_configs = [key for key in list_system]

    df = pd.DataFrame(index=list_configs, columns=columns)
    df = df.fillna(-1.0)

    for key in list_system:
        for test_name, value in sorted(list_test.items()):
            gather_last(test_name, key, df)

    df.index.name = 'name_gpu'

    df.to_csv('pytorch-train-throughput-' + precision + '.csv')

if __name__ == "__main__":
    main()

