import os
import re
import pandas as pd


path_result = 'results'

list_system = ['QuadroRTX8000', '2xQuadroRTX8000', '2xQuadroRTX8000NVlink', '4xQuadroRTX8000', '4xQuadroRTX8000NVlink', '8xQuadroRTX8000', '8xQuadroRTX8000NVlink']
list_test = {
             'PyTorch_SSD_FP32': ('SSD_FP32', "^.*Training performance =.*$", -2),
             'PyTorch_SSD_AMP': ('SSD_AMP', "^.*Training performance =.*$", -2),
             'PyTorch_resnet50_FP32': ('resnet50_FP32', "^.*Summary: train.loss.*$", -2),
             'PyTorch_resnet50_FP16': ('resnet50_FP16', "^.*Summary: train.loss.*$", -2),
             'PyTorch_resnet50_AMP': ('resnet50_AMP', "^.*Summary: train.loss.*$", -2),
             'PyTorch_maskrcnn_FP32': ('maskrcnn_FP32', "^.*Training perf is:.*$", -2),
             'PyTorch_maskrcnn_FP16': ('maskrcnn_FP16', "^.*Training perf is:.*$", -2),
             'PyTorch_gnmt_FP32': ('gnmt_FP32', "^.*Training:.*$", -4),
             'PyTorch_gnmt_FP16': ('gnmt_FP16', "^.*Training:.*$", -4),
             'PyTorch_ncf_FP32': ('ncf_FP32', "^.*best_train_throughput:.*$", -1),
             'PyTorch_ncf_FP16': ('ncf_FP16', "^.*best_train_throughput:.*$", -1),
             'PyTorch_transformerxlbase_FP32': ('transformerxlbase_FP32', "^.*Training throughput:.*$", -2),
             'PyTorch_transformerxlbase_FP16': ('transformerxlbase_FP16', "^.*Training throughput:.*$", -2),
             'PyTorch_transformerxllarge_FP32': ('transformerxllarge_FP32', "^.*Training throughput:.*$", -2),
             'PyTorch_transformerxllarge_FP16': ('transformerxllarge_FP16', "^.*Training throughput:.*$", -2),
             'PyTorch_tacotron2_FP32': ('tacotron2_FP32', "^.*train_epoch_avg_items/sec:.*$", -1),
             'PyTorch_tacotron2_FP16': ('tacotron2_FP16', "^.*train_epoch_avg_items/sec:.*$", -1),
             'PyTorch_waveglow_FP32': ('waveglow_FP32', "^.*train_epoch_avg_items/sec:.*$", -1),
             'PyTorch_waveglow_FP16': ('waveglow_FP16', "^.*train_epoch_avg_items/sec:.*$", -1),
             'PyTorch_bert_large_squad_FP32': ('bert_large_squad_FP32', "^.*training throughput:.*$", -1),
             'PyTorch_bert_large_squad_FP16': ('bert_large_squad_FP16', "^.*training throughput:.*$", -1),
             'PyTorch_bert_base_squad_FP32': ('bert_base_squad_FP32', "^.*training throughput:.*$", -1),
             'PyTorch_bert_base_squad_FP16': ('bert_base_squad_FP16', "^.*training throughput:.*$", -1),
             }

# list_system = ['p3.16xlarge', 'p3.8xlarge', 'LambdaCloud_8xV10016G', 'LambdaCloud_4x1080Ti', '8xV100', '4xV100', '2xV100', 'V100', 'QuadroRTX8000', 'QuadroRTX6000', 'QuadroRTX5000', 'TitanRTX', '2080Ti', '1080Ti'] 


# list_test = {
#              'PyTorch_SSD_FP32': ('SSD_FP32', "^.*Training performance =.*$", -2),
#              'PyTorch_SSD_AMP': ('SSD_AMP', "^.*Training performance =.*$", -2),
#              'PyTorch_resnet50_FP32': ('resnet50_FP32', "^.*Summary: train.loss.*$", -2),
#              'PyTorch_resnet50_FP16': ('resnet50_FP16', "^.*Summary: train.loss.*$", -2),
#              'PyTorch_resnet50_AMP': ('resnet50_AMP', "^.*Summary: train.loss.*$", -2),
#              'PyTorch_maskrcnn_FP32': ('maskrcnn_FP32', "^.*Training perf is:.*$", -2),
#              'PyTorch_maskrcnn_FP16': ('maskrcnn_FP16', "^.*Training perf is:.*$", -2),
#              'PyTorch_gnmt_FP32': ('gnmt_FP32', "^.*Training:.*$", -4),
#              'PyTorch_gnmt_FP16': ('gnmt_FP16', "^.*Training:.*$", -4),
#              'PyTorch_ncf_FP32': ('ncf_FP32', "^.*best_train_throughput:.*$", -1),
#              'PyTorch_ncf_FP16': ('ncf_FP16', "^.*best_train_throughput:.*$", -1),
#              'PyTorch_transformerxlbase_FP32': ('transformerxlbase_FP32', "^.*Training throughput:.*$", -2),
#              'PyTorch_transformerxlbase_FP16': ('transformerxlbase_FP16', "^.*Training throughput:.*$", -2),
#              'PyTorch_transformerxllarge_FP32': ('transformerxllarge_FP32', "^.*Training throughput:.*$", -2),
#              'PyTorch_transformerxllarge_FP16': ('transformerxllarge_FP16', "^.*Training throughput:.*$", -2),
#              'PyTorch_tacotron2_FP32': ('tacotron2_FP32', "^.*train_epoch_avg_items/sec:.*$", -1),
#              'PyTorch_tacotron2_FP16': ('tacotron2_FP16', "^.*train_epoch_avg_items/sec:.*$", -1),
#              'PyTorch_waveglow_FP32': ('waveglow_FP32', "^.*train_epoch_avg_items/sec:.*$", -1),
#              'PyTorch_waveglow_FP16': ('waveglow_FP16', "^.*train_epoch_avg_items/sec:.*$", -1),
#              'PyTorch_bert_large_squad_FP32': ('bert_large_squad_FP32', "^.*training throughput:.*$", -1),
#              'PyTorch_bert_large_squad_FP16': ('bert_large_squad_FP16', "^.*training throughput:.*$", -1),
#              'PyTorch_bert_base_squad_FP32': ('bert_base_squad_FP32', "^.*training throughput:.*$", -1),
#              'PyTorch_bert_base_squad_FP16': ('bert_base_squad_FP16', "^.*training throughput:.*$", -1),
#              }


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
                        

        df.at[system, column_name] = round(total_throughput / count, 2)
    else:
        df.at[system, column_name] = 0.0


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
        df.at[system, column_name] = round(total_throughput / count, 2)
    else:
        df.at[system, column_name] = 0.0

def main():

    columns = []
    for test_name, value in sorted(list_test.items()):
        columns.append(list_test[test_name][0])


    df = pd.DataFrame(index=list_system, columns=columns)
    df = df.fillna(-1.0)

    for system in list_system:
        for test_name, value in sorted(list_test.items()):
            gather_last(test_name, system, df)

    df.index.name = 'name_gpu'

    df.to_csv('pytorch_benchmark_throughput.csv')

if __name__ == "__main__":
    main()

