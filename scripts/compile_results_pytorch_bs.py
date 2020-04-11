import os
import re
import pandas as pd


path_config = 'scripts/config'
#list_system = [('LambdaCloud_4x1080Ti', 4)] 


list_system = [('p3.8xlarge', 4),
               ('LambdaCloud_4x1080Ti', 4),
	       ('8xV100', 8), 
               ('4xV100', 4),
               ('2xV100', 2),
               ('V100', 1), 
               ('QuadroRTX8000', 1),
               ('QuadroRTX6000', 1),
               ('QuadroRTX5000', 1),
               ('TitanRTX', 1),
               ('2080Ti', 1),
               ('1080Ti', 1)] 

# skip lines, offset at the end, need to be multiply by #gpu
list_test = {
             'PyTorch_SSD_FP32': (4, -1, 1, 'SSD_FP32'),
             'PyTorch_SSD_AMP': (4, -1, 1, 'SSD_AMP'),
             'PyTorch_resnet50_FP32': (7, -1, 1, 'resnet50_FP32'),
             'PyTorch_resnet50_FP16': (9, -1, 1, 'resnet50_FP16'),
             'PyTorch_resnet50_AMP': (9, -1, 1, 'resnet50_AMP'),
             'PyTorch_maskrcnn_FP32': (4, -1, 0, 'maskrcnn_FP32'),
             'PyTorch_maskrcnn_FP16': (4, -1, 0, 'maskrcnn_FP16'),
             'PyTorch_gnmt_FP32': (4, -1, 1, 'gnmt_FP32'),
             'PyTorch_gnmt_FP16': (4, -1, 1, 'gnmt_FP16'),
             'PyTorch_ncf_FP32': (5, -1, 0, 'ncf_FP32'),
             'PyTorch_ncf_FP16': (5, -1, 0, 'ncf_FP16'),
             'PyTorch_transformerxlbase_FP32': (5, -1, 0, 'transformerxlbase_FP32'),
             'PyTorch_transformerxlbase_FP16': (5, -1, 0, 'transformerxlbase_FP16'),
             'PyTorch_transformerxllarge_FP32': (5, -1, 0, 'transformerxllarge_FP32'),
             'PyTorch_transformerxllarge_FP16': (5, -1, 0, 'transformerxllarge_FP16'),
             'PyTorch_tacotron2_FP32': (7, -1, 1, 'tacotron2_FP32'),
             'PyTorch_tacotron2_FP16': (7, -1, 1, 'tacotron2_FP16'),
             'PyTorch_waveglow_FP32': (8, -1, 1, 'waveglow_FP32'),
             'PyTorch_waveglow_FP16': (8, -1, 1, 'waveglow_FP16'),
             'PyTorch_bert_large_squad_FP32': (5, -1, 1, 'bert_large_squad_FP32'),
             'PyTorch_bert_large_squad_FP16': (5, -1, 1, 'bert_large_squad_FP16'),
             'PyTorch_bert_base_squad_FP32': (5, -1, 1, 'bert_base_squad_FP32'),
             'PyTorch_bert_base_squad_FP16': (5, -1, 1, 'bert_base_squad_FP16'),
             }


def gather(system, num_gpu, df):
    
    f_name = os.path.join(path_config, 'config_pytorch_' + system + '.sh')
    with open(f_name, 'r') as f:
        lines = f.readlines()

        for test_name, value in sorted(list_test.items()):
            idx = lines.index(test_name + "_PARAMS=(\n")
            line = lines[idx + value[0]].rstrip().split(" ")
            line = list(filter(lambda a: a != "", line))
            bs = int(line[value[1]][1:-1]) * (num_gpu if value[2] else 1)
            #print(system)
	    #print(test_name)
            df.at[system, test_name[8:]] = bs


def main():
    columns = []
    for test_name, value in sorted(list_test.items()):
        columns.append(value[3])
    print(columns)


    df = pd.DataFrame(index=[i[0] for i in list_system], columns=columns)
    # df = df.fillna(-1.0)
    #print(df)
    for (system, num_gpu) in list_system:
        gather(system, num_gpu, df)
    df.index.name = 'name_gpu'

    df.to_csv('pytorch_benchmark_bs.csv')

if __name__ == "__main__":
    main()
