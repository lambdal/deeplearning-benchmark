#!/bin/bash

source config_v1/config_pytorch_40GB.sh

NUM_GPU=8
NUM_EXP=1

declare -A BATCH_SIZE_FIX=(
    [PyTorch_maskrcnn_FP32]=192
    [PyTorch_maskrcnn_FP16]=256
    [PyTorch_ncf_FP32]=32000000
    [PyTorch_ncf_FP16]=40000000
    [PyTorch_transformerxlbase_FP32]=208
    [PyTorch_transformerxlbase_FP16]=416
    [PyTorch_transformerxllarge_FP32]=96
    [PyTorch_transformerxllarge_FP16]=192
)

declare -A SSD_ITER_FIX=(
)

declare -A tacotron2_DATA_FIX=(
)

declare -A BERT_GPU_FIX=(
    [PyTorch_bert_base_squad_FP32]=${NUM_GPU}
    [PyTorch_bert_base_squad_FP16]=${NUM_GPU}
    [PyTorch_bert_large_squad_FP32]=${NUM_GPU}
    [PyTorch_bert_large_squad_FP16]=${NUM_GPU}
)

source config_v1/fix.sh
