#!/bin/bash

source config_v1/config_pytorch_16GB.sh

NUM_GPU=4
NUM_EXP=1

declare -A BATCH_SIZE_FIX=(
    [PyTorch_maskrcnn_FP32]=48
    [PyTorch_maskrcnn_FP16]=64
    [PyTorch_ncf_FP32]=5120000
    [PyTorch_ncf_FP16]=10240000
    [PyTorch_transformerxlbase_FP32]=32
    [PyTorch_transformerxlbase_FP16]=64
    [PyTorch_transformerxllarge_FP32]=8
    [PyTorch_transformerxllarge_FP16]=16
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
