#!/bin/bash

source config_v2/config_pytorch_140GB.sh

NUM_GPU=8
NUM_EXP=3

declare -A BATCH_SIZE_FIX=(
    [PyTorch_maskrcnn_FP32]=768
    [PyTorch_maskrcnn_FP16]=1024
    [PyTorch_ncf_FP32]=96000000
    [PyTorch_ncf_FP16]=96000000
    [PyTorch_transformerxlbase_FP32]=832
    [PyTorch_transformerxlbase_FP16]=1664
    [PyTorch_transformerxllarge_FP32]=384
    [PyTorch_transformerxllarge_FP16]=768
    [PyTorch_tacotron2_FP32]=256
    [PyTorch_tacotron2_FP16]=256
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

source config_v2/fix.sh
