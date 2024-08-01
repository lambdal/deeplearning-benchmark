#!/bin/bash

source config_v2/config_pytorch_140GB.sh

NUM_GPU=4
NUM_EXP=3

declare -A BATCH_SIZE_FIX=(
    [PyTorch_maskrcnn_FP32]=284
    [PyTorch_maskrcnn_FP16]=512
    [PyTorch_ncf_FP32]=64000000
    [PyTorch_ncf_FP16]=80000000
    [PyTorch_transformerxlbase_FP32]=416
    [PyTorch_transformerxlbase_FP16]=832
    [PyTorch_transformerxllarge_FP32]=192
    [PyTorch_transformerxllarge_FP16]=384
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
