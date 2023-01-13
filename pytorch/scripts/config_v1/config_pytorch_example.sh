#!/bin/bash

source config_v1/config_pytorch_48GB.sh

declare -A BATCH_SIZE_FIX=(
    [PyTorch_SSD_FP32]=12
    [PyTorch_ncf_FP32]=121
    [PyTorch_bert_base_squad_FP32]=22
)

source config_v1/fix.sh
