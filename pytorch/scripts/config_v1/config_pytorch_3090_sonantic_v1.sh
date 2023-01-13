#!/bin/bash

source config_v1/config_pytorch_24GB.sh

declare -A BATCH_SIZE_FIX=(
    [PyTorch_tacotron2_FP32]=48
    [PyTorch_tacotron2_FP16]=80
)
source config_v1/fix.sh
