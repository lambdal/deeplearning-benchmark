#!/bin/bash

source config_v1/config_pytorch_48GB.sh

declare -A BATCH_SIZE_FIX=(
    [PyTorch_tacotron2_FP32]=48
    [PyTorch_tacotron2_FP16]=104
)
source config_v1/fix.sh
