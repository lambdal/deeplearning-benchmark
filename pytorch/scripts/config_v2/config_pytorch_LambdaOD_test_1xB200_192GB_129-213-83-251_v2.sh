#!/bin/bash

source config_v2/config_pytorch_192GB.sh

declare -A BATCH_SIZE_FIX=(
    [PyTorch_tacotron2_FP32]="filelists/ljs_audio_text_train_subset_1250_filelist.txt"
    [PyTorch_tacotron2_FP16]="filelists/ljs_audio_text_train_subset_1250_filelist.txt"
)
source config_v2/fix.sh