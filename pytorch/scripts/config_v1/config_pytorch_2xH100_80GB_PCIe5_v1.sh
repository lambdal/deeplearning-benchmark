#!/bin/bash

source config_v1/config_pytorch_2x80GB.sh

declare -A BATCH_SIZE_FIX=(
)

declare -A SSD_ITER_FIX=(
)

declare -A tacotron2_DATA_FIX=(
)

source config_v1/fix.sh
