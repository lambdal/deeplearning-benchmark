#!/bin/bash

source config_v1/config_pytorch_2x40GB.sh

declare -A BATCH_SIZE_FIX=(
)
source config_v1/fix.sh
