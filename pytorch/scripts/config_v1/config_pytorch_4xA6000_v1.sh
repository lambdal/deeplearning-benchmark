#!/bin/bash

# Referencing a template of same amount of GPU memory
source config_v1/config_pytorch_4x48GB.sh

# Place holder for changes to the tempalte
declare -A BATCH_SIZE_FIX=(
)
source config_v1/fix_bs.sh
