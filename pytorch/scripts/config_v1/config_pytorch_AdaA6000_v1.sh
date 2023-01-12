#!/bin/bash

source config_v1/config_pytorch_48GB.sh

declare -A BATCH_SIZE_FIX=(
)
source config_v1/fix.sh