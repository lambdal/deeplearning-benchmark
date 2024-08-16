#!/bin/bash

source config_v2/config_pytorch_140GB.sh

declare -A BATCH_SIZE_FIX=(
)
source config_v2/fix.sh
