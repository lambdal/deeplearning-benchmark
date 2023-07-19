#!/bin/bash

source config_v1/config_pytorch_24GB.sh

declare -A BATCH_SIZE_FIX=(
	[PyTorch_bert_base_squad_FP16]=80
	[PyTorch_bert_large_squad_FP32]=12
	[PyTorch_gnmt_FP32]=256
	[PyTorch_gnmt_FP16]=400
	[PyTorch_tacotron2_FP32]=80
	[PyTorch_tacotron2_FP16]=80
)
source config_v1/fix.sh
