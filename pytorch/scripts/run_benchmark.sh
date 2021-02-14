#!/bin/bash

SYSTEM=${1:-"2080Ti"}
TASK_NAME=${2:-"all"}

# pip install 'git+https://github.com/NVIDIA/dllogger'

cp /scripts/patch/run_squad.py examples/bert
cp /scripts/patch/multiproc.py examples/tacotron2
# cp /scripts/patch/run_squad.sh examples/bert/scripts
# cp /scripts/patch/box_encoder_cuda.cu examples/ssd/csrc


if [[ "${TASK_NAME}" == *"ssd"* ]] || [ $TASK_NAME = "all" ] || [[ "${TASK_NAME}" == *"maskrcnn"* ]]; then
	pushd .
	cd examples/ssd
	pip install .
	popd
fi

./run_system_pytorch.sh $SYSTEM

./run_benchmark_pytorch.sh $SYSTEM $TASK_NAME

