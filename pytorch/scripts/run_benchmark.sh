#!/bin/bash

SYSTEM=${1:-"2080Ti"}
TASK_NAME=${2:-"all"}
TIME_OUT=${3:-"1800"}

pip install termcolor
pip install 'git+https://github.com/NVIDIA/dllogger'

cp /scripts/patch/run_squad.py benchmark/LanguageModeling/BERT
cp /scripts/patch/multiproc.py benchmark/SpeechSynthesis/Tacotron2


if [[ "${TASK_NAME}" == *"ssd"* ]] || [ $TASK_NAME = "all" ] || [[ "${TASK_NAME}" == *"maskrcnn"* ]]; then
	pushd .
	cd benchmark/Detection/SSD
	pip install .
	popd
fi

./run_system_pytorch.sh $SYSTEM

./run_benchmark_pytorch.sh $SYSTEM $TASK_NAME $TIME_OUT

python /scripts/check.py --path /results/${SYSTEM} |& tee /results/${SYSTEM}/summary.txt

