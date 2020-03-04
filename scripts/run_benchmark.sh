#!/bin/bash

SYSTEM=${1:-"2080Ti"}

pip install 'git+https://github.com/NVIDIA/dllogger'

cp /scripts/patch/run_squad.py examples/bert
cp /scripts/patch/multiproc.py examples/tacotron2
cp /scripts/patch/run_squad.sh examples/bert/scripts

./run_system_pytorch.sh $SYSTEM

./run_benchmark_pytorch.sh $SYSTEM

