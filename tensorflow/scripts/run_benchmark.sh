#!/bin/bash

SYSTEM=${1:-"2080Ti"}
TASK_NAME=${2:-"all"}

cp -rf /workspace/nvidia-examples/ssdv1.2/models/research/object_detection /workspace/nvidia-examples/ssdv1.2
cp -f /scripts/patch/ssd320_bench.config /workspace/nvidia-examples/ssdv1.2/configs

./run_system_tensorflow.sh $SYSTEM

./run_benchmark_tensorflow.sh $SYSTEM $TASK_NAME