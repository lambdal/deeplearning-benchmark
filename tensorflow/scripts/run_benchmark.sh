#!/bin/bash

SYSTEM=${1:-"2080Ti"}
TASK_NAME=${2:-"all"}

./run_system_tensorflow.sh $SYSTEM

./run_benchmark_tensorflow.sh $SYSTEM $TASK_NAME