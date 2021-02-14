#!/bin/bash

docker run --gpus all --rm --shm-size=64g -v ~/data:/data -v $(pwd)"/scripts":/scripts -v $(pwd)"/results":/results nvcr.io/nvidia/tensorflow:20.10-tf1-py3 /bin/bash -c "cp -r /scripts/* /workspace; ./run_benchmark.sh 2xTitanRTX all"