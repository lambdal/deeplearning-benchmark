#!/bin/bash

docker run --gpus '"device=0"' --rm --shm-size=128g -v ~/data:/data -v $(pwd)"/scripts":/scripts -v $(pwd)"/results":/results nvcr.io/nvidia/pytorch:21.07-py3 /bin/bash -c "cp -r /scripts/* /workspace; ./run_benchmark.sh 80GB all 120"

