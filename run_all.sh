#!/bin/bash

docker run --gpus all --rm --shm-size=64g -v ~/data:/data -v $(pwd)"/scripts":/scripts -v $(pwd)"/results":/results nvcr.io/nvidia/pytorch:20.10-py3 /bin/bash -c "cp -r /scripts/* /workspace; ./run_benchmark.sh 4xA6000 all"

docker run --gpus all --rm --shm-size=64g -v ~/data:/data -v $(pwd)"/scripts":/scripts -v $(pwd)"/results":/results nvcr.io/nvidia/pytorch:20.10-py3 /bin/bash -c "cp -r /scripts/* /workspace; ./run_benchmark.sh 2xA6000 all"

docker run --gpus all --rm --shm-size=64g -v ~/data:/data -v $(pwd)"/scripts":/scripts -v $(pwd)"/results":/results nvcr.io/nvidia/pytorch:20.10-py3 /bin/bash -c "cp -r /scripts/* /workspace; ./run_benchmark.sh A6000 all"
