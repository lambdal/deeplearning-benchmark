#!/bin/bash


docker run --gpus all --rm --shm-size=64g -v ~/data:/data -v $(pwd)"/scripts":/scripts -v $(pwd)"/results":/results nvcr.io/nvidia/pytorch:21.03-py3 /bin/bash -c "cp -r /scripts/* /workspace; ./run_benchmark.sh A5000 all"

docker run --gpus all --rm --shm-size=64g -v ~/data:/data -v $(pwd)"/scripts":/scripts -v $(pwd)"/results":/results nvcr.io/nvidia/pytorch:21.03-py3 /bin/bash -c "cp -r /scripts/* /workspace; ./run_benchmark.sh 2xA5000 all"

docker run --gpus all --rm --shm-size=64g -v ~/data:/data -v $(pwd)"/scripts":/scripts -v $(pwd)"/results":/results nvcr.io/nvidia/pytorch:21.03-py3 /bin/bash -c "cp -r /scripts/* /workspace; ./run_benchmark.sh 4xA5000 all"
