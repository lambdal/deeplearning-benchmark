#!/bin/bash


docker run --gpus all --rm --shm-size=64g -v ~/data:/data -v $(pwd)"/scripts":/scripts -v $(pwd)"/results":/results nvcr.io/nvidia/pytorch:21.03-py3 /bin/bash -c "cp -r /scripts/* /workspace; ./run_benchmark.sh A4000 gnmt"

docker run --gpus all --rm --shm-size=64g -v ~/data:/data -v $(pwd)"/scripts":/scripts -v $(pwd)"/results":/results nvcr.io/nvidia/pytorch:21.03-py3 /bin/bash -c "cp -r /scripts/* /workspace; ./run_benchmark.sh 2xA4000 gnmt"

#docker run --gpus all --rm --shm-size=64g -v ~/data:/data -v $(pwd)"/scripts":/scripts -v $(pwd)"/results":/results nvcr.io/nvidia/pytorch:21.03-py3 /bin/bash -c "cp -r /scripts/* /workspace; ./run_benchmark.sh 4xA4000 transformerxllarge"

#docker run --gpus all --rm --shm-size=64g -v ~/data:/data -v $(pwd)"/scripts":/scripts -v $(pwd)"/results":/results nvcr.io/nvidia/pytorch:21.03-py3 /bin/bash -c "cp -r /scripts/* /workspace; ./run_benchmark.sh 8xA4000 transformerxllarge"
