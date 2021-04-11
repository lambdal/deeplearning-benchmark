#!/bin/bash

#docker run --gpus all --rm --shm-size=64g -v ~/data:/data -v $(pwd)"/scripts":/scripts -v $(pwd)"/results":/results nvcr.io/nvidia/pytorch:20.10-py3 /bin/bash -c "cp -r /scripts/* /workspace; ./run_benchmark.sh 8x3090 waveglow"
#docker run --gpus all --rm --shm-size=64g -v ~/data:/data -v $(pwd)"/scripts":/scripts -v $(pwd)"/results":/results nvcr.io/nvidia/pytorch:20.10-py3 /bin/bash -c "cp -r /scripts/* /workspace; ./run_benchmark.sh 8x3090 tacotron2"

#docker run --gpus all --rm --shm-size=64g -v ~/data:/data -v $(pwd)"/scripts":/scripts -v $(pwd)"/results":/results nvcr.io/nvidia/pytorch:20.10-py3 /bin/bash -c "cp -r /scripts/* /workspace; ./run_benchmark.sh 4x3090 all"

#docker run --gpus all --rm --shm-size=64g -v ~/data:/data -v $(pwd)"/scripts":/scripts -v $(pwd)"/results":/results nvcr.io/nvidia/pytorch:20.10-py3 /bin/bash -c "cp -r /scripts/* /workspace; ./run_benchmark.sh 2x3070 all"

#docker run --gpus all --rm --shm-size=64g -v ~/data:/data -v $(pwd)"/scripts":/scripts -v $(pwd)"/results":/results nvcr.io/nvidia/pytorch:20.10-py3 /bin/bash -c "cp -r /scripts/* /workspace; ./run_benchmark.sh 3070 all"

#docker run --gpus all --rm --shm-size=64g -v ~/data:/data -v $(pwd)"/scripts":/scripts -v $(pwd)"/results":/results nvcr.io/nvidia/pytorch:20.10-py3 /bin/bash -c "cp -r /scripts/* /workspace; ./run_benchmark.sh LambdaCloud_A6000 resnet50"

#docker run --gpus all --rm --shm-size=64g -v ~/data:/data -v $(pwd)"/scripts":/scripts -v $(pwd)"/results":/results nvcr.io/nvidia/pytorch:20.10-py3 /bin/bash -c "cp -r /scripts/* /workspace; ./run_benchmark.sh LambdaCloud_2xA6000 all"

docker run --gpus all --rm --shm-size=64g -v ~/data:/data -v $(pwd)"/scripts":/scripts -v $(pwd)"/results":/results nvcr.io/nvidia/pytorch:20.10-py3 /bin/bash -c "cp -r /scripts/* /workspace; ./run_benchmark.sh LambdaCloud_4xA6000 resnet50"
