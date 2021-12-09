#!/bin/bash


#docker run --gpus all --rm --shm-size=64g -v ~/data:/data -v $(pwd)"/scripts":/scripts -v $(pwd)"/results":/results nvcr.io/nvidia/pytorch:21.03-py3 /bin/bash -c "cp -r /scripts/* /workspace; ./run_benchmark.sh A4000 resnet50"

#docker run --gpus all --rm --shm-size=64g -v ~/data:/data -v $(pwd)"/scripts":/scripts -v $(pwd)"/results":/results nvcr.io/nvidia/pytorch:21.03-py3 /bin/bash -c "cp -r /scripts/* /workspace; ./run_benchmark.sh 2xA4000 resnet50"

#docker run --gpus all --rm --shm-size=64g -v ~/data:/data -v $(pwd)"/scripts":/scripts -v $(pwd)"/results":/results nvcr.io/nvidia/pytorch:21.03-py3 /bin/bash -c "cp -r /scripts/* /workspace; ./run_benchmark.sh 4x3090 resnet50"

#docker run --gpus all --rm --shm-size=64g -v ~/data:/data -v $(pwd)"/scripts":/scripts -v $(pwd)"/results":/results nvcr.io/nvidia/pytorch:21.03-py3 /bin/bash -c "cp -r /scripts/* /workspace; ./run_benchmark.sh 8x3090 resnet50"

#docker run --gpus all --rm --shm-size=64g -v ~/data:/data -v $(pwd)"/scripts":/scripts -v $(pwd)"/results":/results nvcr.io/nvidia/pytorch:21.03-py3 /bin/bash -c "cp -r /scripts/* /workspace; ./run_benchmark.sh 4xA100_80GB_SXM4 resnet"

#docker run --gpus all --rm --shm-size=64g -v ~/data:/data -v $(pwd)"/scripts":/scripts -v $(pwd)"/results":/results nvcr.io/nvidia/pytorch:21.03-py3 /bin/bash -c "cp -r /scripts/* /workspace; ./run_benchmark.sh 2xA100_80GB_SXM4 all"

#docker run --gpus all --rm --shm-size=64g -v ~/data:/data -v $(pwd)"/scripts":/scripts -v $(pwd)"/results":/results nvcr.io/nvidia/pytorch:21.03-py3 /bin/bash -c "cp -r /scripts/* /workspace; ./run_benchmark.sh A100_80GB_SXM4 all"

docker run --gpus '"device=2,3"' --rm --shm-size=128g -v ~/data:/data -v $(pwd)"/scripts":/scripts -v $(pwd)"/results":/results nvcr.io/nvidia/pytorch:21.03-py3 /bin/bash -c "cp -r /scripts/* /workspace; ./run_benchmark.sh 2xA100_80GB_PCIe all"

#docker run --gpus '"device=2,3"' --rm --shm-size=128g -v ~/data:/data -v $(pwd)"/scripts":/scripts -v $(pwd)"/results":/results nvcr.io/nvidia/pytorch:21.03-py3 /bin/bash -c "cp -r /scripts/* /workspace; ./run_benchmark.sh A100_80GB_PCIe all"
