#!/bin/bash

mkdir -p ~/data/deeplearning-benchmark/tensorflow

docker run --gpus all --rm --shm-size=64g \
-v ~/data/deeplearning-benchmark/tensorflow:/data \
-v $(pwd)"/scripts":/scripts \
-v $(pwd)"/results":/results \
nvcr.io/nvidia/tensorflow:20.12-tf1-py3 \
/bin/bash -c "cp -r /scripts/* /workspace; ./run_benchmark.sh QuadroRTX8000 all"



docker run --gpus all --rm --shm-size=64g \
-v ~/data/deeplearning-benchmark/tensorflow:/data \
-v $(pwd)"/scripts":/scripts \
-v $(pwd)"/results":/results \
nvcr.io/nvidia/tensorflow:20.12-tf1-py3 \
/bin/bash -c "cp -r /scripts/* /workspace; ./run_benchmark.sh 2xTitanRTX all"