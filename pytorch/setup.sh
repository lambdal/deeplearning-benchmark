#!/bin/bash
NAME_NGC=pytorch:22.10-py3

echo "---------------------------------------------------"
echo "Pull Docker Image: "
echo "---------------------------------------------------"
docker pull nvcr.io/nvidia/${NAME_NGC}


echo "---------------------------------------------------"
echo "Clone Repos: "
echo "---------------------------------------------------"
git clone https://github.com/LambdaLabsML/DeepLearningExamples.git && \
cd DeepLearningExamples && \
git checkout lambda/benchmark && \
cd ..
git clone https://github.com/lambdal/deeplearning-benchmark.git && \
cd deeplearning-benchmark/pytorch

echo "---------------------------------------------------"
echo "Prepare Data: "
echo "---------------------------------------------------"
docker run --gpus all --rm --shm-size=64g \
-v ~/DeepLearningExamples/PyTorch:/workspace/benchmark \
-v ~/data:/data \
-v $(pwd)"/scripts":/scripts \
nvcr.io/nvidia/${NAME_NGC} \
/bin/bash -c "cp -r /scripts/* /workspace;  ./run_prepare.sh"