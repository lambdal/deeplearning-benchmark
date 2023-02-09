#!/bin/bash
NAME_NGC=${1:-"pytorch:22.10-py3"}

echo "---------------------------------------------------"
echo "Pull Docker Image: "
echo "---------------------------------------------------"
docker pull nvcr.io/nvidia/${NAME_NGC}


echo "---------------------------------------------------"
echo "Clone Repos: "
echo "---------------------------------------------------"
export TARGET_DIR=DeepLearningExamples
git -C "$TARGET_DIR" pull || git clone https://github.com/LambdaLabsML/${TARGET_DIR}.git "$TARGET_DIR" && \
cd DeepLearningExamples && \
git checkout lambda/benchmark && \
cd ..
export TARGET_DIR=deeplearning-benchmark
git -C "$TARGET_DIR" pull || git clone https://github.com/lambdal/${TARGET_DIR}.git "$TARGET_DIR" && \
cd deeplearning-benchmark/pytorch
