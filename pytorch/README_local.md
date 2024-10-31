# Guide

This is the guide to run benchmark on a Lambda Cloud On-demand instance.

```
# 1xGH200_96GB on Lambda Cloud On-demand
# The benchmark log files will be saved to deeplearning-benchmark/pytorch/results_v2/LambdaOD_test_1xGH200_96GB_$(hostname)_v2
export NAME_NGC=pytorch:24.10-py3
export NAME_TYPE=LambdaOD_test
export NAME_GPU=GH200_96GB
export NUM_GPU=1
export NAME_RESULTS=results_v2
export NAME_DATASET=all
export NAME_TASKS=all


# Remove sudo requirement for docker
sudo usermod -aG docker $USER && newgrp docker
docker pull nvcr.io/nvidia/${NAME_NGC}


# Clone repos
git clone https://github.com/LambdaLabsML/DeepLearningExamples.git && \
cd DeepLearningExamples && \
git checkout lambda/benchmark && \
git pull origin lambda/benchmark && \
cd ..

git clone https://github.com/lambdal/deeplearning-benchmark.git && \
cd deeplearning-benchmark/pytorch

# Prepare data
mkdir ~/data
docker run --gpus all --rm --shm-size=256g \
-v ~/DeepLearningExamples/PyTorch:/workspace/benchmark \
-v ~/data:/data \
-v $(pwd)"/scripts":/scripts \
nvcr.io/nvidia/${NAME_NGC} \
/bin/bash -c "cp -r /scripts/* /workspace;  ./run_prepare.sh $NAME_DATASET"

# Create benchmark config file
cp scripts/config_v2/config_pytorch_{NUM_GPU}x${NAME_GPU}_v2.sh scripts/config_v2/config_pytorch_${NAME_TYPE}_{NUM_GPU}x${NAME_GPU}_$(hostname)_v2.sh

# Run benchmark
mkdir -p ${NAME_RESULTS} && \
docker run --rm --shm-size=1024g \
--gpus all \
-v ~/DeepLearningExamples/PyTorch:/workspace/benchmark \
-v ~/data:/data \
-v $(pwd)"/scripts":/scripts \
-v $(pwd)/${NAME_RESULTS}:/results \
nvcr.io/nvidia/${NAME_NGC} \
/bin/bash -c "cp -r /scripts/* /workspace; ./run_benchmark.sh ${NAME_TYPE}_{NUM_GPU}x${NAME_GPU}_$(hostname)_v2 ${NAME_TASKS} 3000"
```
