#!/bin/bash


# 1xGPU
docker run \
	--rm --shm-size=1024g \
	--gpus all \
	-v ~/DeepLearningExamples/PyTorch:/workspace/benchmark \
	-v ~/data:/data \
	-v $(pwd)"/scripts":/scripts \
	-v $(pwd)"/results":/results \
	nvcr.io/nvidia/${NAME_NGC} \
	/bin/bash -c "cp -r /scripts/* /workspace; ./run_benchmark.sh DellTexas_H100_80GB_SXM5_v1 all 3000"

# 2xGPU
docker run \
	--rm --shm-size=1024g \
	--gpus all \
	-v ~/DeepLearningExamples/PyTorch:/workspace/benchmark \
	-v ~/data:/data \
	-v $(pwd)"/scripts":/scripts \
	-v $(pwd)"/results":/results \
	nvcr.io/nvidia/${NAME_NGC} \
	/bin/bash -c "cp -r /scripts/* /workspace; ./run_benchmark.sh DellTexas_2xH100_80GB_SXM5_v1 all 3000"

# 4xGPU
docker run \
	--rm --shm-size=1024g \
	--gpus all \
	-v ~/DeepLearningExamples/PyTorch:/workspace/benchmark \
	-v ~/data:/data \
	-v $(pwd)"/scripts":/scripts \
	-v $(pwd)"/results":/results \
	nvcr.io/nvidia/${NAME_NGC} \
	/bin/bash -c "cp -r /scripts/* /workspace; ./run_benchmark.sh DellTexas_4xH100_80GB_SXM5_v1 all 3000"

# 4xGPU
docker run \
	--rm --shm-size=1024g \
	--gpus all \
	-v ~/DeepLearningExamples/PyTorch:/workspace/benchmark \
	-v ~/data:/data \
	-v $(pwd)"/scripts":/scripts \
	-v $(pwd)"/results":/results \
	nvcr.io/nvidia/${NAME_NGC} \
	/bin/bash -c "cp -r /scripts/* /workspace; ./run_benchmark.sh DellTexas_8xH100_80GB_SXM5_v1 all 3000"

