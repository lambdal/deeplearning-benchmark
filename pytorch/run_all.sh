#!/bin/bash

docker run \
	--rm --shm-size=1024g \
	--gpus all \
	-v ~/DeepLearningExamples/PyTorch:/workspace/benchmark \
	-v ~/data:/data \
	-v $(pwd)"/scripts":/scripts \
	-v $(pwd)"/results":/results \
	nvcr.io/nvidia/${NAME_NGC} \
	/bin/bash -c "cp -r /scripts/* /workspace; ./run_benchmark.sh H100_80GB_PCIe_v1 all 3000"

docker run \
	--rm --shm-size=1024g \
	--gpus all \
	-v ~/DeepLearningExamples/PyTorch:/workspace/benchmark \
	-v ~/data:/data \
	-v $(pwd)"/scripts":/scripts \
	-v $(pwd)"/results":/results \
	nvcr.io/nvidia/${NAME_NGC} \
	/bin/bash -c "cp -r /scripts/* /workspace; ./run_benchmark.sh 2xH100_80GB_PCIe_v1 all 3000"


#docker run \
#	--rm --shm-size=1024g \
#	--gpus all \
#	-v ~/DeepLearningExamples/PyTorch:/workspace/benchmark \
#	-v ~/data:/data \
#	-v $(pwd)"/scripts":/scripts \
#	-v $(pwd)"/results":/results \
#	nvcr.io/nvidia/${NAME_NGC} \
#	/bin/bash -c "cp -r /scripts/* /workspace; ./run_benchmark.sh 4xH100_80GB_PCIe_v1 all 3000"
#
#
#docker run \
#	--rm --shm-size=1024g \
#	--gpus all \
#	-v ~/DeepLearningExamples/PyTorch:/workspace/benchmark \
#	-v ~/data:/data \
#	-v $(pwd)"/scripts":/scripts \
#	-v $(pwd)"/results":/results \
#	nvcr.io/nvidia/${NAME_NGC} \
#	/bin/bash -c "cp -r /scripts/* /workspace; ./run_benchmark.sh 8xH100_80GB_PCIe_v1 all 3000"
