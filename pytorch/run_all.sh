#!/bin/bash

#docker run \
#	--rm --shm-size=128g \
#	--gpus all \
#	-v ~/DeepLearningExamples/PyTorch:/workspace/benchmark \
#	-v ~/data:/data \
#	-v $(pwd)"/scripts":/scripts \
#	-v $(pwd)"/results":/results \
#	nvcr.io/nvidia/${NAME_NGC} \
#	/bin/bash -c "cp -r /scripts/* /workspace; ./run_benchmark.sh A100_80GB_SXM4_v1 ncf 3000"

docker run \
	--rm --shm-size=128g \
	--gpus all \
	-v ~/DeepLearningExamples/PyTorch:/workspace/benchmark \
	-v ~/data:/data \
	-v $(pwd)"/scripts":/scripts \
	-v $(pwd)"/results":/results \
	nvcr.io/nvidia/${NAME_NGC} \
	/bin/bash -c "cp -r /scripts/* /workspace; ./run_benchmark.sh 2xA100_80GB_SXM4_v1 waveglow_fp32 3000"


#docker run \
#	--rm --shm-size=128g \
#	--gpus all \
#	-v ~/DeepLearningExamples/PyTorch:/workspace/benchmark \
#	-v ~/data:/data \
#	-v $(pwd)"/scripts":/scripts \
#	-v $(pwd)"/results":/results \
#	nvcr.io/nvidia/${NAME_NGC} \
#	/bin/bash -c "cp -r /scripts/* /workspace; ./run_benchmark.sh 4xA100_80GB_SXM4_v1 ncf 3000"
#
#
#docker run \
#	--rm --shm-size=128g \
#	--gpus all \
#	-v ~/DeepLearningExamples/PyTorch:/workspace/benchmark \
#	-v ~/data:/data \
#	-v $(pwd)"/scripts":/scripts \
#	-v $(pwd)"/results":/results \
#	nvcr.io/nvidia/${NAME_NGC} \
#	/bin/bash -c "cp -r /scripts/* /workspace; ./run_benchmark.sh 8xA100_80GB_SXM4_v1 ncf 3000"
