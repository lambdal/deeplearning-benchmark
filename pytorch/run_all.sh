#!/bin/bash

#docker run \
#	--rm --shm-size=1024g \
#	--gpus all \
#	-v ~/DeepLearningExamples/PyTorch:/workspace/benchmark \
#	-v ~/data:/data \
#	-v $(pwd)"/scripts":/scripts \
#	-v $(pwd)"/results":/results \
#	nvcr.io/nvidia/${NAME_NGC} \
#	/bin/bash -c "cp -r /scripts/* /workspace; ./run_benchmark.sh LambdaCloud_V100_16GB_v1 transformerxllarge 3000"
#
#docker run \
#	--rm --shm-size=1024g \
#	--gpus all \
#	-v ~/DeepLearningExamples/PyTorch:/workspace/benchmark \
#	-v ~/data:/data \
#	-v $(pwd)"/scripts":/scripts \
#	-v $(pwd)"/results":/results \
#	nvcr.io/nvidia/${NAME_NGC} \
#	/bin/bash -c "cp -r /scripts/* /workspace; ./run_benchmark.sh LambdaCloud_2xV100_16GB_v1 transformerxllarge 3000"
#
docker run \
	--rm --shm-size=1024g \
	--gpus all \
	-v ~/DeepLearningExamples/PyTorch:/workspace/benchmark \
	-v ~/data:/data \
	-v $(pwd)"/scripts":/scripts \
	-v $(pwd)"/results":/results \
	nvcr.io/nvidia/${NAME_NGC} \
	/bin/bash -c "cp -r /scripts/* /workspace; ./run_benchmark.sh LambdaCloud_4xV100_16GB_v1 transformerxllarge 3000"

docker run \
	--rm --shm-size=1024g \
	--gpus all \
	-v ~/DeepLearningExamples/PyTorch:/workspace/benchmark \
	-v ~/data:/data \
	-v $(pwd)"/scripts":/scripts \
	-v $(pwd)"/results":/results \
	nvcr.io/nvidia/${NAME_NGC} \
	/bin/bash -c "cp -r /scripts/* /workspace; ./run_benchmark.sh LambdaCloud_8xV100_16GB_v1 transformerxllarge 3000"
