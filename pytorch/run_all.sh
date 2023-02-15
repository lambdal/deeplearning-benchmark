#!/bin/bash


docker run \
	--rm --shm-size=1024g \
	--gpus all \
	-v ~/DeepLearningExamples/PyTorch:/workspace/benchmark \
	-v ~/data:/data \
	-v $(pwd)"/scripts":/scripts \
	-v $(pwd)"/results":/results \
	nvcr.io/nvidia/${NAME_NGC} \
	/bin/bash -c "cp -r /scripts/* /workspace; ./run_benchmark.sh LambdaCloud_A10_v1 ssd 3000"

