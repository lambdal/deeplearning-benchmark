PATH_STORAGE ?= "/home/ubuntu"

NAME_NGC=pytorch:23.06-py3
NAME_DATASET=object_detection
NAME_CONFIG=StorageTest_24GB_v1
NAME_MODEL=ssd_amp
TIME_OUT=3000
PATH_OUTPUT=results/StorageTest_24GB_v1/PyTorch_SSD_AMP


install:
	docker pull nvcr.io/nvidia/${NAME_NGC} && \
	cd ${PATH_STORAGE} && \
	wget https://raw.githubusercontent.com/lambdal/deeplearning-benchmark/master/pytorch/setup.sh && \
	chmod +x setup.sh && \
	./setup.sh $NAME_NGC


create_data:
	cd ${PATH_STORAGE}/deeplearning-benchmark/pytorch && \
	docker run -u $USER --gpus all --rm -t --shm-size=128g \
		-v ${PATH_STORAGE}/DeepLearningExamples/PyTorch:/workspace/benchmark \
		-v ${PATH_STORAGE}/data:/data \
		-v ${PATH_STORAGE}/deeplearning-benchmark/pytorch/scripts:/scripts \
		nvcr.io/nvidia/${NAME_NGC} \
		/bin/bash -c "cp -r /scripts/* /workspace;  ./run_prepare.sh ${NAME_DATASET}"


benchmark:
	cd ${PATH_STORAGE}/deeplearning-benchmark/pytorch && \
	docker run \
		-u $USER \
		--rm --shm-size=128g \
		--gpus all \
		-v ${PATH_STORAGE}/DeepLearningExamples/PyTorch:/workspace/benchmark \
		-v ${PATH_STORAGE}/data:/data \
		-v ${PATH_STORAGE}/deeplearning-benchmark/pytorch/scripts:/scripts \
		-v ${PATH_STORAGE}/deeplearning-benchmark/pytorch/results:/results \
		nvcr.io/nvidia/${NAME_NGC} \
		/bin/bash -c "cp -r /scripts/* /workspace; ./run_benchmark.sh ${NAME_CONFIG} ${NAME_MODEL} ${TIME_OUT}" && \
	grep -r '^Training performance = ' ${PATH_STORAGE}/deeplearning-benchmark/pytorch/${PATH_OUTPUT}/*.txt

	
