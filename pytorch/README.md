# PyTorch Benchmarks


This project provides a wrapper to run PyTorch benchmarks using NVidia's [Deep Learning Examples](https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch) repo. 

Reference numbers can be found on this [GPU benchmark](https://lambdalabs.com/gpu-benchmarks) website.


## Overview


The benchmarks are containerized. You need to install NVIDIA driver, docker, and nvidia-container-toolkit, and pull the [PyTroch NGC](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch) Docker image. 

All the benchmarks here are for single-node (single GPU or multiple GPUs). They do NOT work for multiple nodes.


(Update 2022.09) NVIDIA has stopped packaging [Deep Learning Examples](https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch) into their PyTorch NGC images since `pytorch:21.08-py3`. This repo is an effort towards maintaining the support of those examples in more recent PyTorch NGC.


## Quick Start

You can follow the following steps to benchmark a Ubuntu machine with NVIDIA GPU(s), docker, up-to-date NVIDIA driver and container runtime libraries. You can use [Lambda stack](https://lambdalabs.com/lambda-stack-deep-learning-software) to install the dependencies on a fresh Ubuntu machine.

### Prepare data

```
export NAME_NGC=pytorch:22.10-py3
export NAME_DATASET=all # Set to all to prepare all datasets. You can also select a particular dataset from the dataset list
sudo usermod -aG docker $USER
newgrp docker
wget https://raw.githubusercontent.com/lambdal/deeplearning-benchmark/new-guide/pytorch/setup.sh
chmod +x setup.sh
./setup.sh $NAME_NGC

docker run --gpus all --rm --shm-size=64g \
-v ~/DeepLearningExamples/PyTorch:/workspace/benchmark \
-v ~/data:/data \
-v $(pwd)"/scripts":/scripts \
nvcr.io/nvidia/${NAME_NGC} \
/bin/bash -c "cp -r /scripts/* /workspace;  ./run_prepare.sh $NAME_DATASET"
```

### Run benchmark
```
export NAME_NGC=pytorch:22.10-py3
export NAME_CONFIG=8xA100_40GB_SXM4_v1 # Select the configuration from deeplearning-benchmark/scripts/config_v1
export NAME_MODEL=all # Set to all for benchmark all the models. You can also select a particular model from the model list
export TIME_OUT=3000

docker run \
	--rm --shm-size=1024g \
	--gpus all \
	-v ~/DeepLearningExamples/PyTorch:/workspace/benchmark \
	-v ~/data:/data \
	-v $(pwd)"/scripts":/scripts \
	-v $(pwd)"/results":/results \
	nvcr.io/nvidia/${NAME_NGC} \
	/bin/bash -c "cp -r /scripts/* /workspace; ./run_benchmark.sh $NAME_CONFIG $NAME_MODEL $TIME_OUT"
```

### List of datasets
```
object_detection
gnmt
ncf
transformer
tacotron2
bert
```

### List of configurations
```
3090_v1
4090_v1
A100_40GB_PCIe_v1
A100_40GB_SXM4_v1
A100_80GB_PCIe_v1
A100_80GB_SXM4_v1
A6000_v1
AdaA6000_v1
H100_80GB_PCIe5_v1
LambdaCloud_A100_40GB_PCIe_v1
LambdaCloud_V100_16GB_v1
QuadroRTX8000_v1
2x3090_v1
2x4090_v1
2xA100_40GB_SXM4_v1
2xA100_80GB_PCIe_v1
2xA100_80GB_SXM4_v1
2xA6000_v1
2xAdaA6000_v1
2xH100_80GB_PCIe_v1
2xH100_80GB_PCIe5_v1
LambdaCloud_2xV100_16GB_v1
4x4090_v1
4xA100_40GB_SXM4_v1
4xA100_80GB_SXM4_v1
4xAdaA6000_v1
4xH100_80GB_PCIe_v1
4xH100_80GB_PCIe5_v1
LambdaCloud_4xV100_16GB_v1
8x4090_v1
8xA100_40GB_SXM4_v1
8xA100_80GB_SXM4_v1
8xAdaA6000_v1
8xH100_80GB_PCIe_v1
8xH100_80GB_PCIe5_v1
LambdaCloud_8xV100_16GB_v1
```

### List of models
```
ssd amp
ssd_fp32
bert_base_squad_fp16
bert_base_squad_fp32
bert_large_squad_fp16
bert_large_squad_fp32
gnmt_fp16
gnmt_fp32
ncf_fp16
ncf_fp32
resnet50_amp
resnet50_fp32
tacotron2_fp16
tacotron2_fp32
transformerxlbase_fp16
transformerxlbase_fp32
transformerxllarge_fp16
transformerxllarge_fp32
waveglow_fp16
waveglow_fp32
```

## Explanations
### Step 0: Prerequisite

This benchmark requires an Ubuntu machine with at least one GPU and up-to-date NVIDIA driver. Access to the internet.

If you start from a baremetal Ubuntu machine without NVIDIA driver. Install [Lambda stack](https://lambdalabs.com/lambda-stack-deep-learning-software). 

For workstation

```
wget -nv -O- https://lambdalabs.com/install-lambda-stack.sh | sh -
sudo reboot
```

For server
```
wget -nv -O- https://lambdalabs.com/install-lambda-stack.sh | I_AGREE_TO_THE_CUDNN_LICENSE=1 sh -
sudo reboot
```

### Step 1: Install Docker

The following steps install docker and NVIDIA's runtime libraries that enables GPUs for container. It also enables using docker without `sudo` (reboot is required):

```
sudo apt-get install docker.io nvidia-container-toolkit
sudo groupadd docker
sudo usermod -aG docker $USER
newgrp docker
```


### Step 2: Pull images

The latest tested PyTorch NGC is `pytorch:22.10-py3`.
```
export NAME_NGC=pytorch:22.10-py3
docker pull nvcr.io/nvidia/${NAME_NGC}
```

### Step 3: Clone Repo

```
# Lambda's fork of DeepLearningExamples (a few patches to make sure they work with the recent NGC)
git clone https://github.com/LambdaLabsML/DeepLearningExamples.git && \
cd DeepLearningExamples && \
git checkout lambda/benchmark && \
cd ..

# Clone this repo for streamlining the benchmark
git clone https://github.com/lambdal/deeplearning-benchmark.git && \
cd deeplearning-benchmark/pytorch
```

### Step 4: Prepare data

```
docker run --gpus all --rm --shm-size=64g \
-v ~/DeepLearningExamples/PyTorch:/workspace/benchmark \
-v ~/data:/data \
-v $(pwd)"/scripts":/scripts \
nvcr.io/nvidia/${NAME_NGC} \
/bin/bash -c "cp -r /scripts/* /workspace;  ./run_prepare.sh"
```

This step takes about 25 mins.

### Step 5: Prepare configuration files

Benchmark tasks are defined in a configuration file inside of `deeplearning-benchmark/pytorch/scripts/config_v1`. In short words, you need to 

* Create a new config file in that folder for your benchmark, either based on a pre-existing template, or from scratch (not recommended)
* Run the benchmark and reduce the batch size in the config file for tests that failed due to out of memory.

Here is an example of a customized benchmarking a single QuadroRTX8000. You can find it at `deeplearning-benchmark/pytorch/scripts/config_v1/config_pytorch_QuadroRTX8000_v1.sh`.

```
#!/bin/bash

# Referencing a template of same amount of GPU memory
source config_v1/config_pytorch_48GB.sh

# Place holder for changes to the tempalte
declare -A BATCH_SIZE_FIX=(
)
source config_v1/fix.sh
```

We will use this config later. Notice we added `v1` in the path and filename to diffferentiate it from some [old config files](https://github.com/lambdal/deeplearning-benchmark/tree/feat/timeout/pytorch/scripts/config_v0) that were created without referencing templates. 

The above cumstomized config file doesn't make any changes to the batch size set in the template. Here is an example of how to do it:

```
declare -A BATCH_SIZE_FIX=(
    [PyTorch_SSD_FP32]=12
    [PyTorch_ncf_FP32]=121
    [PyTorch_bert_base_squad_FP32]=22
)
```


The referenced [template](https://github.com/lambdal/deeplearning-benchmark/blob/feat/timeout/pytorch/scripts/config_v1/config_pytorch_48GB.sh) configures all the jobs for a single GPU with `48GB` of memory. Memory size is crucial here because it decides the batch size for different types of GPUs. The template file also specifies the number of GPUs, the number of experiments for each task, and the input arguments for individual task (SSD, ResNet, TransformerXL etc.)


The config file for multi-GPU benchmark uses the settings of a single-GPU config file, and allows specific changes to be added. For example, you can  customize the number of iterations for SSD (`--benchmark-iterations`) and the dataset for tacotron2/waveglow (`--training-files`). This is useful for benchmarking multi-gpu training with GPUs that have large memory (so to make sure the number of steps/dataset are enough to get valid results). Below is an example of how to set it in the config file:

```
declare -A SSD_ITER_FIX=(
    [PyTorch_SSD_FP32]=100
    [PyTorch_SSD_AMP]=100
)

declare -A tacotron2_DATA_FIX=(
    [PyTorch_tacotron2_FP32]="filelists/ljs_audio_text_train_subset_1250_filelist.txt"
    [PyTorch_tacotron2_FP16]="filelists/ljs_audio_text_train_subset_2500_filelist.txt"
    [PyTorch_waveglow_FP32]="filelists/ljs_audio_text_train_subset_1250_filelist.txt"
    [PyTorch_waveglow_FP16]="filelists/ljs_audio_text_train_subset_2500_filelist.txt"
)
```
The config file for multi-GPU benchmark should always include the following BERT customization. This is because the number of GPUs it is explicitly set as an argument of BERT training. It will stay as `1` (when the single-GPU config file is sourced), unless explictly overwritten in the multi-GPU config file.  

```
declare -A BERT_GPU_FIX=(
    [PyTorch_bert_base_squad_FP32]=${NUM_GPU}
    [PyTorch_bert_base_squad_FP16]=${NUM_GPU}
    [PyTorch_bert_large_squad_FP32]=${NUM_GPU}
    [PyTorch_bert_large_squad_FP16]=${NUM_GPU}
)
```

**Your customized config file should always reference a template that uses the SAME number of GPUs and the SAME GPU memory**. For example, referencing `config_pytorch_48GB.sh` in `config_pytorch_QuadroRTX8000_v1.sh` and `config_pytorch_A6000_v1.sh`, and referencing `config_pytorch_2x24GB.sh` in `config_pytorch_2x3090_v1.sh`.


### Step 6: Run Benchmark

Use `docker run` to execute the `run_benchmark.sh` script with the correct number of GPUs, paths for mounting the data, code and results, the name of the config file, and the task to run. Last but not the least, set a timeout limit to avoid being stuck at a task for too long.

Docker options:
* number of GPUs: by default use `--gpus all`, which let the docker container use all the GPUs in the machine. You can also choose specific ones, e.g. `--gpus '"device=2,3"'`, which only allow the container to use GPU2 and GPU3 in the machine. This can be helpful to benchmark a server wtih mixed type of GPUs.
* data path: `-v ~/data:/data` is the default path if your data is created using the above "Prepare data" command.
* code path: `-v $(pwd)"/scripts":/scripts` is the correct path and make sure you are inside of `deeplearning-benchmark/pytorch`
* result path: `-v $(pwd)"/results"` is the correct path and make sure you are inside of `deeplearning-benchmark/pytorch`

`run_benchmark_sh` options:
* config name: The name of the config file you want to call. For example, use`QuadroRTX8000_v1` to use `config_pytorch_QuadroRTX8000_v1.sh`
* task: use `all` for all the tasks defined in the config file (23 tasks in total). Or you can call specific task. For example, use `resnet50_fp32` (__all lower cases__) to only run a single task, or `resnet50` for all the resnet50 related task (fp32 and fp16)
* timeout limit: `600` is the default (in seconds). The benchmark will kill a task if it takes longer than this (e.g. weirdness when a GPU hangs in a multi-gpu job)

Here is the full command to benchmark the above `QuadroRTX8000_v1` config for all models, with 1500 secs timeout limit:

```
docker run \
--rm --shm-size=128g \
--gpus '"device=0"' \
-v ~/DeepLearningExamples/PyTorch:/workspace/benchmark \
-v ~/data:/data \
-v $(pwd)"/scripts":/scripts \
-v $(pwd)"/results":/results \
nvcr.io/nvidia/${NAME_NGC} \
/bin/bash -c "cp -r /scripts/* /workspace; ./run_benchmark.sh QuadroRTX8000_v1 all 1500"
```

If everything went well, you should see all the tasked are masked as successful at the end:

```
PyTorch_SSD_AMP                     :  sucessful
PyTorch_SSD_FP32                    :  sucessful
PyTorch_bert_base_squad_FP16        :  sucessful
PyTorch_bert_base_squad_FP32        :  sucessful
PyTorch_bert_large_squad_FP16       :  sucessful
PyTorch_bert_large_squad_FP32       :  sucessful
PyTorch_gnmt_FP16                   :  sucessful
PyTorch_gnmt_FP32                   :  sucessful
PyTorch_ncf_FP16                    :  sucessful
PyTorch_ncf_FP32                    :  sucessful
PyTorch_resnet50_AMP                :  sucessful
PyTorch_resnet50_FP32               :  sucessful
PyTorch_tacotron2_FP16              :  sucessful
PyTorch_tacotron2_FP32              :  sucessful
PyTorch_transformerxlbase_FP16      :  sucessful
PyTorch_transformerxlbase_FP32      :  sucessful
PyTorch_transformerxllarge_FP16     :  sucessful
PyTorch_transformerxllarge_FP32     :  sucessful
PyTorch_waveglow_FP16               :  sucessful
PyTorch_waveglow_FP32               :  sucessful
```

If any of these task failed, you can adjust the batch size in the customized config file, and re-run it. For example, you can re-run `PyTorch_resnet50_FP32` with this command

```
docker run \
--rm --shm-size=128g \
--gpus '"device=0"' \
-v ~/DeepLearningExamples/PyTorch:/workspace/benchmark \
-v ~/data:/data \
-v $(pwd)"/scripts":/scripts \
-v $(pwd)"/results":/results \
nvcr.io/nvidia/${NAME_NGC} \
/bin/bash -c "cp -r /scripts/* /workspace; ./run_benchmark.sh QuadroRTX8000_v1 resnet50_fp32 1500"
```

### Step 7: Gather Results

We provide some simply script to gather the results (everything in the results folder) to [CSV](https://github.com/lambdal/deeplearning-benchmark/blob/master/pytorch-train-throughput-fp32.csv) files for both training throughput and batch size.

```
python scripts/compile_results_pytorch.py --precision fp32 --system all
python scripts/compile_results_pytorch.py --precision fp16 --system all
```

To gather your own benchmarks, you need to add your system to the `list_system`. For example, this is the line you need to add to the `compile_results_pytorch.py` script:

```
# key: QuadroRTX8000_v1 as the config name
# value: ([version, num_gpus], rename)
# version: 0 for pytorch:20.01-py3, and 1 for pytorch:22.09-py3 and later
# num_gpus: sometimes num_gpus can't be inferred from config name (for example p3.16xlarge) or missing from the result log. So we ask user to specify it here.
# rename: renaming the system so it is easier to read (for releasing on our benchmark webpage)
# watt per gpu
# price per gpu

'QuadroRTX8000_v1': ([1, 1], 'Quadro RTX 8000 V1', 260, 6900),
```


See the [script](https://github.com/lambdal/deeplearning-benchmark/blob/master/pytorch/scripts/compile_results_pytorch.py) for details. 


## Notes

### Kill the benchmark 

```
ubuntu@ubuntu-desktop:~$ docker container ls
CONTAINER ID   IMAGE                              COMMAND                  CREATED          STATUS          PORTS                NAMES
e32cf156915c   nvcr.io/nvidia/pytorch:22.09-py3   "/usr/local/bin/nvid…"   27 seconds ago   Up 26 seconds   6006/tcp, 8888/tcp   elastic_austin


ubuntu@ubuntu-desktop:~$ docker top e32cf156915c
UID                 PID                 PPID                C                   STIME               TTY                 TIME                CMD
root                224082              224056              0                   09:38               ?                   00:00:00            /bin/bash -c cp -r /scripts/* /workspace; ./run_benchmark.sh QuadroRTX8000_v1 all 600

ubuntu@ubuntu-desktop:~$ sudo kill -9 224082
```

### Batch size 
Here are some pitfalls about creating benchmarks (more precisely, setting the input arguments for tasks in the config files).

Different models have different way to set batch size -- some of them are set for per-gpu, others are global. For the case of global, one should scale batch size by `num_gpu` for multiple GPU training.

`SSD` and `Tactron` needs some special settings for number of iterations or dataset size.


| Model | Batch | 
|---|---|
| PyTorch SSD  | Per GPU. `benchmark-iterations` needs to be reduced for more GPUs or larger batch size (otherwise GPU hang at 100%)  |
| PyTorch ResNet  | Per GPU.  |
| PyTorch MaskRCNN  | Global. Need to be scaled by `num_gpu` |
| PyTorch GNMT | Per GPU. |
| PyTorch NCF | Global. Need to be scaled by `num_gpu`|
| PyTorch TransformerXL | Global. Need to be scaled by `num_gpu` |
| PyTorch Tactron | Per GPU. Choose "subset_2500", "subset_1250", and "subset_625" based on number of GPUs|
| Bert | Per GPU. |


### Real ImageNet data for resnet50
You can use synthetic data or real data to benchmark ResNet. To run benchmark with synthetic data, simply add `--data-backend syntetic` to the [config file](https://github.com/lambdal/deeplearning-benchmark/blob/master/pytorch/scripts/config/config_pytorch_2xA100_p4.sh#L38) (right, there is a typo in NVidia's code).

If you want to benchmark ResNet with real data (we don't often do this unless want to stress test system I/O), here are the steps assuming `ILSVRC2012_img_train.tar` and `ILSVRC2012_img_val.tar` have already been downloaded to your home directory.

```
cd
mkdir -p data/imagenet && cd data/imagenet
mkdir train && cd train 
tar -xvf ~/ILSVRC2012_img_train.tar 
find . -name "*.tar" | while read NAME ; do mkdir -p "${NAME%.tar}"; tar -xvf "${NAME}" -C "${NAME%.tar}"; rm -f "${NAME}"; done 
cd ..
mkdir val && cd val && tar -xvf ~/ILSVRC2012_img_val.tar
wget -qO- https://raw.githubusercontent.com/soumith/imagenetloader.torch/master/valprep.sh | bash
cd ..
```
