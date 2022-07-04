# PyTorch Benchmarks


This project provides a wrapper to run PyTorch benchmarks using NVidia's [Deep Learning Examples](https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch) repo. 

Reference numbers can be found on this [GPU benchmark](https://lambdalabs.com/gpu-benchmarks) website.


### Instructions

The benchmarks are containerized. You need to have NVidia driver and docker on the baremetal, and pull the PyTroch Docker image. Below are the configurations we used to produce the numbers on [this](https://lambdalabs.com/gpu-benchmarks) website.


| GPU Generation | NVidia Driver | NGC Version |
|---|---|---|
| Pre-Ampere  | >=440.33 | >=pytorch:20.01-py3 |
| Ampere  | >=455.32 | >=pytorch:20.10-py3 |


#### Install Docker

```
# Docker
sudo apt-get install apt-transport-https ca-certificates curl gnupg-agent software-properties-common && curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo apt-key add - && sudo add-apt-repository "deb [arch=amd64] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable"
sudo apt-get update && sudo apt-get install docker-ce docker-ce-cli containerd.io

# NVidia docker
distribution=$(. /etc/os-release;echo $ID$VERSION_ID) && curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add - && curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit && sudo systemctl restart docker
sudo groupadd docker
sudo usermod -aG docker $USER
sudo reboot

# Test
docker run --gpus all nvidia/cuda:10.0-base nvidia-smi
```

#### Pull images

NVIDIA has removed the examples from their PyTorch NGC container. The latest container has it is `pytorch:21.07-py3`
```
export NAME_NGC=pytorch:21.07-py3
docker pull nvcr.io/nvidia/${NAME_NGC}
```

#### Prepare data


__ImageNet (For ResNet only)__

You can use synthetic data or real data to benchmark ResNet. To run benchmark with synthetic data, simply add `--data-backend syntetic` to the [config file](https://github.com/lambdal/deeplearning-benchmark/blob/master/pytorch/scripts/config/config_pytorch_2xA100_p4.sh#L38) (right, there is a typo in NVidia's code).

If you want to benchmark ResNet with real data, here are the steps assuming `ILSVRC2012_img_train.tar` and `ILSVRC2012_img_val.tar` have already been downloaded to your home directory.

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


__Data for Other Models__

Here is the one line to get data ready for other (non-ResNet) models.

```
cd deeplearning-benchmark/pytorch

docker run --gpus all --rm --shm-size=64g \
-v ~/data:/data -v $(pwd)"/scripts":/scripts nvcr.io/nvidia/${NAME_NGC} \
/bin/bash -c "cp -r /scripts/* /workspace;  ./run_prepare.sh"
```

#### Prepare configuration files

Benchmark is defined in a configuration file. For example, here is a [config file](https://github.com/lambdal/deeplearning-benchmark/blob/master/pytorch/scripts/config/config_pytorch_2xV100.sh) that creates benchmark jobs for a 2xV100 setup. It specifies the number of GPUs, the number of experiments for each task, and the input arguments for individual task (SSD, ResNet, TransformerXL etc.)

```
# Number of GPUs
NUM_GPU=2 

# Number of experiments to run for each task
NUM_EXP=1

# Task: benchmark SSD in FP32
PyTorch_SSD_FP32_PARAMS=(
             "examples/ssd"
             args
             --data                   "/data/object_detection"
             --batch-size             "108"
             --benchmark-warmup       "50"
             --benchmark-iterations   "400"
             --learning-rate          "0"
           )

# Task: benchmarking SSD in Automatic Mixed Precision (AMP)
PyTorch_SSD_AMP_PARAMS=(
             "examples/ssd"
             args
             --data                   "/data/object_detection"
             --batch-size             "192"
             --benchmark-warmup       "50"
             --benchmark-iterations   "200"
             --amp
             --learning-rate          "0"
           )

# More tasks
...
```

See [this folder](https://github.com/lambdal/deeplearning-benchmark/blob/master/scripts/config) for reference of different GPU configurations.

#### Run Benchmark

Simply call the `run_benchmark.sh` script with the correct number of GPUs and the name of the config file. 

You can customize the folders to mount the data, scripts and results, the following examples assumes the data lives on the home directory on the host machine, and you want to read script/write results to the default folders in the `deeplearning-benchmark` repo. 

You can also tasks for a particular model. For example, pass `resnet` to `run_benchmark.sh` means you want to benchmark ResNet (in both fp32, amp, and fp16), and `resnet_fp32` means you only want to benchmark ResNet in FP32.

```
cd deeplearning-benchmark/pytorch

# 2xV100 on all tasks
docker run --gpus '"device=list-of-gpus"' --rm --shm-size=64g \
-v ~/data:/data -v $(pwd)"/scripts":/scripts -v $(pwd)"/results":/results nvcr.io/nvidia/${NAME_NGC} \
/bin/bash -c "cp -r /scripts/* /workspace; ./run_benchmark.sh 2xV100 all"

# TitanRTX on resnet tasks (fp32 and fp16)
docker run --gpus '"device=list-of-gpus"' --rm --shm-size=64g \
-v ~/data:/data -v $(pwd)"/scripts":/scripts -v $(pwd)"/results":/results nvcr.io/nvidia/${NAME_NGC} \
/bin/bash -c "cp -r /scripts/* /workspace; ./run_benchmark.sh titanrtx resnet"

# TitanRTX on renset fp32
docker run --gpus '"device=list-of-gpus"' --rm --shm-size=64g \
-v ~/data:/data -v $(pwd)"/scripts":/scripts -v $(pwd)"/results":/results nvcr.io/nvidia/${NAME_NGC} \
/bin/bash -c "cp -r /scripts/* /workspace; ./run_benchmark.sh titanrtx resnet_fp32"
```

#### Gather Results

We provide some simply script to gather the results (everything in the results folder) to [CSV](https://github.com/lambdal/deeplearning-benchmark/blob/master/pytorch-train-throughput-fp32.csv) files for both training throughput and batch size.

```
python scripts/compile_results_pytorch_throughput.py --precision fp32 --system all

python scripts/compile_results_pytorch_bs.py --precision fp32 --system all
```

To gather your own benchmarks, you need to add your system to the `list_system`. See the scripts ([1](https://github.com/lambdal/deeplearning-benchmark/blob/master/pytorch/scripts/compile_results_pytorch_throughput.py),[2](https://github.com/lambdal/deeplearning-benchmark/blob/master/pytorch/scripts/compile_results_pytorch_bs.py)) for details.


### Notes

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



<!-- ### Log

#### 2020-03-08

- [x] Tune Performance on different cards
- [x] Fixed bugs related to data pipeline
- [x] Refresh results for V100s, QuadroRTX6000 and 2080Ti

#### 2020-03-01

- [x] Code Refactoring
- [x] Gather System Info
- [x] Gather PyTorch Benchmark Statistics


#### 2020-02-28

- [x] PyTorch + BERT base finetune on SQUAD
- [x] PyTorch + BERT lager finetune on SQUAD


#### 2020-02-25

- [x] PyTorch + Tacotron 2
- [x] PyTorch + WaveGlow

#### 2020-02-24

- [x] PyTorch + ResNet50
- [x] PyTorch + gnmt
- [x] PyTorch + NCF
- [ ] ~~PyTorch + transformer~~
- [x] PyTorch + transformer XL base
- [x] PyTorch + transformer XL large


#### 2020-02-23

- [x] Refactorize code
- [x] PyTorch + MaskRCNN

#### 2020-02-22

- [x] Project created.
- [x] Add PyTorch SSD

 -->
