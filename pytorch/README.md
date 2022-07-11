# PyTorch Benchmarks


This project provides a wrapper to run PyTorch benchmarks using NVidia's [Deep Learning Examples](https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch) repo. 

Reference numbers can be found on this [GPU benchmark](https://lambdalabs.com/gpu-benchmarks) website.


### Instructions


The benchmarks are containerized. You need to install NVIDIA driver, docker, and nvidia-container-toolkit, and pull the [PyTroch NGC](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch) Docker image. 

All the benchmarks here are for single-node (single GPU or multiple GPUs). They do NOT work for multiple nodes.


(Update 2022.07) NVIDIA has stopped packaging [Deep Learning Examples](https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch) into their PyTorch NGC images. This guide uses the last PyTorch NGC release (`pytorch:21.07-py3`) that supports the examples.


Different NGC releases can cause non-trivial performance changes to the same hardware. Depend on specific deep learning models, a newer version of NGC can cause the same hardware to run either faster or slower. Below is a major NGC upgrade we adpated to produce [our benchmark](https://lambdalabs.com/gpu-benchmarks). The upgrade was necessary for benchmarking Ampere GPUs, and might be required again for future achitectures (e.g. Hopper)


| GPU Generation | NVidia Driver | NGC Version |
|---|---|---|
| Pre-Ampere  | >=440.33 | >=pytorch:20.01-py3 |
| Ampere  | >=455.32 | >=pytorch:20.10-py3 |


#### Step 0: Prerequisite

A Ubuntu machine with at least one GPU and up-to-date NVIDIA driver. Access to the internet.

If you start from a baremetal Ubuntu machine without NVIDIA driver. Install [Lambda stack](https://lambdalabs.com/lambda-stack-deep-learning-software). 

For workstation

```
LAMBDA_REPO=$(mktemp) && \
wget -O${LAMBDA_REPO} https://lambdalabs.com/static/misc/lambda-stack-repo.deb && \
sudo dpkg -i ${LAMBDA_REPO} && rm -f ${LAMBDA_REPO} && \
sudo apt-get update && sudo apt-get install -y lambda-stack-cuda
sudo reboot
```

For server
```
LAMBDA_REPO=$(mktemp) && \
wget -O${LAMBDA_REPO} https://lambdalabs.com/static/misc/lambda-stack-repo.deb && \
sudo dpkg -i ${LAMBDA_REPO} && rm -f ${LAMBDA_REPO} && \
sudo apt-get update && sudo apt-get install -y lambda-stack-cuda
sudo reboot
```


#### Step 1: Install Docker

Lambda stack doesn't ship nvidia-container-toolkit out of the box. You can install it by 

```
sudo apt-get install docker.io nvidia-container-toolkit && sudo systemctl restart docker
```

Also enable using docker without `sudo` (reboot is required):

```
sudo groupadd docker
sudo usermod -aG docker $USER
sudo reboot

# After reboot, run a quick test to have nvidia-smi listing all the GPUs
docker run --gpus all nvidia/cuda:10.0-base nvidia-smi
```


#### Step 2: Pull images

We pin the PyTorch NGC image to `pytorch:21.07-py3` -- the last version that has [Deep Learning Examples](https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch) in it.
```
export NAME_NGC=pytorch:21.07-py3
docker pull nvcr.io/nvidia/${NAME_NGC}
```

#### Step 3: Clone Repo

```
git clone https://github.com/lambdal/deeplearning-benchmark.git
cd deeplearning-benchmark/pytorch
```

#### Step 4: Prepare data

```
docker run --gpus all --rm --shm-size=64g \
-v ~/data:/data -v $(pwd)"/scripts":/scripts nvcr.io/nvidia/${NAME_NGC} \
/bin/bash -c "cp -r /scripts/* /workspace;  ./run_prepare.sh"
```

This step takes about 25 mins.

#### Step 5: Prepare configuration files

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
source config_v1/fix_bs.sh
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



```
# Number of GPUs
NUM_GPU=1 

# Number of experiments to run for each task. Set to one unless you want to run the same model multiple times
NUM_EXP=1

# Task: benchmark SSD in FP32
PyTorch_SSD_FP32_PARAMS=(
             "examples/ssd"
              args
              --data                   "/data/object_detection"
              --batch-size             "144"
              --benchmark-warmup       "10"
              --benchmark-iterations   "40"
              --learning-rate          "0"
           )



# Task: benchmarking SSD in Automatic Mixed Precision (AMP)
PyTorch_SSD_AMP_PARAMS=(
              "examples/ssd"
              args
              --data                   "/data/object_detection"
              --batch-size             "256"
              --benchmark-warmup       "10"
              --benchmark-iterations   "40"
              --amp
              --learning-rate          "0"
           )

# More tasks
...
```

**Your customized config file should always reference a template that uses the SAME number of GPUs and the SAME GPU memory**. For example, referencing `config_pytorch_48GB.sh` in `config_pytorch_QuadroRTX8000_v1.sh` and `config_pytorch_A6000_v1.sh`, and referencing `config_pytorch_2x24GB.sh` in `config_pytorch_2x3090_v1.sh`.


#### Step 6: Run Benchmark

Use `docker run` to execute the `run_benchmark.sh` script with the correct number of GPUs, paths for mounting the data, code and results, the name of the config file, and the task to run. Last but not the least, set a timeout limit to avoid being stuck at a task for too long.

Docker options:
* number of GPUs: by default use `--gpus all`, which let the docker container use all the GPUs in the machine. You can also choose specific ones, e.g. `--gpus '"device=2,3"'`, which only allow the container to use GPU2 and GPU3 in the machine. This can be helpful to benchmark a server wtih mixed type of GPUs.
* data path: `-v ~/data:/data` is the default path if your data is created using the above "Prepare data" command.
* code path: `-v $(pwd)"/scripts":/scripts` is the correct path and make sure you are inside of `deeplearning-benchmark/pytorch`
* result path: `-v $(pwd)"/results"` is the correct path and make sure you are inside of `deeplearning-benchmark/pytorch`

`run_benchmark_sh` options:
* config name: The name of the config file you want to call. For example, use`QuadroRTX8000_v1` to use `config_pytorch_QuadroRTX8000_v1.sh`
* task: use `all` for all the tasks defined in the config file (23 tasks in total). Or you can call specific task. For example, use `resnet50_fp32` (all lower cases) to only run a single task, or `resnet50` for all the resnet50 related task (fp32 and fp16)
* timeout limit: `600` is the default (in seconds). The benchmark will kill a task if it takes longer than this (e.g. weirdness when a GPU hangs in a multi-gpu job)

Here is the full command to benchmark the above `QuadroRTX8000_v1` config for all models, with 600 secs timeout limit:

```
docker run \
--rm --shm-size=128g \
--gpus all \
-v ~/data:/data \
-v $(pwd)"/scripts":/scripts \
-v $(pwd)"/results":/results \
nvcr.io/nvidia/pytorch:21.07-py3 \
/bin/bash -c "cp -r /scripts/* /workspace; ./run_benchmark.sh QuadroRTX8000_v1 all 600"
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
PyTorch_maskrcnn_FP16               :  sucessful
PyTorch_maskrcnn_FP32               :  sucessful
PyTorch_ncf_FP16                    :  sucessful
PyTorch_ncf_FP32                    :  sucessful
PyTorch_resnet50_AMP                :  sucessful
PyTorch_resnet50_FP16               :  sucessful
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
--gpus all \
-v ~/data:/data \
-v $(pwd)"/scripts":/scripts \
-v $(pwd)"/results":/results \
nvcr.io/nvidia/pytorch:21.07-py3 \
/bin/bash -c "cp -r /scripts/* /workspace; ./run_benchmark.sh QuadroRTX8000_v1 resnet50_fp32 600"
```

#### Step 7: Gather Results

We provide some simply script to gather the results (everything in the results folder) to [CSV](https://github.com/lambdal/deeplearning-benchmark/blob/master/pytorch-train-throughput-fp32.csv) files for both training throughput and batch size.

```
python scripts/compile_results_pytorch_throughput.py --precision fp32 --system all
python scripts/compile_results_pytorch_throughput.py --precision fp16 --system all

python scripts/compile_results_pytorch_bs.py --precision fp32 --system all
python scripts/compile_results_pytorch_bs.py --precision fp16 --system all
```

To gather your own benchmarks, you need to add your system to the `list_system`. For example, this is the line you need to add to the `compile_results_pytorch_throughput.py` script:

```
# key: QuadroRTX8000_v1 as the config name
# value: ([version, num_gpus], rename)
# version: 0 for pytorch:20.01-py3, and 1 for pytorch:20.10-py3 and later
# num_gpus: sometimes num_gpus can't be inferred from config name (for example p3.16xlarge) or missing from the result log. So we ask user to specify it here.
# rename: renaming the system so it is easier to read (for releasing on our benchmark webpage)
# watt per gpu
# price per gpu

'QuadroRTX8000_v1': ([1, 1], 'Quadro RTX 8000 V1', 260, 6900),
```

And this is the line you need to add to the `compile_results_pytorch_bs.py` script:

```
# key: config name
# full name: renaming the system so it is easier to read (for releasing on our benchmark webpage)
# watt per gpu
# price per gpu

('QuadroRTX8000_v1', 'Quadro RTX 8000 V1', 260, 6900, "v1"),
```


See the scripts ([1](https://github.com/lambdal/deeplearning-benchmark/blob/master/pytorch/scripts/compile_results_pytorch_throughput.py),[2](https://github.com/lambdal/deeplearning-benchmark/blob/master/pytorch/scripts/compile_results_pytorch_bs.py)) for details. 


### Notes

#### Batch size 
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


#### Real ImageNet data for resnet50
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
