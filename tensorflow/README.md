# TensorFlow Benchmarks


This project provides a wrapper to benchmark GPUs using NVIDIA's [TensorFlow GPU-Accelerated Containers](https://ngc.nvidia.com/catalog/containers/nvidia:tensorflow). 

Reference numbers can be found on this [GPU benchmark](https://lambdalabs.com/gpu-benchmarks) website.


### Instructions

The benchmarks are containerized. You need to have NVIDIA driver and docker on the baremetal, and pull the Tensor Docker image. Below are the configurations we used to produce the numbers on [this](https://lambdalabs.com/gpu-benchmarks) website.


| GPU Generation | NVIDIA Driver | NGC Version |
|---|---|---|
| Pre-Ampere  | =455.45.01 | =tensorflow:20.12-tf1-py3 |



#### Install Docker


All you need to do is to install a version of GPU accelerated Docker with this command:

```
sudo apt-get install docker.io nvidia-container-toolkit
```


It is useful to create a Unix group called `docker` and add users to it. This allows you to skip prefacing the docker command with `sudo`:

```
USER=ubuntu
sudo groupadd docker
sudo usermod -aG docker $USER
sudo reboot

# Test
docker run --gpus all nvidia/cuda:10.0-base nvidia-smi
```
