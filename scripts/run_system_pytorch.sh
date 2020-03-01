#!/bin/bash

SYSTEM=${1:-"2080Ti"}

CPU_NAME="$(lscpu | grep "Model name:" | sed -r 's/Model name:\s{1,}//g')"

CPU_MEM="$(free -h | grep Mem: | awk '{ print $2 }')"

GPU_NAME="$(nvidia-smi -i 0 --query-gpu=gpu_name --format=csv,noheader)"
GPU_NAME="${GPU_NAME// /_}"

GPU_MEM="$(nvidia-smi -i 0 --query-gpu=memory.total --format=csv,noheader)"
GPU_MEM="${GPU_MEM// /_}"

NVIDIA_DRIVER="$(nvidia-smi | grep "Driver Version:" | awk '{ print $3 }')"

CUDA_VERSION="$(nvcc --version | grep release | awk '{ print $NF }')"

CUDNN_MAJOR="$(cat /usr/include/cudnn.h | grep "#define CUDNN_MAJOR" | awk '{ print $NF }')"
CUDNN_MINOR="$(cat /usr/include/cudnn.h | grep "#define CUDNN_MINOR" | awk '{ print $NF }')"
CUDNN_PATCHLEVEL="$(cat /usr/include/cudnn.h | grep "#define CUDNN_PATCHLEVEL" | awk '{ print $NF }')"
CUDNN_VERSION=${CUDNN_MAJOR}"."${CUDNN_MINOR}"."${CUDNN_PATCHLEVEL}

MB="$(cat /sys/devices/virtual/dmi/id/board_{vendor,name,version} | tr '\n' ' ')"

PLATFORM_NAME="$(cat /etc/os-release | grep "PRETTY_NAME=" | cut -c 14- | rev | cut -c 2- | rev)"

PT_VERSION="$(python -c "import torch; print(torch.__version__)" | awk '{ print $NF }')"

RESULTS_PATH=/results/${SYSTEM}
mkdir -p $RESULTS_PATH

SYSTEM_FILE=${RESULTS_PATH}/sys_pytorch.txt

echo "CPU: "${CPU_NAME} >> $SYSTEM_FILE
echo "CPU Memory: "${CPU_MEM} >> $SYSTEM_FILE
echo "GPU: "${GPU_NAME} >> $SYSTEM_FILE
echo "GPU Memory: "${GPU_MEM} >> $SYSTEM_FILE
echo "NVIDIA driver: "${NVIDIA_DRIVER} >> $SYSTEM_FILE
echo "CUDA Version: "${CUDA_VERSION} >> $SYSTEM_FILE
echo "CUDNN Version: "$CUDNN_VERSION >> $SYSTEM_FILE
echo "Motherboard: "${MB} >> $SYSTEM_FILE
echo "OS: "${PLATFORM_NAME} >> $SYSTEM_FILE
echo "PyTorch Version: "${PT_VERSION} >> $SYSTEM_FILE

chmod -R a+rwx $SYSTEM_FILE
