#!/bin/bash

SYSTEM=${1:-"2080Ti"}


source config/config_${SYSTEM}.sh

# SSD
pushd .
mkdir -p /results/PyTorch/SSD/
sudo chmod -R a+rwx /results/PyTorch/SSD/

cd examples/ssd
python -m torch.distributed.launch --nproc_per_node=${NUM_GPU} main.py \
--mode benchmark-training --data /data/object_detection ${SSD_PARAMS[@]} |& tee "/results/PyTorch/SSD/"$(date +%d-%m-%Y_%H-%M-%S)".txt"

echo "**********************************************"
echo $SYSTEM
echo ${NUM_GPU}
echo ${SSD_PARAMS[@]}

popd