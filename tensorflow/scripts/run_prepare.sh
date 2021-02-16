#!/bin/bash


# SSD
cp /scripts/patch/download_all.sh /workspace/nvidia-examples/ssdv1.2
cp -rf /workspace/nvidia-examples/ssdv1.2/models/research/object_detection /workspace/nvidia-examples/ssdv1.2

./run_prepare_tensorflow.sh