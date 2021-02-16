#!/bin/bash

# -------------------------
# SSD
# -------------------------

pushd .

OLD_PYTHONPATH=$PYTHONPATH
export PYTHONPATH=/workspace/nvidia-examples/ssdv1.2/models/research:/workspace/nvidia-examples/ssdv1.2/models/research/slim:$PYTHONPATH

cd nvidia-examples/ssdv1.2
./download_all.sh nvidia_ssd /data/object_detection /data/object_detection/checkpoints

chmod -R a+rwx /data/object_detection
chmod -R a+rwx /data/coco2017_tfrecords

PYTHONPATH=$OLD_PYTHONPATH

popd