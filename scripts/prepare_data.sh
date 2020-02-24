#!/bin/bash

# PyTorch_SSD & MaskRCNN
pushd .

cd examples/ssd

./download_dataset.sh /data/object_detection
chmod -R a+rwx /data/object_detection

popd
