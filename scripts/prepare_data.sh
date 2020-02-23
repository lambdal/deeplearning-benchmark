#!/bin/bash

# PyTorch_SSD
pushd .

cd examples/ssd

./download_dataset.sh /data/object_detection
chmod -R a+rwx /data/object_detection

popd