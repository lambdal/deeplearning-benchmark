#!/bin/bash

# PyTorch_SSD & MaskRCNN
# pushd .

# cd examples/ssd

# ./download_dataset.sh /data/object_detection
# chmod -R a+rwx /data/object_detection

# popd

# PyTorch gnmt
pushd .

cd examples/gnmt
mkdir -p /data/gnmt/wmt16_de_en
bash scripts/wmt16_en_de.sh /data/gnmt/wmt16_de_en
chmod -R a+rwx /data/gnmt/

popd

