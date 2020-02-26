#!/bin/bash

# ## -------------------------
# ## PyTorch_SSD & MaskRCNN
# ## -------------------------

# pushd .

# cd examples/ssd

# ./download_dataset.sh /data/object_detection
# chmod -R a+rwx /data/object_detection

# popd

# ## -------------------------
# ## PyTorch gnmt
# ## -------------------------

# pushd .

# cd examples/gnmt
# mkdir -p /data/gnmt/wmt16_de_en
# bash scripts/wmt16_en_de.sh /data/gnmt/wmt16_de_en
# chmod -R a+rwx /data/gnmt/

# popd

# ## -------------------------
# ## PyTorch NCF
# ## -------------------------

# pushd .

# cd examples/ncf
# mkdir -p /data/ncf
# ./prepare_dataset.sh ml-20m /data/ncf
# chmod -R a+rwx /data/ncf

# popd


# ## -------------------------
# ## PyTorch transformer XL
# ## -------------------------

# pushd .
# mkdir -p /data/transformer-xl

# cd examples/transformer-xl
# bash getdata.sh

# chmod -R a+rwx /data/transformer-xl
# popd

# ## -------------------------
# ## PyTorch tacotron2
# ## -------------------------
pushd .

mkdir -p /data/tacotron2/LJSpeech-1.1
cd examples/tacotron2/scripts
./prepare_dataset.sh
chmod -R a+rwx /data/tacotron2/LJSpeech-1.1

popd
