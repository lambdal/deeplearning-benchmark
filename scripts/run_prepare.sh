#!/bin/bash

# PyTorch
cp /scripts/patch/wmt16_en_de.sh examples/gnmt/scripts
cp /scripts/patch/getdata.sh examples/transformer-xl
cp /scripts/patch/prepare_dataset.sh examples/tacotron2/scripts
cp /scripts/patch/squad_download.sh examples/bert/data/squad

./run_prepare_pytorch.sh