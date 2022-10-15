#!/bin/bash

# PyTorch
# cp /scripts/patch/wmt16_en_de.sh benchmark/Translation/GNMT/scripts
# cp /scripts/patch/getdata.sh benchmark/LanguageModeling/Transformer-XL
# cp /scripts/patch/prepare_dataset.sh benchmark/SpeechSynthesis/Tacotron2/scripts
cp /scripts/patch/squad_download.sh benchmark/LanguageModeling/BERT/data/squad

./run_prepare_pytorch.sh
