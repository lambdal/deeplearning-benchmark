#!/bin/bash
NAME_DATASET=${1:-"all"}

# PyTorch
cp /scripts/patch/wmt16_en_de.sh benchmark/Translation/GNMT/scripts
cp /scripts/patch/getdata.sh benchmark/LanguageModeling/Transformer-XL
cp /scripts/patch/prepare_dataset.sh benchmark/SpeechSynthesis/Tacotron2/scripts
cp /scripts/patch/squad_download.sh benchmark/LanguageModeling/BERT/data/squad
cp /scripts/patch/download_dataset.sh benchmark/Recommendation/NCF
cp /scripts/patch/prepare_dataset_ncf.sh benchmark/Recommendation/NCF/prepare_dataset.sh

./run_prepare_pytorch.sh $NAME_DATASET
