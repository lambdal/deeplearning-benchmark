#!/bin/bash

# -------------------------
# PyTorch_SSD & MaskRCNN
# -------------------------

pushd .

cd benchmark/Detection/SSD

./download_dataset.sh /data/object_detection
chmod -R a+rwx /data/object_detection

popd

# -------------------------
# PyTorch gnmt
# -------------------------

pushd .

cd benchmark/Translation/GNMT
pip install -r requirements.txt
mkdir -p /data/gnmt/wmt16_de_en
bash scripts/wmt16_en_de.sh /data/gnmt/wmt16_de_en
chmod -R a+rwx /data/gnmt/

popd

## -------------------------
## PyTorch NCF
## -------------------------

pushd .

cd benchmark/Recommendation/NCF
mkdir -p /data/ncf
./prepare_dataset.sh ml-20m /data/ncf
chmod -R a+rwx /data/ncf

popd

## -------------------------
## PyTorch transformer XL
## -------------------------

pushd .
mkdir -p /data/transformer-xl

cd benchmark/LanguageModeling/Transformer-XL
bash getdata.sh

chmod -R a+rwx /data/transformer-xl
popd

## -------------------------
## PyTorch tacotron2
## -------------------------
pushd .

mkdir -p /data/tacotron2/LJSpeech-1.1
cd benchmark/SpeechSynthesis/Tacotron2/scripts
./prepare_dataset.sh
chmod -R a+rwx /data/tacotron2/LJSpeech-1.1

popd


## ------------------------
## PyTorch BERT Squad
## ------------------------
pushd .
mkdir -p /data/squad
cd benchmark/LanguageModeling/BERT/data/squad
./squad_download.sh /data/squad
popd
chmod -R a+rwx /data/squad

pushd .
mkdir -p /data/bert_large
cd /data/bert_large
curl -LO https://lambdalabs-files.s3-us-west-2.amazonaws.com/bert_large/bert_large_uncased.pt
curl -LO https://lambdalabs-files.s3-us-west-2.amazonaws.com/bert_large/bert_config.json
wget https://lambdalabs-files.s3-us-west-2.amazonaws.com/bert_large/bert-large-uncased-vocab.txt
popd
chmod -R a+rwx /data/bert_large

pushd .
mkdir -p /data/bert_base
cd /data/bert_base
curl -LO https://lambdalabs-files.s3-us-west-2.amazonaws.com/bert_base/bert_base_uncased.pt
curl -LO https://lambdalabs-files.s3-us-west-2.amazonaws.com/bert_base/bert_config.json
wget https://lambdalabs-files.s3-us-west-2.amazonaws.com/bert_base/bert-base-uncased-vocab.txt
popd
chmod -R a+rwx /data/bert_base
