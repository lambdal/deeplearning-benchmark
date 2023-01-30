#!/bin/bash
NAME_DATASET=${1:-"all"}

declare -A DATASETS=(
    [object_detection]=prepare_object_detection
    [gnmt]=prepare_gnmt
    [ncf]=prepare_ncf
    [transformer]=prepare_transformer
    [tacotron2]=prepare_tacotron2
    [bert]=prepare_bert
)


prepare_object_detection() {
    # -------------------------
    # PyTorch_SSD & MaskRCNN
    # -------------------------
    echo "PyTorch_SSD & MaskRCNN"
    pushd .

    cd benchmark/Detection/SSD

    ./download_dataset.sh /data/object_detection
    chmod -R a+rwx /data/object_detection

    popd
}

prepare_gnmt() {
    # -------------------------
    # PyTorch gnmt
    # -------------------------
    echo "PyTorch gnmt"
    pushd .

    cd benchmark/Translation/GNMT
    pip install -r requirements.txt
    mkdir -p /data/gnmt/wmt16_de_en
    bash scripts/wmt16_en_de.sh /data/gnmt/wmt16_de_en
    chmod -R a+rwx /data/gnmt/

    popd
}

prepare_ncf() {
    ## -------------------------
    ## PyTorch NCF
    ## -------------------------
    echo "PyTorch NCF"
    pushd .

    cd benchmark/Recommendation/NCF
    mkdir -p /data/ncf
    ./prepare_dataset.sh ml-20m /data/ncf
    chmod -R a+rwx /data/ncf

    popd
}

prepare_transformer() {
    ## -------------------------
    ## PyTorch transformer XL
    ## -------------------------
    echo "PyTorch transformer XL"
    pushd .
    mkdir -p /data/transformer-xl

    cd benchmark/LanguageModeling/Transformer-XL
    bash getdata.sh

    chmod -R a+rwx /data/transformer-xl
    popd
}


prepare_tacotron2() {
    ## -------------------------
    ## PyTorch tacotron2
    ## -------------------------
    echo "PyTorch tacotron2"
    pushd .

    mkdir -p /data/tacotron2/LJSpeech-1.1
    cd benchmark/SpeechSynthesis/Tacotron2/scripts
    ./prepare_dataset.sh
    chmod -R a+rwx /data/tacotron2/LJSpeech-1.1

    popd
}


prepare_bert() {
    ## ------------------------
    ## PyTorch BERT Squad
    ## ------------------------
    echo "PyTorch BERT Squad"
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
}

main() {
    for dataset in "${!DATASETS[@]}"; do
        if [[ "${dataset,,}" == *"$NAME_DATASET"* ]] || [ "$NAME_DATASET" == "all" ]; then
            ${DATASETS[${dataset}]}
        fi
    done
}

main "$@"