# Benchmarks


This project provides a unified library to benchmark deep learning tasks. 


### Instructions

#### Pull images

```
docker pull nvcr.io/nvidia/pytorch:20.01-py3
```

#### Prepare data

```


# PyTorch ResNet
# Make Sure you have ILSVRC2012_img_train.tar and ILSVRC2012_img_val.tar at home directory,

cd
mkdir -p data/imagenet && cd data/imagenet
mkdir train && cd train 
tar -xvf ~/ILSVRC2012_img_train.tar 
find . -name "*.tar" | while read NAME ; do mkdir -p "${NAME%.tar}"; tar -xvf "${NAME}" -C "${NAME%.tar}"; rm -f "${NAME}"; done 
cd ..
mkdir val && cd val && tar -xvf ~/ILSVRC2012_img_val.tar
wget -qO- https://raw.githubusercontent.com/soumith/imagenetloader.torch/master/valprep.sh | bash
cd ..


# PyTorch Others
docker run --gpus all --rm --shm-size=16g -v ~/data:/data -v $(pwd)"/scripts":/scripts nvcr.io/nvidia/pytorch:20.01-py3 /bin/bash -c "cp -r /scripts/* /workspace; cp /scripts/config/wmt16_en_de.sh examples/gnmt/scripts; cp /scripts/config/getdata.sh examples/transformer-xl; cp /scripts/config/prepare_dataset.sh examples/tacotron2/scripts; cp /scripts/config/squad_download.sh examples/bert/data/squad; ./prepare_data.sh"
```

#### Run 

```
# PyTorch
docker run --gpus all --rm --shm-size=16g -v ~/data:/data -v $(pwd)"/scripts":/scripts -v $(pwd)"/results":/results nvcr.io/nvidia/pytorch:20.01-py3 /bin/bash -c "cp -r /scripts/* /workspace; pip install 'git+https://github.com/NVIDIA/dllogger'; cp /scripts/config/run_squad.py examples/bert; ./run_benchmark_pytorch.sh TitanRTX"
```


### Log

#### 2020-02-28

- [x] PyTorch + BERT base finetune on SQUAD
- [x] PyTorch + BERT lager finetune on SQUAD


#### 2020-02-25

- [x] PyTorch + Tacotron 2
- [x] PyTorch + WaveGlow

#### 2020-02-24

- [x] PyTorch + ResNet50
- [x] PyTorch + gnmt
- [x] PyTorch + NCF
- [ ] ~~PyTorch + transformer~~
- [x] PyTorch + transformer XL base
- [x] PyTorch + transformer XL large


#### 2020-02-23

- [x] Refactorize code
- [x] PyTorch + MaskRCNN

#### 2020-02-22

- [x] Project created.
- [x] Add PyTorch SSD

