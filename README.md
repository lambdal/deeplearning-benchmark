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


# Others
docker run --gpus all --rm --shm-size=16g -v ~/data:/data -v $(pwd)"/scripts":/scripts nvcr.io/nvidia/pytorch:20.01-py3 /bin/bash -c "cp -r /scripts/* /workspace;  ./run_prepare.sh"
```

#### Run 

```
# TitanRTX
docker run --gpus '"device=list-of-gpus"' --rm --shm-size=16g -v ~/data:/data -v $(pwd)"/scripts":/scripts -v $(pwd)"/results":/results nvcr.io/nvidia/pytorch:20.01-py3 /bin/bash -c "cp -r /scripts/* /workspace; ./run_benchmark.sh TitanRTX"
```

#### Gather Results

```
python scripts/compile_results_pytorch.py
```


### Notes

| Model | Batch | 
|---|---|
| PyTorch SSD  | Per GPU. `benchmark-iterations` needs to be reduced for more GPUs or larger batch size  |
| PyTorch ResNet  | Per GPU.  |
| PyTorch MaskRCNN  | Global. Need to be scaled by `num_gpu` |
| PyTorch GNMT | Per GPU. |
| PyTorch NCF | Global. Need to be scaled by `num_gpu`|
| PyTorch TransformerXL | Global. Need to be scaled by `num_gpu` |
| PyTorch Tactron | Per GPU. |
| Bert | Per GPU. |



### Log

#### 2020-03-08

- [x] Tune Performance on different cards
- [x] Fixed bugs related to data pipeline
- [x] Refresh results for V100s, QuadroRTX6000 and 2080Ti

#### 2020-03-01

- [x] Code Refactoring
- [x] Gather System Info
- [x] Gather PyTorch Benchmark Statistics


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

