# Benchmarks


This project provides a unified library to benchmark deep learning tasks. 


### Instructions

#### Pull images

```
docker pull nvcr.io/nvidia/pytorch:20.01-py3
```

#### Prepare data

```
# PyTorch
docker run --gpus all --rm -v ~/data:/data -v $(pwd)"/scripts":/scripts nvcr.io/nvidia/pytorch:20.01-py3 /bin/bash -c "cp -r /scripts/* /workspace; ./prepare_data.sh"
```

#### Run 

```
# PyTorch
docker run --gpus all --rm --shm-size=16g -v ~/data:/data -v $(pwd)"/scripts":/scripts -v $(pwd)"/results":/results nvcr.io/nvidia/pytorch:20.01-py3 /bin/bash -c "cp -r /scripts/* /workspace; ./run_benchmark_pytorch.sh 2080Ti"
```


### Log

#### 2020-02-24

- [x] Refactorize code
- [ ] PyTorch + MaskRCNN

#### 2020-02-22

- [x] Project created.
- [x] Add PyTorch SSD

