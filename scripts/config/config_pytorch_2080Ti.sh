#!/bin/bash

NUM_GPU=1
NUM_EXP=2

PyTorch_SSD_FP32_PARAMS=(   
             --data                   "/data/object_detection"
             --batch-size             "16"
             --benchmark-warmup       "100"
             --benchmark-iterations   "200"
           )


PyTorch_SSD_AMP_PARAMS=(   
             --data                   "/data/object_detection"
             --batch-size             "32"
             --benchmark-warmup       "100"
             --benchmark-iterations   "200"
           )


PyTorch_resnet50_FP32_PARAMS=(   
             --data                   "/data/object_detection"
             --batch-size             "32"
             --benchmark-warmup       "100"
             --benchmark-iterations   "200"
           )

PyTorch_resnet50_FP16_PARAMS=(   
             --data                   "/data/object_detection"
             --batch-size             "32"
             --benchmark-warmup       "100"
             --benchmark-iterations   "200"
           )

PyTorch_resnet50_AMP_PARAMS=(   
             --data                   "/data/object_detection"
             --batch-size             "32"
             --benchmark-warmup       "100"
             --benchmark-iterations   "200"
           )