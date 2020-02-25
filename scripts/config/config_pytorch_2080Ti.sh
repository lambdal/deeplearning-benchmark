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
             --amp
           )

PyTorch_resnet50_FP32_PARAMS=(
                                      "/data/imagenet"
             --arch                   "resnet50"
             --epochs                 "1" 
             --prof                   "100" 
             --batch-size             "64"
             --raport-file            "benchmark.json"
             --print-freq             "1"
             --training-only
           )

PyTorch_resnet50_FP16_PARAMS=(
                                      "/data/imagenet"
             --arch                   "resnet50"
             --fp16
             --static-loss-scale      "256"
             --epochs                 "1" 
             --prof                   "100" 
             --batch-size             "128"
             --raport-file            "benchmark.json"
             --print-freq             "1"
             --training-only  
           )

PyTorch_resnet50_AMP_PARAMS=(
                                      "/data/imagenet"
             --arch                   "resnet50"
             --amp
             --static-loss-scale      "256"
             --epochs                 "1" 
             --prof                   "100" 
             --batch-size             "128"
             --raport-file            "benchmark.json"
             --print-freq             "1"
             --training-only   
           )

PyTorch_maskrcnn_FP32_PARAMS=(
             --config-file            "/workspace/config/e2e_mask_rcnn_R_50_FPN_1x.yaml"
             SOLVER.IMS_PER_BATCH     "1"
             DTYPE                    "float32"
             SOLVER.MAX_ITER          "3665"
             OUTPUT_DIR               "/results"
             PATHS_CATALOG            "/workspace/config/paths_catalog_ci.py"
           )

PyTorch_maskrcnn_FP16_PARAMS=(
             --config-file            "/workspace/config/e2e_mask_rcnn_R_50_FPN_1x.yaml"
             SOLVER.IMS_PER_BATCH     "1"
             DTYPE                    "float16"
             SOLVER.MAX_ITER          "3665"
             OUTPUT_DIR               "/results"
             PATHS_CATALOG            "/workspace/config/paths_catalog_ci.py"
           )

PyTorch_gnmt_FP32_PARAMS=(
            --dataset-dir             "/data/gnmt/wmt16_de_en"
            --train-batch-size        "96"
            --math                    "fp32"
            --epochs                  "1"
            --seed                    "2"
           )

PyTorch_gnmt_FP16_PARAMS=(
            --dataset-dir             "/data/gnmt/wmt16_de_en"
            --train-batch-size        "192"
            --math                    "fp16"
            --epochs                  "1"
            --seed                    "2"
           )

PyTorch_ncf_FP32_PARAMS=(
            --data                    "/data/ncf/cache/ml-20m"
            --epochs                  "1"
            --opt_level               "O0"
           )

PyTorch_ncf_FP16_PARAMS=(
            --data                    "/data/ncf/cache/ml-20m"
            --epochs                  "1"
            --opt_level               "O2"
           )