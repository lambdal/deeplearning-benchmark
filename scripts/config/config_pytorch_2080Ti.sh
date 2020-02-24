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
           )

PyTorch_resnet50_FP16_PARAMS=(   
           )

PyTorch_resnet50_AMP_PARAMS=(   
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