#!/bin/bash

NUM_GPU=1
NUM_EXP=1


TensorFlow_resnet50_FP32_PARAMS=(
             "nvidia-examples/resnet50v1.5"
             args
             --mode "training_benchmark" 
             --batch_size "256" 
             --results_dir "output" 
             --num_iter "100" 
             --data_format "NHWC" 
             --use_xla
           )

TensorFlow_resnet50_FP16_PARAMS=(
             "nvidia-examples/resnet50v1.5"
             args
             --mode "training_benchmark" 
             --batch_size "256" 
             --results_dir "output" 
             --num_iter "100" 
             --data_format "NHWC" 
             --use_xla
             --use_tf_amp 
             --use_static_loss_scaling 
             --loss_scale "128"
           )

