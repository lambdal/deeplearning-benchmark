#!/bin/bash

NUM_GPU=2
NUM_EXP=1


TensorFlow_SSD_FP32_PARAMS=(
             "nvidia-examples/ssdv1.2"
             args
             --model_dir "output"
             --pipeline_config_path "/workspace/nvidia-examples/ssdv1.2/configs/ssd320_bench.config"
           )


TensorFlow_SSD_FP16_PARAMS=(
             "nvidia-examples/ssdv1.2"
             args
             --model_dir "output"
             --pipeline_config_path "/workspace/nvidia-examples/ssdv1.2/configs/ssd320_bench.config"
             --amp
           )


TensorFlow_resnet50_FP32_PARAMS=(
             "nvidia-examples/resnet50v1.5"
             args
             --mode "training_benchmark" 
             --batch_size "128" 
             --results_dir "output" 
             --num_iter "100" 
             --data_format "NHWC" 
             --use_xla
           )

TensorFlow_resnet50_FP16_PARAMS=(
             "nvidia-examples/resnet50v1.5"
             args
             --mode "training_benchmark" 
             --batch_size "128" 
             --results_dir "output" 
             --num_iter "100" 
             --data_format "NHWC" 
             --use_xla
             --use_tf_amp 
             --use_static_loss_scaling 
             --loss_scale "128"
           )

