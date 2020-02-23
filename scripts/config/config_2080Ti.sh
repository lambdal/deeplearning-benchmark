#!/bin/bash

NUM_GPU=1

SSD_PARAMS=(    
             --batch-size             "16"
             --benchmark-warmup       "100"
             --benchmark-iterations   "200"
           )