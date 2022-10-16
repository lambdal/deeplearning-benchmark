#!/bin/bash

for fix_task in "${!BATCH_SIZE_FIX[@]}"; do
    # https://unix.stackexchange.com/questions/638775/retrieve-bash-array-by-referencing-its-name-as-a-variable
    declare -n FIX_TASK_PARAMS=${fix_task}_PARAMS

    # dealt with bert seperately (benchmark launched by shell script)
    if [[ "$fix_task" == *"bert"* ]];
    then
        FIX_TASK_PARAMS[5]=${BATCH_SIZE_FIX[$fix_task]}
    else
        for i in ${!FIX_TASK_PARAMS[@]}; do
            if [ ${FIX_TASK_PARAMS[i]} == "--batch-size" ] || [ ${FIX_TASK_PARAMS[i]} == "SOLVER.IMS_PER_BATCH" ] || [ ${FIX_TASK_PARAMS[i]} == "--train-batch-size" ] || [ ${FIX_TASK_PARAMS[i]} == "--batch_size" ]; then
                FIX_TASK_PARAMS[$i+1]=${BATCH_SIZE_FIX[$fix_task]}
            fi
        done
    fi
done


# Number of iterations for SSD 
# It needs to be reduced for more GPUs or larger batch size (otherwise GPU hang at 100%)
for fix_task in "${!SSD_ITER_FIX[@]}"; do
    declare -n FIX_TASK_PARAMS=${fix_task}_PARAMS

    for i in ${!FIX_TASK_PARAMS[@]}; do
        if [ ${FIX_TASK_PARAMS[i]} == "--benchmark-iterations" ]; then
            FIX_TASK_PARAMS[$i+1]=${SSD_ITER_FIX[$fix_task]}
        fi
    done
done


# Dataset for tacotron2 and waveglow
# Choose "subset_2500", "subset_1250", and "subset_625" based on number of GPUs
for fix_task in "${!tacotron2_DATA_FIX[@]}"; do
    declare -n FIX_TASK_PARAMS=${fix_task}_PARAMS

    for i in ${!FIX_TASK_PARAMS[@]}; do
        if [ ${FIX_TASK_PARAMS[i]} == "--training-files" ]; then
            FIX_TASK_PARAMS[$i+1]=${tacotron2_DATA_FIX[$fix_task]}
        fi
    done
done