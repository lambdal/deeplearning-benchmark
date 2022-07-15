#!/bin/bash

for task in "${!BATCH_SIZE_FIX[@]}"; do
    # https://unix.stackexchange.com/questions/638775/retrieve-bash-array-by-referencing-its-name-as-a-variable
    declare -n FIX_TASK_PARAMS=${task}_PARAMS

    # dealt with bert seperately (benchmark launched by shell script)
    if [[ "$task" == *"bert"* ]];
    then
        FIX_TASK_PARAMS[5]=${BATCH_SIZE_FIX[$task]}
    else
        for i in ${!FIX_TASK_PARAMS[@]}; do
            if [ ${FIX_TASK_PARAMS[i]} == "--batch-size" ] || [ ${FIX_TASK_PARAMS[i]} == "SOLVER.IMS_PER_BATCH" ] || [ ${FIX_TASK_PARAMS[i]} == "--train-batch-size" ] || [ ${FIX_TASK_PARAMS[i]} == "--batch_size" ]; then
                FIX_TASK_PARAMS[$i+1]=${BATCH_SIZE_FIX[$task]}
            fi
        done
    fi
done
