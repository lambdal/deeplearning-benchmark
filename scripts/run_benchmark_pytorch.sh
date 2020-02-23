#!/bin/bash

SYSTEM=${1:-"2080Ti"}

source config/config_pytorch_${SYSTEM}.sh

echo ${SYSTEM}
echo ${NUM_GPU}


declare -A TASKS=(
    [PyTorch_SSD_FP32]=benchmark_pytorch_ssd
    [PyTorch_SSD_AMP]=benchmark_pytorch_ssd
    [PyTorch_resnet50_FP32]=benchmark_pytorch_resnet50
    [PyTorch_resnet50_FP16]=benchmark_pytorch_resnet50
    [PyTorch_resnet50_AMP]=benchmark_pytorch_resnet50
)


benchmark_pytorch_ssd() {
    
    local task="$1"

    echo "${task} started: "
    pushd .
    RESULTS_PATH=/results/${SYSTEM}/${task}/
    TASK_PARAMS=${task}_PARAMS[@]

    mkdir -p $RESULTS_PATH
    chmod -R a+rwx $RESULTS_PATH

    cd examples/ssd
    for i in $(seq 1 $NUM_EXP); do
        RESULTS_FILE=${RESULTS_PATH}$(date +%d-%m-%Y_%H-%M-%S)".txt"
        echo $RESULTS_FILE

        python -m torch.distributed.launch --nproc_per_node=${NUM_GPU} main.py \
        --mode benchmark-training ${!TASK_PARAMS} |& tee $RESULTS_FILE

        chmod a+rwx $RESULTS_FILE

        sleep 2
    done
    echo ${!TASK_PARAMS}
    echo "${TASK_NAME} ended."
    popd    

}


benchmark_pytorch_resnet50() {
    
    local task="$1"

    echo "${task} started: "
    pushd .
    RESULTS_PATH=/results/${SYSTEM}/${task}/
    TASK_PARAMS=${task}_PARAMS[@]

    mkdir -p $RESULTS_PATH
    chmod -R a+rwx $RESULTS_PATH

    cd examples/resnet50v1.5
    for i in $(seq 1 $NUM_EXP); do
        RESULTS_FILE=${RESULTS_PATH}$(date +%d-%m-%Y_%H-%M-%S)".txt"
        echo $RESULTS_FILE

        # python -m torch.distributed.launch --nproc_per_node=${NUM_GPU} main.py \
        # --mode benchmark-training ${!TASK_PARAMS} |& tee $RESULTS_FILE
        
        # chmod a+rwx $RESULTS_FILE

        sleep 2
    done
    echo ${!TASK_PARAMS}
    echo "${TASK_NAME} ended."
    popd    

}


main() {
    for task in "${!TASKS[@]}"; do
        ${TASKS[${task}]} $task
    done
}

main "$@"
