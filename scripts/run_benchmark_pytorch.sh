#!/bin/bash

SYSTEM=${1:-"2080Ti"}

source config/config_pytorch_${SYSTEM}.sh

echo ${SYSTEM}
echo ${NUM_GPU}


declare -A TASKS=(
    # [PyTorch_SSD_FP32]=benchmark_pytorch_ssd
    # [PyTorch_SSD_AMP]=benchmark_pytorch_ssd
    # [PyTorch_resnet50_FP32]=benchmark_pytorch_resnet50
    # [PyTorch_resnet50_FP16]=benchmark_pytorch_resnet50
    # [PyTorch_resnet50_AMP]=benchmark_pytorch_resnet50
    [PyTorch_maskrcnn_FP32]=benchmark_pytorch_maskrcnn
    [PyTorch_maskrcnn_FP16]=benchmark_pytorch_maskrcnn
    # [PyTorch_gnmt_FP32]=benchmark_pytorch_gnmt
    # [PyTorch_gnmt_FP16]=benchmark_pytorch_gnmt
    # [PyTorch_ncf_FP32]=benchmark_pytorch_ncf
    # [PyTorch_ncf_FP16]=benchmark_pytorch_ncf
    # [PyTorch_transformerxlbase_FP32]=benchmark_pytorch_transformerxl
    # [PyTorch_transformerxlbase_FP16]=benchmark_pytorch_transformerxl
    # [PyTorch_transformerxllarge_FP32]=benchmark_pytorch_transformerxl
    # [PyTorch_transformerxllarge_FP16]=benchmark_pytorch_transformerxl    
)


benchmark_pytorch_ssd() {
    
    local task="$1"

    echo "${task} started: "
    pushd .
    RESULTS_PATH=/results/${SYSTEM}/${task}/
    TASK_PARAMS=${task}_PARAMS[@]

    mkdir -p $RESULTS_PATH
    
    cd examples/ssd
    for i in $(seq 1 $NUM_EXP); do
        RESULTS_FILE=${RESULTS_PATH}$(date +%d-%m-%Y_%H-%M-%S)".txt"
        echo $RESULTS_FILE

        python -m torch.distributed.launch --nproc_per_node=${NUM_GPU} main.py \
        --mode benchmark-training ${!TASK_PARAMS} |& tee $RESULTS_FILE

        chmod a+rwx $RESULTS_FILE

        sleep 2
    done

    chmod -R a+rwx $RESULTS_PATH
    echo ${!TASK_PARAMS}
    echo "${task} ended."
    popd    

}


benchmark_pytorch_resnet50() {
    
    local task="$1"

    echo "${task} started: "
    pushd .
    RESULTS_PATH=/results/${SYSTEM}/${task}/
    TASK_PARAMS=${task}_PARAMS[@]

    mkdir -p $RESULTS_PATH

    cd examples/resnet50v1.5
    for i in $(seq 1 $NUM_EXP); do
        RESULTS_FILE=${RESULTS_PATH}$(date +%d-%m-%Y_%H-%M-%S)".txt"
        echo $RESULTS_FILE
        
        python ./multiproc.py --nproc_per_node ${NUM_GPU} ./main.py \
        ${!TASK_PARAMS} |& tee $RESULTS_FILE
        
        chmod a+rwx $RESULTS_FILE

        sleep 2
    done

    chmod -R a+rwx $RESULTS_PATH
    echo ${!TASK_PARAMS}
    echo "${task} ended."
    popd    

}


benchmark_pytorch_maskrcnn() {
    local task="$1"

    echo "${task} started: "
    pushd .
    RESULTS_PATH=/results/${SYSTEM}/${task}/
    TASK_PARAMS=${task}_PARAMS[@]

    mkdir -p $RESULTS_PATH
    
    LOGFILE="/results/joblog.log"
    GLOBAL_BATCH=`echo ${!TASK_PARAMS} | grep -oP '(?<=SOLVER.IMS_PER_BATCH )\w+'`

    cd examples/maskrcnn/pytorch
    for i in $(seq 1 $NUM_EXP); do
        RESULTS_FILE=${RESULTS_PATH}$(date +%d-%m-%Y_%H-%M-%S)".txt"
        echo $RESULTS_FILE

        python -m torch.distributed.launch --nproc_per_node=${NUM_GPU} tools/train_net.py \
        --skip-test \
        ${!TASK_PARAMS} \
        | tee $LOGFILE
        
        time=`cat $LOGFILE | grep -F 'maskrcnn_benchmark.trainer INFO: Total training time' | tail -n 1 | awk -F'(' '{print $2}' | awk -F' s ' '{print $1}' | egrep -o [0-9.]+`
        statement=`cat $LOGFILE | grep -F 'maskrcnn_benchmark.trainer INFO: Total training time' | tail -n 1`
        calc=$(echo $time 1.0 $GLOBAL_BATCH | awk '{ printf "%f", $2 * $3 / $1 }')
        echo "Training perf is: "$calc" FPS" |& tee $RESULTS_FILE
        rm $LOGFILE
        rm /results/*.txt
        rm /results/*.pth
        rm /results/*checkpoint*        
        sleep 2
    done

    chmod -R a+rwx $RESULTS_PATH
    echo ${!TASK_PARAMS}
    echo "${task} ended."
    popd

}


benchmark_pytorch_gnmt() {
    local task="$1"

    echo "${task} started: "
    pushd .
    RESULTS_PATH=/results/${SYSTEM}/${task}/
    TASK_PARAMS=${task}_PARAMS[@]

    mkdir -p $RESULTS_PATH
    
    cd examples/gnmt
    for i in $(seq 1 $NUM_EXP); do
        RESULTS_FILE=${RESULTS_PATH}$(date +%d-%m-%Y_%H-%M-%S)".txt"
        echo $RESULTS_FILE

        python3 -m launch --nproc_per_node=${NUM_GPU} train.py ${!TASK_PARAMS} |& tee $RESULTS_FILE

        sleep 2
    done

    chmod -R a+rwx $RESULTS_PATH
    echo ${!TASK_PARAMS}
    echo "${task} ended."
    popd 
}


benchmark_pytorch_ncf() {
    local task="$1"

    echo "${task} started: "
    pushd .
    RESULTS_PATH=/results/${SYSTEM}/${task}/
    TASK_PARAMS=${task}_PARAMS[@]

    mkdir -p $RESULTS_PATH
    

    cd examples/ncf
    for i in $(seq 1 $NUM_EXP); do
        RESULTS_FILE=${RESULTS_PATH}$(date +%d-%m-%Y_%H-%M-%S)".txt"
        echo $RESULTS_FILE

        python -m torch.distributed.launch --nproc_per_node=${NUM_GPU} --use_env ncf.py ${!TASK_PARAMS} |& tee $RESULTS_FILE

        sleep 2
    done

    chmod -R a+rwx $RESULTS_PATH
    echo ${!TASK_PARAMS}
    echo "${task} ended."
    popd         
}


benchmark_pytorch_transformerxl() {
    local task="$1"

    echo "${task} started: "
    pushd .
    RESULTS_PATH=/results/${SYSTEM}/${task}/
    TASK_PARAMS=${task}_PARAMS[@]

    mkdir -p $RESULTS_PATH

    cd examples/transformer-xl/pytorch
    for i in $(seq 1 $NUM_EXP); do
        RESULTS_FILE=${RESULTS_PATH}$(date +%d-%m-%Y_%H-%M-%S)".txt"
        echo $RESULTS_FILE

        python -m torch.distributed.launch --nproc_per_node=${NUM_GPU} --use_env train.py ${!TASK_PARAMS} |& tee $RESULTS_FILE

        sleep 2
    done

    chmod -R a+rwx $RESULTS_PATH
    echo ${!TASK_PARAMS}
    echo "${task} ended."
    popd    
}


main() {
    for task in "${!TASKS[@]}"; do
        ${TASKS[${task}]} $task
    done
}

main "$@"
