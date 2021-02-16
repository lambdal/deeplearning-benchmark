#!/bin/bash

SYSTEM=${1:-"2080Ti"}
TASK_NAME=${2:-"all"}

source config/config_tensorflow_${SYSTEM}.sh

echo ${SYSTEM}
echo ${NUM_GPU}


declare -A TASKS=(
    # [TensorFlow_resnet50_FP32]=benchmark_tensorflow_resnet50
    # [TensorFlow_resnet50_FP16]=benchmark_tensorflow_resnet50
    [TensorFlow_SSD_FP32]=benchmark_tensorflow_ssd
    [TensorFlow_SSD_FP16]=benchmark_tensorflow_ssd
)


benchmark_tensorflow_resnet50() {
    
    local task="$1"
    local result="$2"

    TASK_PARAMS=${task}_PARAMS[@]
    local command_para=$(sed 's/.*args //' <<<${!TASK_PARAMS})

    mpiexec --allow-run-as-root --bind-to socket -np ${NUM_GPU} python ./main.py \
    ${command_para} |& tee ${result}   

    rm -rf output

    echo "DONE!" >> ${result}
}



benchmark_tensorflow_ssd() {
    
    local task="$1"
    local result="$2"

    TASK_PARAMS=${task}_PARAMS[@]
    local command_para=$(sed 's/.*args //' <<<${!TASK_PARAMS})

    export PYTHONPATH=/workspace/nvidia-examples/ssdv1.2/models/research:/workspace/nvidia-examples/ssdv1.2/models/research/slim:$PYTHONPATH
    
    TENSOR_OPS=0
    export TF_ENABLE_CUBLAS_TENSOR_OP_MATH_FP32=${TENSOR_OPS}
    export TF_ENABLE_CUDNN_TENSOR_OP_MATH_FP32=${TENSOR_OPS}
    export TF_ENABLE_CUDNN_RNN_TENSOR_OP_MATH_FP32=${TENSOR_OPS}

    mpirun --allow-run-as-root \
           -np $NUM_GPU \
           -H localhost:$NUM_GPU \
           -bind-to none \
           -map-by slot \
           -x NCCL_DEBUG=INFO \
           -x LD_LIBRARY_PATH \
           -x PATH \
           -mca pml ob1 \
           -mca btl ^openib \
            python -u ./object_detection/model_main.py \
                   --alsologtostder \
                   ${command_para} |& tee ${result}  

    rm -rf output

    PERF=$(cat ${result} | sed -n 's|.*global_step/sec: \(\S\+\).*|\1|p' | python -c "import sys; x = sys.stdin.readlines(); x = [float(a) for a in x[int(len(x)*3/4):]]; print(32*$NUM_GPU*sum(x)/len(x), 'img/s')")
    echo "Single GPU single precision training performance: $PERF" >> ${result}

    echo "DONE!" >> ${result}
    
}


benchmark_tensorflow() {
    local task="$1"

    echo "${task} started: "
    
    RESULTS_PATH=/results/${SYSTEM}/${task}/
    TASK_PARAMS=${task}_PARAMS[@]
    MONITOR_INTERVAL=2

    local command_path=$(sed 's/\.*args.*//' <<<${!TASK_PARAMS})

    mkdir -p $RESULTS_PATH
    pushd .
    cd $command_path

    for i in $(seq 1 $NUM_EXP); do
	name=${RESULTS_PATH}$(date +%d-%m-%Y_%H-%M-%S)
	file_result=$name".txt"
	file_monitor=$name"_monitor.csv"
        
	flag_monitor=true

        ${TASKS[${task}]} $task $file_result &

        while $flag_monitor;
	do
	    last_line="$(tail -1 $file_result)"
	    if [ "$last_line" == "DONE!" ]; then
	        flag_monitor=false
	    else
	        status="$(nvidia-smi --query-gpu=temperature.gpu,utilization.gpu,memory.used --format=csv | tail -1)"
		echo "${status}" >> $file_monitor
            fi
	    sleep $MONITOR_INTERVAL
	done	

        sleep 5
    done

    chmod -R a+rwx $RESULTS_PATH
    echo "${task} ended."
    popd 

}


main() {
    for task in "${!TASKS[@]}"; do
	if [[ "${task,,}" == *"$TASK_NAME"* ]] || [ "$TASK_NAME" == "all" ]; then
		benchmark_tensorflow $task 
	fi	
    done

    chmod -R a+rwx /results/${SYSTEM}
}

main "$@"
