#!/bin/bash

SYSTEM=${1:-"2080Ti"}
TASK_NAME=${2:-"all"}

source config/config_tensorflow_${SYSTEM}.sh

echo ${SYSTEM}
echo ${NUM_GPU}


declare -A TASKS=(
    [TensorFlow_resnet50_FP32]=benchmark_tensorflow_resnet50
    [TensorFlow_resnet50_FP16]=benchmark_tensorflow_resnet50
)


benchmark_tensorflow_resnet50() {
    
    local task="$1"
    local result="$2"

    echo "benchmark_tensorflow_resnet50 is called"

    TASK_PARAMS=${task}_PARAMS[@]
    local command_para=$(sed 's/.*args //' <<<${!TASK_PARAMS})

    mpiexec --allow-run-as-root --bind-to socket -np ${NUM_GPU} python ./main.py \
    ${command_para} |& tee ${result}   

    rm -rf output

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
