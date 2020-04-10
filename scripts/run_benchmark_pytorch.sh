#!/bin/bash

SYSTEM=${1:-"2080Ti"}

source config/config_pytorch_${SYSTEM}.sh

echo ${SYSTEM}
echo ${NUM_GPU}

#declare -A TASKS=(
#    [PyTorch_SSD_FP32]=benchmark_pytorch_ssd
#    [PyTorch_SSD_AMP]=benchmark_pytorch_ssd
#)

declare -A TASKS=(
    [PyTorch_SSD_FP32]=benchmark_pytorch_ssd
    [PyTorch_SSD_AMP]=benchmark_pytorch_ssd
    [PyTorch_maskrcnn_FP32]=benchmark_pytorch_maskrcnn
    [PyTorch_maskrcnn_FP16]=benchmark_pytorch_maskrcnn
    [PyTorch_gnmt_FP32]=benchmark_pytorch_gnmt
    [PyTorch_gnmt_FP16]=benchmark_pytorch_gnmt
    [PyTorch_ncf_FP32]=benchmark_pytorch_ncf
    [PyTorch_ncf_FP16]=benchmark_pytorch_ncf
    [PyTorch_transformerxlbase_FP32]=benchmark_pytorch_transformerxl
    [PyTorch_transformerxlbase_FP16]=benchmark_pytorch_transformerxl
    [PyTorch_transformerxllarge_FP32]=benchmark_pytorch_transformerxl
    [PyTorch_transformerxllarge_FP16]=benchmark_pytorch_transformerxl
    [PyTorch_tacotron2_FP32]=benchmark_pytorch_tacotron2
    [PyTorch_tacotron2_FP16]=benchmark_pytorch_tacotron2
    [PyTorch_waveglow_FP32]=benchmark_pytorch_tacotron2
    [PyTorch_waveglow_FP16]=benchmark_pytorch_tacotron2
    [PyTorch_bert_base_squad_FP32]=benchmark_pytorch_bert_squad
    [PyTorch_bert_base_squad_FP16]=benchmark_pytorch_bert_squad    
    [PyTorch_bert_large_squad_FP32]=benchmark_pytorch_bert_squad
    [PyTorch_bert_large_squad_FP16]=benchmark_pytorch_bert_squad
)

#declare -A TASKS=(
#    [PyTorch_SSD_FP32]=benchmark_pytorch_ssd
#    [PyTorch_SSD_AMP]=benchmark_pytorch_ssd
#    [PyTorch_resnet50_FP32]=benchmark_pytorch_resnet50
#    [PyTorch_resnet50_FP16]=benchmark_pytorch_resnet50
#    [PyTorch_resnet50_AMP]=benchmark_pytorch_resnet50
#    [PyTorch_maskrcnn_FP32]=benchmark_pytorch_maskrcnn
#    [PyTorch_maskrcnn_FP16]=benchmark_pytorch_maskrcnn
#    [PyTorch_gnmt_FP32]=benchmark_pytorch_gnmt
#    [PyTorch_gnmt_FP16]=benchmark_pytorch_gnmt
#    [PyTorch_ncf_FP32]=benchmark_pytorch_ncf
#    [PyTorch_ncf_FP16]=benchmark_pytorch_ncf
#    [PyTorch_transformerxlbase_FP32]=benchmark_pytorch_transformerxl
#    [PyTorch_transformerxlbase_FP16]=benchmark_pytorch_transformerxl
#    [PyTorch_transformerxllarge_FP32]=benchmark_pytorch_transformerxl
#    [PyTorch_transformerxllarge_FP16]=benchmark_pytorch_transformerxl
#    [PyTorch_tacotron2_FP32]=benchmark_pytorch_tacotron2
#    [PyTorch_tacotron2_FP16]=benchmark_pytorch_tacotron2
#    [PyTorch_waveglow_FP32]=benchmark_pytorch_tacotron2
#    [PyTorch_waveglow_FP16]=benchmark_pytorch_tacotron2
#    [PyTorch_bert_base_squad_FP32]=benchmark_pytorch_bert_squad
#    [PyTorch_bert_base_squad_FP16]=benchmark_pytorch_bert_squad    
#    [PyTorch_bert_large_squad_FP32]=benchmark_pytorch_bert_squad
#    [PyTorch_bert_large_squad_FP16]=benchmark_pytorch_bert_squad
#)

benchmark_pytorch_ssd() {
    
    local task="$1"
    local result="$2"

    TASK_PARAMS=${task}_PARAMS[@]
    local command_para=$(sed 's/.*args //' <<<${!TASK_PARAMS})

    python -m torch.distributed.launch --nproc_per_node=${NUM_GPU} main.py \
    --mode benchmark-training ${command_para} |& tee ${result}  
    echo "DONE!" >> ${result}
}


benchmark_pytorch_resnet50() {
    
    local task="$1"
    local result="$2"

    TASK_PARAMS=${task}_PARAMS[@]
    local command_para=$(sed 's/.*args //' <<<${!TASK_PARAMS})

    python ./multiproc.py --nproc_per_node ${NUM_GPU} ./main.py \
    ${command_para} |& tee ${result}   
    echo "DONE!" >> ${result}
}


benchmark_pytorch_maskrcnn() {

    local task="$1"
    local result="$2"

    TASK_PARAMS=${task}_PARAMS[@]
    local command_para=$(sed 's/.*args //' <<<${!TASK_PARAMS})

    GLOBAL_BATCH=`echo ${!TASK_PARAMS} | grep -oP '(?<=SOLVER.IMS_PER_BATCH )\w+'`

    python -m torch.distributed.launch --nproc_per_node=${NUM_GPU} tools/train_net.py \
    --skip-test \
    ${command_para} \
    | tee $result
    
    time=`cat $result | grep -F 'maskrcnn_benchmark.trainer INFO: Total training time' | tail -n 1 | awk -F'(' '{print $2}' | awk -F' s ' '{print $1}' | egrep -o [0-9.]+`
    statement=`cat $result | grep -F 'maskrcnn_benchmark.trainer INFO: Total training time' | tail -n 1`
    calc=$(echo $time 1.0 $GLOBAL_BATCH | awk '{ printf "%f", $2 * $3 / $1 }')
    
    echo "Training perf is: "$calc" FPS" >> ${result}
    echo "DONE!" >> ${result}
    rm /results/*.txt
    rm /results/*.pth
    rm /results/*checkpoint* 
}


benchmark_pytorch_gnmt() {

    local task="$1"
    local result="$2"

    TASK_PARAMS=${task}_PARAMS[@]
    local command_para=$(sed 's/.*args //' <<<${!TASK_PARAMS})

    python3 -m launch --nproc_per_node=${NUM_GPU} train.py ${command_para} |& tee ${result}
    echo "DONE!" >> ${result}
}


benchmark_pytorch_ncf() {
    
    local task="$1"
    local result="$2"

    TASK_PARAMS=${task}_PARAMS[@]
    local command_para=$(sed 's/.*args //' <<<${!TASK_PARAMS})

    python -m torch.distributed.launch --nproc_per_node=${NUM_GPU} ncf.py ${command_para} |& tee ${result}
    echo "DONE!" >> ${result}
}


benchmark_pytorch_transformerxl() {
    
    local task="$1"
    local result="$2"

    TASK_PARAMS=${task}_PARAMS[@]
    local command_para=$(sed 's/.*args //' <<<${!TASK_PARAMS})

    python -m torch.distributed.launch --nproc_per_node=${NUM_GPU} train.py ${command_para} |& tee ${result}
    echo "DONE!" >> ${result}
}


benchmark_pytorch_tacotron2() {
    
    local task="$1"
    local result="$2"

    TASK_PARAMS=${task}_PARAMS[@]
    local command_para=$(sed 's/.*args //' <<<${!TASK_PARAMS})

    python -m multiproc ${NUM_GPU} train.py \
    ${command_para}  |& tee ${result}
    echo "DONE!" >> ${result}
}


benchmark_pytorch_bert_squad() {

    local task="$1"
    local result="$2"

    TASK_PARAMS=${task}_PARAMS[@]
    local command_para=$(sed 's/.*args //' <<<${!TASK_PARAMS})

    bash scripts/run_squad.sh ${command_para} |& tee ${result}
    echo "DONE!" >> ${result}
}


benchmark_pytorch() {

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
        benchmark_pytorch $task 
    done

    chmod -R a+rwx /results/${SYSTEM}
}

main "$@"
