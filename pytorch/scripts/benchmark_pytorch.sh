#!/bin/bash

SYSTEM=${1:-"2080Ti"}
func=${2:-"benchmark_pytorch_ncf"}
task=${3:-"PyTorch_ncf_FP32"}

source config_v1/config_pytorch_${SYSTEM}.sh

benchmark_pytorch_ssd() {
    
    local task="$1"
    local result="$2"

    TASK_PARAMS=${task}_PARAMS[@]
    local command_para=$(sed 's/.*args //' <<<${!TASK_PARAMS})

    python -m torch.distributed.launch --nproc_per_node=${NUM_GPU} main.py \
    --mode benchmark-training ${command_para} |& tee ${result} 

    if ! grep -q "RuntimeError" "$result"; then
        echo "DONE!" >> ${result}
    fi    
}


benchmark_pytorch_resnet50() {
    
    local task="$1"
    local result="$2"

    TASK_PARAMS=${task}_PARAMS[@]
    local command_para=$(sed 's/.*args //' <<<${!TASK_PARAMS})

    python ./multiproc.py --nproc_per_node ${NUM_GPU} ./main.py \
    ${command_para} |& tee ${result}

    if ! grep -q "RuntimeError" "$result"; then
        echo "DONE!" >> ${result}
    fi 
}


benchmark_pytorch_maskrcnn() {

    echo "Skip MaskRCNN until maskrcnn_benchmark can be built."
    return 1

    local task="$1"
    local result="$2"

    TASK_PARAMS=${task}_PARAMS[@]
    local command_para=$(sed 's/.*args //' <<<${!TASK_PARAMS})

    GLOBAL_BATCH=`echo ${!TASK_PARAMS} | grep -oP '(?<=SOLVER.IMS_PER_BATCH )\w+'`

    # python setup.py install
    # pip install -r requirements.txt

    python -m torch.distributed.launch --nproc_per_node=${NUM_GPU} --use_env tools/train_net.py \
    --skip-test \
    ${command_para} \
    | tee $result
    
    time=`cat $result | grep -F 'maskrcnn_benchmark.trainer INFO: Total training time' | tail -n 1 | awk -F'(' '{print $2}' | awk -F' s ' '{print $1}' | egrep -o [0-9.]+`
    statement=`cat $result | grep -F 'maskrcnn_benchmark.trainer INFO: Total training time' | tail -n 1`
    calc=$(echo $time 1.0 $GLOBAL_BATCH | awk '{ printf "%f", $2 * $3 / $1 }')
    
    echo "Training perf is: "$calc" FPS" >> ${result}
    if ! grep -q "RuntimeError" "$result"; then
        echo "DONE!" >> ${result}
    fi 
    rm /results/*.txt
    rm /results/*.pth
    rm /results/*checkpoint* 
}


benchmark_pytorch_gnmt() {

    local task="$1"
    local result="$2"

    pip install -r requirements.txt

    TASK_PARAMS=${task}_PARAMS[@]
    local command_para=$(sed 's/.*args //' <<<${!TASK_PARAMS})
    python3 -m torch.distributed.launch --nproc_per_node=${NUM_GPU} train.py ${command_para} |& tee ${result}

    if ! grep -q "RuntimeError" "$result"; then
        echo "DONE!" >> ${result}
    fi 
}


benchmark_pytorch_ncf() {
    
    local task="$1"
    local result="$2"

    TASK_PARAMS=${task}_PARAMS[@]
    local command_para=$(sed 's/.*args //' <<<${!TASK_PARAMS})

    python -m torch.distributed.launch --nproc_per_node=${NUM_GPU} --use_env ncf.py ${command_para} |& tee ${result}

    if ! grep -q "RuntimeError" "$result"; then
        echo "DONE!" >> ${result}
    fi 
}


benchmark_pytorch_transformerxl() {
    
    local task="$1"
    local result="$2"

    TASK_PARAMS=${task}_PARAMS[@]
    local command_para=$(sed 's/.*args //' <<<${!TASK_PARAMS})

    python -m torch.distributed.launch --nproc_per_node=${NUM_GPU} train.py ${command_para} |& tee ${result}


    if ! grep -q "RuntimeError" "$result"; then
        echo "DONE!" >> ${result}
    fi
}


benchmark_pytorch_tacotron2() {
    
    local task="$1"
    local result="$2"

    pip install -r requirements.txt

    TASK_PARAMS=${task}_PARAMS[@]
    local command_para=$(sed 's/.*args //' <<<${!TASK_PARAMS})

    python -m multiproc ${NUM_GPU} train.py \
    ${command_para}  |& tee ${result}
    

    if ! grep -q "RuntimeError" "$result"; then
        echo "DONE!" >> ${result}
    fi
}


benchmark_pytorch_bert_squad() {

    local task="$1"
    local result="$2"
    
    pip install -r requirements.txt

    TASK_PARAMS=${task}_PARAMS[@]
    local command_para=$(sed 's/.*args //' <<<${!TASK_PARAMS})

    bash scripts/run_squad.sh ${command_para} |& tee ${result}
    

    if ! grep -q "RuntimeError" "$result"; then
        echo "DONE!" >> ${result}
    fi
}


echo "${task} started: "

RESULTS_PATH=/results/${SYSTEM}/${task}/
TASK_PARAMS=${task}_PARAMS[@]
MONITOR_INTERVAL=2

command_path=$(sed 's/\.*args.*//' <<<${!TASK_PARAMS})

mkdir -p $RESULTS_PATH

pushd .
cd $command_path

for i in $(seq 1 $NUM_EXP); do
    name=${RESULTS_PATH}$(date +%d-%m-%Y_%H-%M-%S)
    file_result=$name".txt"
    $func $task $file_result
    sleep 5
done

chmod -R a+rwx $RESULTS_PATH
echo "${task} ended."
popd 

