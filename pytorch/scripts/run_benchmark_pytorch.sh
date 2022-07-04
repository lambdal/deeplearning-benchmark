#!/bin/bash

SYSTEM=${1:-"2080Ti"}
TASK_NAME=${2:-"all"}
TIME_OUT=${3:-"1800"}

echo ${SYSTEM}
echo ${NUM_GPU}

declare -A TASKS=(
    [PyTorch_SSD_FP32]=benchmark_pytorch_ssd
    [PyTorch_SSD_AMP]=benchmark_pytorch_ssd
    [PyTorch_resnet50_FP32]=benchmark_pytorch_resnet50
    [PyTorch_resnet50_FP16]=benchmark_pytorch_resnet50
    [PyTorch_resnet50_AMP]=benchmark_pytorch_resnet50
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

main() {
    for task in "${!TASKS[@]}"; do
	if [[ "${task,,}" == *"$TASK_NAME"* ]] || [ "$TASK_NAME" == "all" ]; then
		timeout -s SIGKILL $TIME_OUT bash ./benchmark_pytorch.sh $SYSTEM ${TASKS[${task}]} $task $TIME_OUT
	fi	
    done

    chmod -R a+rwx /results/${SYSTEM}
}

main "$@"
