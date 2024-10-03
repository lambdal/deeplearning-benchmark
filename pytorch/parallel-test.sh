#!/bin/bash

# Steps: 
# 1) Parallelized remove sudo requirements on all nodes

# Check if CONFIG_PATH is set and if the file exists
if [ -z "${NAME_HOSTFILE}" ]; then
    echo "NAME_HOSTFILE hasn't been set or found."
    exit 1
fi

if [ ! -f "${NAME_HOSTFILE}" ]; then
    echo "File specified by CONFIG_PATH (${NAME_HOSTFILE}) doesn't exist."
    exit 1
fi

# Read hosts from the hostfile into an array
readarray -t workers < $NAME_HOSTFILE
echo "Here are the workers:"
for worker in "${workers[@]}"; do
    echo $worker    
done

# 4) Test dataset
echo "Test dataset on all nodes"
cmd_dataset=""
cmd_dataset+="du -hs ~/data "
echo $cmd_dataset
parallel-ssh -v -P  -t 0 -e /home/ubuntu/pssh-debug -x "-i $SSH_KEY" -h "$NAME_HOSTFILE" -i $cmd_dataset

# 5) Run benchmark
echo "Running benchmark"


if [ "$NUM_GPU" -ge 1 ]; then
    cmd_benchmark=""
    cmd_benchmark+="cd deeplearning-benchmark/pytorch && "
    cmd_benchmark+="mkdir -p ${NAME_RESULTS} && "
    cmd_benchmark+="docker run --rm --shm-size=1024g "
    cmd_benchmark+="--gpus all "
    cmd_benchmark+="-v ~/DeepLearningExamples/PyTorch:/workspace/benchmark "
    cmd_benchmark+="-v ~/data:/data "
    cmd_benchmark+="-v \$(pwd)\"/scripts\":/scripts "
    cmd_benchmark+="-v \$(pwd)\"/${NAME_RESULTS}\":/results "
    cmd_benchmark+="nvcr.io/nvidia/${NAME_NGC} "
    cmd_benchmark+="/bin/bash -c \"cp -r /scripts/* /workspace; ./run_benchmark.sh ${NAME_TYPE}_${NAME_GPU}_\$(hostname)_v2 ${NAME_TASKS} 3000\""
    echo $cmd_benchmark
    parallel-ssh -v -P  -t 0 -e /home/$(whoami)/pssh-debug -x "-i $SSH_KEY" -h "$NAME_HOSTFILE" -i $cmd_benchmark
fi
if [ "$NUM_GPU" -ge 2 ]; then
    cmd_benchmark=""
    cmd_benchmark+="cd deeplearning-benchmark/pytorch && "
    cmd_benchmark+="mkdir -p ${NAME_RESULTS} && "
    cmd_benchmark+="docker run --rm --shm-size=1024g "
    cmd_benchmark+="--gpus all "
    cmd_benchmark+="-v ~/DeepLearningExamples/PyTorch:/workspace/benchmark "
    cmd_benchmark+="-v ~/data:/data "
    cmd_benchmark+="-v \$(pwd)\"/scripts\":/scripts "
    cmd_benchmark+="-v \$(pwd)\"/${NAME_RESULTS}\":/results "
    cmd_benchmark+="nvcr.io/nvidia/${NAME_NGC} "
    cmd_benchmark+="/bin/bash -c \"cp -r /scripts/* /workspace; ./run_benchmark.sh ${NAME_TYPE}_2x${NAME_GPU}_\$(hostname)_v2 ${NAME_TASKS} 3000\""
    echo $cmd_benchmark
    parallel-ssh -v -P  -t 0 -e /home/$(whoami)/pssh-debug -x "-i $SSH_KEY" -h "$NAME_HOSTFILE" -i $cmd_benchmark
fi
if [ "$NUM_GPU" -ge 4 ]; then
    cmd_benchmark=""
    cmd_benchmark+="cd deeplearning-benchmark/pytorch && "
    cmd_benchmark+="mkdir -p ${NAME_RESULTS} && "
    cmd_benchmark+="docker run --rm --shm-size=1024g "
    cmd_benchmark+="--gpus all "
    cmd_benchmark+="-v ~/DeepLearningExamples/PyTorch:/workspace/benchmark "
    cmd_benchmark+="-v ~/data:/data "
    cmd_benchmark+="-v \$(pwd)\"/scripts\":/scripts "
    cmd_benchmark+="-v \$(pwd)\"/${NAME_RESULTS}\":/results "
    cmd_benchmark+="nvcr.io/nvidia/${NAME_NGC} "
    cmd_benchmark+="/bin/bash -c \"cp -r /scripts/* /workspace; ./run_benchmark.sh ${NAME_TYPE}_4x${NAME_GPU}_\$(hostname)_v2 ${NAME_TASKS} 3000\""
    echo $cmd_benchmark
    parallel-ssh -v -P  -t 0 -e /home/$(whoami)/pssh-debug -x "-i $SSH_KEY" -h "$NAME_HOSTFILE" -i $cmd_benchmark
fi
if [ "$NUM_GPU" -ge 8 ]; then
    cmd_benchmark=""
    cmd_benchmark+="cd deeplearning-benchmark/pytorch && "
    cmd_benchmark+="mkdir -p ${NAME_RESULTS} && "
    cmd_benchmark+="docker run --rm --shm-size=1024g "
    cmd_benchmark+="--gpus all "
    cmd_benchmark+="-v ~/DeepLearningExamples/PyTorch:/workspace/benchmark "
    cmd_benchmark+="-v ~/data:/data "
    cmd_benchmark+="-v \$(pwd)\"/scripts\":/scripts "
    cmd_benchmark+="-v \$(pwd)\"/${NAME_RESULTS}\":/results "
    cmd_benchmark+="nvcr.io/nvidia/${NAME_NGC} "
    cmd_benchmark+="/bin/bash -c \"cp -r /scripts/* /workspace; ./run_benchmark.sh ${NAME_TYPE}_8x${NAME_GPU}_\$(hostname)_v2 ${NAME_TASKS} 3000\""
    echo $cmd_benchmark
    parallel-ssh -v -P  -t 0 -e /home/$(whoami)/pssh-debug -x "-i $SSH_KEY" -h "$NAME_HOSTFILE" -i $cmd_benchmark
fi

