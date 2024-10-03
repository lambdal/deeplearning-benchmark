#!/bin/bash

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

# 1) Parallelized remove sudo requirements on all nodes
echo "Finalize Docker setup on all nodes ..."
cmd_docker_setup=""
cmd_docker_setup+="sudo usermod -aG docker $USER && "
cmd_docker_setup+="newgrp docker"
parallel-ssh -v -P  -t 0 -e /home/$(whoami)/pssh-debug -x "-i $SSH_KEY" -h "$NAME_HOSTFILE" -i "$cmd_docker_setup" 

# 2) Parallelized pull docker image on all nodes
echo "Pull Docker image on all nodes ..."
cmd_docker_pull=""
cmd_docker_pull+="docker pull nvcr.io/nvidia/${NAME_NGC} 2>&1"
parallel-ssh -v -P  -t 0 -e /home/$(whoami)/pssh-debug -x "-i $SSH_KEY" -h "$NAME_HOSTFILE" -i $cmd_docker_pull

# 3) Parallelized git clone DeepLearningExamples on all nodes
echo "Clone DeepLearningExamples on all nodes ..."
cmd_clone=""
cmd_clone+="[ ! -d \"DeepLearningExamples\" ] && git clone https://github.com/LambdaLabsML/DeepLearningExamples.git DeepLearningExamples || echo \"Directory already exists. Skipping clone.\" && "
cmd_clone+="cd DeepLearningExamples && "
cmd_clone+="git checkout lambda/benchmark && "
cmd_clone+="git pull origin lambda/benchmark && "
cmd_clone+="cd .. && "
cmd_clone+="[ ! -d \"deeplearning-benchmark\" ] && git clone https://github.com/lambdal/deeplearning-benchmark.git deeplearning-benchmark || echo \"Directory already exists. Skipping clone.\" && "
cmd_clone+="cd deeplearning-benchmark && "
cmd_clone+="git checkout v2 && "
cmd_clone+="git pull origin v2"
parallel-ssh -v -P  -t 0 -e /home/$(whoami)/pssh-debug -x "-i $SSH_KEY" -h "$NAME_HOSTFILE" -i $cmd_clone


# 4) Prepare dataset
echo "Prepare dataset on all nodes"
cmd_dataset=""
cmd_dataset+="cd deeplearning-benchmark/pytorch && "
cmd_dataset+="docker run --gpus all --rm --shm-size=256g "
cmd_dataset+="-v ~/DeepLearningExamples/PyTorch:/workspace/benchmark "
cmd_dataset+="-v ~/data:/data "
cmd_dataset+="-v $(pwd)"/scripts":/scripts "
cmd_dataset+="nvcr.io/nvidia/${NAME_NGC} "
cmd_dataset+="/bin/bash -c \"cp -r /scripts/* /workspace;  ./run_prepare.sh $NAME_DATASET\""
echo $cmd_dataset
parallel-ssh -v -P  -t 0 -e /home/$(whoami)/pssh-debug -x "-i $SSH_KEY" -h "$NAME_HOSTFILE" -i $cmd_dataset

# 5) Create config
echo "Create config files on all nodes"
cmd_config=""
cmd_config+="cd deeplearning-benchmark/pytorch && "
if [ "$NUM_GPU" -ge 1 ]; then
    cmd_config+="cp scripts/config_v2/config_pytorch_${NAME_GPU}_v2.sh scripts/config_v2/config_pytorch_${NAME_TYPE}_${NAME_GPU}_\$(hostname)_v2.sh"
fi
if [ "$NUM_GPU" -ge 2 ]; then
    cmd_config+=" && cp scripts/config_v2/config_pytorch_2x${NAME_GPU}_v2.sh scripts/config_v2/config_pytorch_${NAME_TYPE}_2x${NAME_GPU}_\$(hostname)_v2.sh"
fi
if [ "$NUM_GPU" -ge 4 ]; then
    cmd_config+=" && cp scripts/config_v2/config_pytorch_4x${NAME_GPU}_v2.sh scripts/config_v2/config_pytorch_${NAME_TYPE}_4x${NAME_GPU}_\$(hostname)_v2.sh"
fi
if [ "$NUM_GPU" -ge 8 ]; then
    cmd_config+=" && cp scripts/config_v2/config_pytorch_8x${NAME_GPU}_v2.sh scripts/config_v2/config_pytorch_${NAME_TYPE}_8x${NAME_GPU}_\$(hostname)_v2.sh"
fi

echo $cmd_config
parallel-ssh -v -P  -t 0 -e /home/$(whoami)/pssh-debug -x "-i $SSH_KEY" -h "$NAME_HOSTFILE" -i $cmd_config