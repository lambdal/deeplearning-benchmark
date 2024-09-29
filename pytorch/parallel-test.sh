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

# # 1) Parallelized remove sudo requirements on all nodes
# echo "Finalize Docker setup on all nodes ..."
# cmd_docker_setup=""
# cmd_docker_setup+="sudo usermod -aG docker $USER && "
# cmd_docker_setup+="newgrp docker"
# parallel-ssh -v -P  -t 0 -e /home/ubuntu/pssh-debug -x "-i $SSH_KEY" -h "$NAME_HOSTFILE" -i "$cmd_docker_setup" 

# # 2) Parallelized pull docker image on all nodes
# echo "Pull Docker image on all nodes ..."
# cmd_docker_pull=""
# cmd_docker_pull+="docker pull nvcr.io/nvidia/${NAME_NGC} 2>&1"
# parallel-ssh -v -P  -t 0 -e /home/ubuntu/pssh-debug -x "-i $SSH_KEY" -h "$NAME_HOSTFILE" -i $cmd_docker_pull

# # 3) Parallelized git clone DeepLearningExamples on all nodes
# echo "Clone DeepLearningExamples on all nodes ..."
# cmd_clone=""
# cmd_clone+="[ ! -d \"DeepLearningExamples\" ] && git clone https://github.com/LambdaLabsML/DeepLearningExamples.git DeepLearningExamples || echo \"Directory already exists. Skipping clone.\" && "
# cmd_clone+="cd DeepLearningExamples && "
# cmd_clone+="git checkout lambda/benchmark && "
# cmd_clone+="cd .. && "
# cmd_clone+="[ ! -d \"deeplearning-benchmark\" ] && git clone https://github.com/lambdal/deeplearning-benchmark.git deeplearning-benchmark || echo \"Directory already exists. Skipping clone.\" && "
# cmd_clone+="cd deeplearning-benchmark && "
# cmd_clone+="git checkout v2"
# parallel-ssh -v -P  -t 0 -e /home/ubuntu/pssh-debug -x "-i $SSH_KEY" -h "$NAME_HOSTFILE" -i $cmd_clone


# 4) Test dataset
echo "Test dataset on all nodes"
cmd_dataset=""
cmd_dataset+="du -hs ~/data "
echo $cmd_dataset
parallel-ssh -v -P  -t 0 -e /home/ubuntu/pssh-debug -x "-i $SSH_KEY" -h "$NAME_HOSTFILE" -i $cmd_dataset