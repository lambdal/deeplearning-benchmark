#!/bin/bash

# Steps: 
# 1) global config ssh 

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

# 1) global config ssh 
echo "Config SSH ..."
cmd_ssh_config=""
cmd_ssh_config+="git config --global user.name chuanli11 && "
cmd_ssh_config+="git config --global user.email cl.chuanli@gmail.com"
parallel-ssh -v -P  -t 0 -e /home/ubuntu/pssh-debug -x "-i $SSH_KEY" -h "$NAME_HOSTFILE" -i $cmd_ssh_config

# 2) commit results from each worker
echo "Commit results ..."

cmd_ssh_commit=""
cmd_ssh_commit+="cd deeplearning-benchmark/pytorch && "
# cmd_ssh_commit+="git config pull.rebase false && "
cmd_ssh_commit+="git pull origin v2 && "
cmd_ssh_commit+="git add scripts/config_v2 ; "
cmd_ssh_commit+="git add results_v2 ; "
cmd_ssh_commit+="git commit -m \"Update from \$(hostname) \" ; "
cmd_ssh_commit+="git remote set-url origin https://$GIT_USERNAME:$GIT_PASSWORD@github.com/lambdal/deeplearning-benchmark.git ; "
cmd_ssh_commit+="git push origin $BRANCH"

for worker in "${workers[@]}"; do
    # echo $cmd_ssh_commit
    ssh -i $SSH_KEY $worker $cmd_ssh_commit
done