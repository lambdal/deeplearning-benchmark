# Guide


```
sudo apt install -y pssh
```

```
# Environment variables
# 1x
export NAME_HOSTFILE=hostfile_1xH100
export NUM_GPU=1
export SSH_KEY=/home/ubuntu/.ssh/ml.pem
export NAME_NGC=pytorch:24.03-py3
export NAME_TYPE=LambdaOD_1x_Texas
export NAME_GPU=H100_80GB_SXM5
export NAME_RESULTS=results_v2
export NAME_DATASET=all
export NAME_TASKS=all

# 2x
export NAME_HOSTFILE=hostfile_2xH100
export NUM_GPU=2
export SSH_KEY=/home/ubuntu/.ssh/ml.pem
export NAME_NGC=pytorch:24.03-py3
export NAME_TYPE=LambdaOD_2x_Texas
export NAME_GPU=H100_80GB_SXM5
export NAME_RESULTS=results_v2
export NAME_DATASET=all
export NAME_TASKS=all

# 4x
export NAME_HOSTFILE=hostfile_4xH100
export NUM_GPU=4
export SSH_KEY=/home/ubuntu/.ssh/ml.pem
export NAME_NGC=pytorch:24.03-py3
export NAME_TYPE=LambdaOD_4x_Texas
export NAME_GPU=H100_80GB_SXM5
export NAME_RESULTS=results_v2
export NAME_DATASET=all
export NAME_TASKS=all

# 8x
export NAME_HOSTFILE=hostfile_8xH100
export NUM_GPU=8
export SSH_KEY=/home/ubuntu/.ssh/ml.pem
export NAME_NGC=pytorch:24.03-py3
export NAME_TYPE=LambdaOD_8x_Texas
export NAME_GPU=H100_80GB_SXM5
export NAME_RESULTS=results_v2
export NAME_DATASET=all
export NAME_TASKS=all


export GIT_USERNAME=
export GIT_PASSWORD=
export BRANCH=v2

# Setup
bash ./parallel-setup.sh

# Run test
bash ./parallel-test.sh

# Commit results
bash ./parallel-commit.sh
```
