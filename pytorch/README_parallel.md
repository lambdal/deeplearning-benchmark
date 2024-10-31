# Guide


```
sudo apt install -y pssh
```

```
# Environment variables
export BRANCH=gh200-od
export GIT_USERNAME=
export GIT_PASSWORD=

# 1x
export NAME_HOSTFILE=hostfile_1xGH200
export NUM_GPU=1
export SSH_KEY=/home/ubuntu/.ssh/chuan-china
export NAME_NGC=pytorch:24.03-py3
export NAME_TYPE=LambdaOD_1x_Test
export NAME_GPU=GH200_96GB
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


# Setup
bash ./parallel-setup.sh

# Run test
bash ./parallel-test.sh

# Commit results
bash ./parallel-commit.sh
```
