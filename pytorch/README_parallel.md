# Guide


```
sudo apt install -y pssh
```

```
# Environment variables
export BRANCH=master
export GIT_USERNAME=
export GIT_PASSWORD=
export SSH_KEY=

export NAME_NGC=pytorch:24.10-py3
export NAME_TYPE=MyTestInstance
export NAME_RESULTS=results_v2
export NAME_DATASET=all
export NAME_TASKS=all

# 1xH100 
export NAME_HOSTFILE=hostfile_1xH200
export NUM_GPU=1
export NAME_GPU=H100_80GB_SXM5


# 2xH100
export NAME_HOSTFILE=hostfile_2xH100
export NUM_GPU=2
export NAME_GPU=H100_80GB_SXM5

# 4xH100
export NAME_HOSTFILE=hostfile_4xH100
export NUM_GPU=4
export NAME_GPU=H100_80GB_SXM5

# 8xH100
export NAME_HOSTFILE=hostfile_8xH100
export NUM_GPU=8
export NAME_GPU=H100_80GB_SXM5


# Setup
bash ./parallel-setup.sh

# Run test
bash ./parallel-test.sh

# Commit results
bash ./parallel-commit.sh
```
