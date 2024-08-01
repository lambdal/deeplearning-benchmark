# Guide


```
sudo apt install -y pssh
```

```
export NAME_HOSTFILE=hostfile
export NAME_NGC=pytorch:24.03-py3
export NAME_DATASET=all
export NAME_TYPE=LambdaH200BM
export NAME_RESULTS=results_v2
export NAME_TASKS=bert
export NAME_GPU=H100_140GB_SXM

bash ./parallel-setup.sh


export NAME_HOSTFILE=hostfile
export SSH_KEY=/home/ubuntu/.ssh/ml.pem
export NAME_NGC=pytorch:24.03-py3
export NAME_DATASET=all

bash ./parallel-test.sh



export NAME_HOSTFILE=hostfile
export SSH_KEY=/home/ubuntu/.ssh/ml.pem
export GIT_USERNAME=chuanli11
export GIT_PASSWORD=
export BRANCH=v2

bash ./parallel-commit.sh
```
