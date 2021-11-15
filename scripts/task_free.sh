#!/bin/bash

#MY_PYTHON="CUDA_VISIBLE_DEVICES=4 python"
#beta 0.05 lr 0.01
gpu=$1
lr=0.03
beta=0.05
reg=10.0
temp=2.0
path=neurips


CIFAR_100i="--data_path data/ --save_path $path --batch_size 10 --cuda yes --seed 0 --n_epochs 1 --use 1 --inner_steps 2  --n_outer $2 --replay_batch_size 10 --train_csv data/mini_cl_train.csv --test_csv data/mini_cl_test.csv --memory_strength $reg"

CUDA_VISIBLE_DEVICES=$gpu python main.py $CIFAR_100i --model dualnet_tf --lr $lr --n_runs 3 --n_memories 100 --augmentation --beta $beta --temperature $temp --n_epochs 1
CUDA_VISIBLE_DEVICES=$gpu python main.py $CIFAR_100i --model dualnet_tf --lr $lr --n_runs 3 --n_memories 100 --augmentation --beta $beta --temperature $temp --n_epochs 1 --train_csv data/core50_tr.csv --test_csv data/core50_te.csv 
