#!/bin/bash

#MY_PYTHON="CUDA_VISIBLE_DEVICES=4 python"
#beta 0.05 lr 0.01
gpu=$1
lr=0.03
n=2
path=results/

CIFAR_100i="--save_path $path --batch_size 10 --cuda yes --seed 0 --n_epochs 1 --use 1 --inner_steps 2  --n_outer $2 --replay_batch_size 10  --temperature 2 --memory_strength 10 --n_epochs 1"

CUDA_VISIBLE_DEVICES=$gpu python main.py $CIFAR_100i --model dualnet --lr $lr --n_runs 5 --n_memories 50 --augmentation --beta 0.05 --train_csv data/mini_cl_train.csv --test_csv data/mini_cl_test.csv
CUDA_VISIBLE_DEVICES=$gpu python main.py $CIFAR_100i --model dualnet --lr $lr --n_runs 5 --n_memories 50 --augmentation --beta 0.1 --train_csv data/core50_tr.csv --test_csv data/core50_te.csv --batch_size 32 --replay_batch_size 32



