#!/bin/bash

temp=5
lr=0.03
n=2
path=results/

CIFAR_100i="--data_path data/ --save_path $path --batch_size 10 --cuda yes --seed 0 --n_epochs 1 --use 1 --inner_steps 2  --n_outer 2 --replay_batch_size 32 --train_csv data/mini_cl_train.csv --test_csv data/mini_cl_test.csv --temperature 2 --memory_strength 10"

CUDA_VISIBLE_DEVICES=0 python semi.py $CIFAR_100i --model semi_dualnet --lr $lr --n_runs 3 --n_memories 50 --augmentation --beta 0.05 --rho $3
