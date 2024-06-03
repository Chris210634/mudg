#!/bin/bash

echo dataset = $1
echo iters = $2
echo waffles = $3
echo seed = $4

python main_crossdataset.laion.py --lr 0.00016 \
    --init_lam 0.2 --n_descriptors 16 --teacher_temp 10 --train_with_descriptors 1 \
    --modelname ViT-L-14 --pretrained openai --d 768 \
    --descriptor_file cache/better_descriptors_sorted_ViT-L-14_$1.list \
    --iters_per_epoch $2 --dataset $1 --n_descriptors 8 \
    --n_epochs 1 --bs 64 --loss ce --temp 25 --score_averaging 1 \
    --tup_file cache_paired/64nn_x_$3waffle/$1/result_tup_list.$1_64nn_x_$3waffle.clustered \
> final_train_$1_initreg_seed$4_paired.L14.o