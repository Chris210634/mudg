#!/bin/bash

echo dataset = $1
echo iters = $2
echo waffles = $3
echo seed = $4

echo 64nn_x_$3waffle/$1/result_tup_list.$1_64nn_x_$3waffle.clustered
python main_crossdataset.laion.py --lr 0.00064 \
    --teacher_temp 75 \
    --iters_per_epoch $2 --dataset $1 --n_descriptors 8 \
    --n_epochs 1 --bs 128 --loss ce --temp 25 --score_averaging 1 \
    --tup_file /projectnb/textconv/cliao25/LLAVA/word_soups/cache/64nn_x_$3waffle/$1/result_tup_list.$1_64nn_x_$3waffle.clustered \
> final_train_$1_seed$4_soft.B16.o