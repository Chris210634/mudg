#!/bin/bash

echo dataset = $1
echo iters = $2
echo waffles = $3
echo seed = $4

python main_crossdataset.laion.py --lr 0.00016 \
    --modelname ViT-L-14 --pretrained openai --d 768 \
    --iters_per_epoch $2 --dataset $1 --n_descriptors 8 \
    --n_epochs 1 --bs 64 --loss ce --temp 25 --score_averaging 1 \
    --tup_file cache/64nn_x_$3waffle/$1/result_tup_list.$1_64nn_x_$3waffle.clustered \
> final_train_$1_seed$4.L14.o