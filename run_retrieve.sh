#!/bin/bash

# first  argument: datasetname                 e.g. ImageNet
# second argument: number of waffles           (should default to 8)
# third  argument: number of nearest neighbors (should default to 64)
# fourth argument: number of clusteres         (should defaul to 48) 

export N_WAFFLE=$2
export N_NEAREST_NEIGHBORS=$3
export N_NUMBER_CLUSTERS=$4
export NUMBER_OF_PROCESSES=8
export TMP_DIR="cache/$3nn_x_$2waffle"

python scripts/setup.py --dataset $1 --n_waffle ${N_WAFFLE} --tmp_directory ${TMP_DIR}
# output: TMP_DIR/$1/waffle.fs

sh scripts/retrieve.sh $1
# outputs:
# TMP_DIR/$1/retrieval_results_{}.list
# TMP_DIR/$1/retrieval_results_condensed.list

sh scripts/make_parquet.sh $1
# outputs:
# TMP_DIR/$1/nn_sub_{}.parquet
# TMP_DIR/$1/nn_sub_combined.parquet

# downlaod images
! ~/.local/bin/img2dataset \
         --url_list $TMP_DIR/$1/nn_sub_combined.parquet \
         --input_format "parquet"\
         --url_col "url" \
         --caption_col "caption" \
         --output_format webdataset \
         --output_folder $TMP_DIR/$1/laion \
         --processes_count 8 \
         --thread_count 64 \
         --image_size 2048 \
         --resize_only_if_bigger=True \
         --resize_mode="keep_ratio" \
         --skip_reencode=True \
         --enable_wandb False
# output tar files into: $TMP_DIR/$1/laion
         
# untar images into: $TMP_DIR/$1/laion_jpegs
mkdir $TMP_DIR/$1/laion_jpegs
for file in $TMP_DIR/$1/laion/*.tar; do
    echo tar -xf "$file" -C $TMP_DIR/$1/laion_jpegs
    tar -xf "$file" -C $TMP_DIR/$1/laion_jpegs
done

python scripts/prepare_kmeans_dataset.py --dataset $1 --n_waffle ${N_WAFFLE} \
    --num_nearest_neighbors ${N_NEAREST_NEIGHBORS} --n_clusters ${N_NUMBER_CLUSTERS} --tmp_directory ${TMP_DIR}
# outputs:
# TMP_DIR/$1/result_tup_list.$1_64nn_x_8waffle
# TMP_DIR/$1/result_tup_list.$1_64nn_x_8waffle.clustered
