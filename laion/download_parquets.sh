#!/bin/bash
for n in {0000..2313}; 
do
    wget https://huggingface.co/datasets/laion/laion2b-en-vit-l-14-embeddings/resolve/refs%2Fconvert%2Fparquet/default/train/$n.parquet
done