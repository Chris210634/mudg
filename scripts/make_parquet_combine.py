import os, sys
sys.path.append(os.getcwd())

import torch
import pandas as pd
from copy import deepcopy
from tqdm import tqdm
from argparse_parameters import get_arg_parser
import os

parser = get_arg_parser()
args = parser.parse_args()
NUMBER_OF_PROCESSES = args.nproc # also number of waffle augmentations
N_WAFFLE = args.n_waffle
NEAREST_NEIGHBORS = args.num_nearest_neighbors
pwd = args.index_directory

dfs = []
for sub_id in range(NUMBER_OF_PROCESSES):
    in_parquet_file = os.path.join(
        args.tmp_directory, 
        args.dataset,
        'nn_sub_{}.parquet'.format(sub_id)
    )
    dfi = pd.read_parquet(in_parquet_file)
    print('length of sub parquet file {} is {}'.format(sub_id, len(dfi)))
    dfs.append(dfi)
df = pd.concat(dfs)
print('length of combined parquet file is {}'.format(len(df)))

results_file = os.path.join(
               args.tmp_directory, args.dataset,
               'retrieval_results_condensed.list'
           )
_, _, master_list = torch.load(results_file)

print('check:', sum([len(hi) for hi in master_list]))
print('asserting that the above two numbers are equal')
assert sum([len(hi) for hi in master_list]) == len(df)

out_parquet_file = os.path.join(
    args.tmp_directory, 
    args.dataset,
    'nn_sub_combined.parquet'
)
df.to_parquet(out_parquet_file)

print('Done. saved sub parquet of length {} in file: {}'.format(
    len(df), out_parquet_file
))