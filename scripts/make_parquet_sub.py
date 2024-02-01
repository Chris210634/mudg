import os, sys
sys.path.append(os.getcwd())

import torch
import pandas as pd
from copy import deepcopy
from tqdm import tqdm
import sys
from argparse_parameters import get_arg_parser
import os

def _get_start_end(proc_id, ntasks, nproc):
    assert proc_id < ntasks
    assert nproc <= ntasks
    start = (ntasks // nproc) * proc_id
    end = (ntasks // nproc) * (proc_id + 1)
    if proc_id == nproc-1:
        end = ntasks
    return start, end

parser = get_arg_parser()
args = parser.parse_args()
NUMBER_OF_PROCESSES = args.nproc # also number of waffle augmentations
N_WAFFLE = args.n_waffle
NEAREST_NEIGHBORS = args.num_nearest_neighbors
pwd = args.index_directory

results_file = os.path.join(
               args.tmp_directory, args.dataset,
               'retrieval_results_condensed.list'
           )

_, _, master_list = torch.load(results_file)
file_paths, lengths, index_list = torch.load(pwd + 'parquet_lengths.list')

# split the parquet querying evenly across 2100+ parquet files
begin, end = _get_start_end(args.proc_id, len(file_paths), NUMBER_OF_PROCESSES)

results = []
for i in tqdm(range(begin, end)):
    __indices = master_list[i].numpy()
    if len(__indices) > 0:
        dirname = 'laion-2B-en/'
        df = pd.read_parquet(os.path.join(pwd, dirname, file_paths[i]))
        results.append(deepcopy(df.iloc[__indices, :]))
    
df = pd.concat(results)    
out_parquet_file = os.path.join(
    args.tmp_directory, 
    args.dataset,
    'nn_sub_{}.parquet'.format(args.proc_id)
)
df.to_parquet(out_parquet_file)

print('Done. saved sub parquet of length {} in file: {}'.format(
    len(df), out_parquet_file
))

    
    
    
    
    
    
    
    
    