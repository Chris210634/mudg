import os, sys
sys.path.append(os.getcwd())

import torch, faiss, open_clip
from PIL import Image
import urllib.request
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
import sys
from argparse_parameters import get_arg_parser
import os

parser = get_arg_parser()
args = parser.parse_args()
NUMBER_OF_PROCESSES = args.nproc # also number of waffle augmentations
N_WAFFLE = args.n_waffle
NEAREST_NEIGHBORS = args.num_nearest_neighbors
pwd = args.index_directory

################################################################################
save_retrieval_results_file = os.path.join(
               args.tmp_directory, args.dataset,
               'retrieval_results_{}.list'
           )
rl = []
for i in range(NUMBER_OF_PROCESSES):
    sim_matrix, index_matrix = torch.load(save_retrieval_results_file.format(i))
    rl.append((sim_matrix, index_matrix))
    
sims = torch.cat([rli[0] for rli in rl], dim=1)
indices = torch.cat([rli[1] for rli in rl], dim=1)

assert indices.shape[1] == NEAREST_NEIGHBORS * N_WAFFLE

################################################################################

unique_indices = indices.unique().sort().values

################################################################################

## labels based on max retrieved rank
labels = torch.zeros((len(unique_indices),)).long() 
# indices.shape: torch.Size([1000, 512])

labels = labels.cuda()
indices = indices.cuda()
unique_indices = unique_indices.cuda()

for i in tqdm(range(len(labels))):
    v = (indices == unique_indices[i])
    
    v_nonzero = v.nonzero()
    _m = v_nonzero[:,0].unique()
    
    if len(_m) > 1: # label conflict
        unique_labels = _m.view(-1)
        scores = torch.zeros((len(unique_labels),)).long() # score based on rank
        for __i, __lab in enumerate(unique_labels):
            scores[__i] = (NEAREST_NEIGHBORS - (v_nonzero[:,1][(v_nonzero[:,0]) == __lab] % NEAREST_NEIGHBORS) ).sum()
        rand = torch.randperm(len(unique_labels))
        ii = scores[rand].argmax()
        labels[i] = unique_labels[rand][ii]
        
    else:
        labels[i] = _m[0].item()
        
print('number of unique entries from waffle retrieval:', unique_indices.shape[0])

_v, _i = unique_indices.sort()
sorted_unique_indices = _v
labels = labels[_i.cpu()]

file_paths, lengths, index_list = torch.load(pwd + 'parquet_lengths.list')
master_list = []
cum_length = 0
_index_list = list(sorted_unique_indices)[::-1]
for i in range(len(file_paths)):
    local_list = []
    length = lengths[i]
    while len(_index_list) > 0 and _index_list[-1] < cum_length + length:
        _ind = _index_list.pop()
        local_list.append(_ind - cum_length)

    cum_length += length
    master_list.append(torch.tensor(local_list))
    
assert sum([len(hi) for hi in master_list]) == sorted_unique_indices.shape[0]

results_file = os.path.join(
               args.tmp_directory, args.dataset,
               'retrieval_results_condensed.list'
           )

torch.save((
    labels.cpu(),                 # this is the labels
    sorted_unique_indices.cpu(),  # this is the indices into parquets
    master_list                   # This is the local indices (into each sub-parquet file)
), results_file)

print('Done. saved tuple of (labels, sorted_unique_indices, master_list) length {} in file: {}'.format(
    len(labels), results_file
))
