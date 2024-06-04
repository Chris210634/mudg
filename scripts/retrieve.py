import os, sys
sys.path.append(os.getcwd())

import torch, faiss, open_clip
from PIL import Image
import urllib.request
import torch.nn.functional as F
from tqdm import tqdm
from argparse_parameters import get_arg_parser
import os

parser = get_arg_parser()
args = parser.parse_args()
NUMBER_OF_PROCESSES = args.nproc # also number of waffle augmentations
NUM_NEAREST_NEIGHBORS = args.num_nearest_neighbors
pwd = args.index_directory

def _get_start_end(proc_id, ntasks, nproc):
    assert proc_id < ntasks
    assert nproc <= ntasks
    start = (ntasks // nproc) * proc_id
    end = (ntasks // nproc) * (proc_id + 1)
    if proc_id == nproc-1:
        end = ntasks
    return start, end

def _get(distances, indices, n_nn):
    '''
    There are multiple index files that haven't been combined.
    Because of this, the way we're retrieving is somewhat hacky.
    
    n_nn: number of nearest neighbors
    There are 55 index files.
    For each search, we need to retrieve the top n_nn neighbors 
        from each of the 55 index files.
    We then need to sort and take the top n_nn neighbors.
    '''
    assert len(distances.shape) == 2
    c,w = distances.shape # c is number of classes, w is the retrieval results to be sorted
    sim_matrix = torch.zeros((c, n_nn)).float()
    index_matrix = torch.zeros((c, n_nn)).long()
    for i in range(c):
        vals, inds = distances[i].sort(descending=True)
        index_matrix[i] = indices[i][inds[:n_nn]]
        sim_matrix[i] = vals[:n_nn]
        # vals is cos similarity 
        # inds is index into parquet table
    return sim_matrix, index_matrix

file_paths, lengths, index_list = torch.load(pwd + args.parquet_lengths_filename)
###   file_paths: 
#     ['0000.parquet',
#      '0001.parquet',
#      ...
#      '2313.parquet']

###   lengths: (length of each parquet)
#     [938763,
#      931244, ...]

###   index_list:
#     ['laion-2B-en/knn.index00',
#      'laion-2B-en/knn.index01',
#      ...
#      'laion-2B-en/knn.index54']
 
retrieval_results = []

# torch.Size([n_classes*n_waffle, feature_dim])
f = torch.load(
    os.path.join(args.tmp_directory, args.dataset, 'waffle.fs')
)
n_classes = f[0].shape[0]
start, end = _get_start_end(args.proc_id, len(f), NUMBER_OF_PROCESSES)
f = torch.cat(f[start:end]).cpu().numpy() 
# torch.Size([n_classes*n_tasks, feature_dim])

print('Shape of search feature: ', f.shape)
faiss.omp_set_num_threads(8)
torch.set_num_threads(8)

for index_fn in tqdm(index_list):
    index = faiss.read_index(pwd + index_fn)
    
    faiss.extract_index_ivf(index).nprobe = args.nprobe
    assert faiss.extract_index_ivf(index).nprobe == args.nprobe
    
#     res = faiss.StandardGpuResources()
#     co = faiss.GpuClonerOptions()
#     co.useFloat16 = True
#     index = faiss.index_cpu_to_gpu(res, 0, index, co)

    distances, indices = index.search(f, NUM_NEAREST_NEIGHBORS)
    del index
    retrieval_results.append((distances, indices))
    
distances = torch.cat([torch.tensor(lli[0]) for lli in retrieval_results], dim=1)
indices = torch.cat([torch.tensor(lli[1]) for lli in retrieval_results], dim=1)

print('raw distances.shape:', distances.shape)
print('raw indices.shape:',   indices.shape)
print('---------------------------------')

sim_matrix, index_matrix = _get(distances, indices, NUM_NEAREST_NEIGHBORS)

# we need to reshape such that the first dimension is the number of classes
assert sim_matrix.shape[0] == index_matrix.shape[0]
assert (end-start)*n_classes == sim_matrix.shape[0]
assert NUM_NEAREST_NEIGHBORS == sim_matrix.shape[1]
new_sim_matrix = torch.zeros((n_classes, (end-start)*NUM_NEAREST_NEIGHBORS)).float()
new_index_matrix = torch.zeros((n_classes, (end-start)*NUM_NEAREST_NEIGHBORS)).long()
for i in range(end-start):
    new_sim_matrix[:,i*NUM_NEAREST_NEIGHBORS:(i+1)*NUM_NEAREST_NEIGHBORS] = sim_matrix[i*n_classes:(i+1)*n_classes,:]
    new_index_matrix[:,i*NUM_NEAREST_NEIGHBORS:(i+1)*NUM_NEAREST_NEIGHBORS] = index_matrix[i*n_classes:(i+1)*n_classes,:]
sim_matrix = new_sim_matrix
index_matrix = new_index_matrix

print('sorted sim_matrix.shape:', sim_matrix.shape)
print('sorted index_matrix.shape:', index_matrix.shape)

save_retrieval_results_file = os.path.join(
               args.tmp_directory, args.dataset,
               'retrieval_results_{}.list'.format(
                   int(args.proc_id)
               )
           )
torch.save((sim_matrix, index_matrix), 
           save_retrieval_results_file
          )
print('Done. saved tuple of (sim_matrix, index_matrix) shapes {} {} in file: {}'.format(
    sim_matrix.shape, index_matrix.shape, save_retrieval_results_file
))