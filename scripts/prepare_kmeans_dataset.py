import os, sys
sys.path.append(os.getcwd())

import os
from source.utils import replace_underscores, get_text_labels, get_dataset_classnames
from source.trainer import prompt_strings
from source.models import MyClip
from source.samplers import get_laion_dataset
from source.transforms import get_test_transform
from source.utils import get_features

import torch, faiss, open_clip
from PIL import Image
import urllib.request
import torch.nn.functional as F
from tqdm import tqdm
import sys
from argparse_parameters import get_arg_parser
import argparse
import pandas as pd
from copy import deepcopy

parser = get_arg_parser()
args = parser.parse_args()
NUMBER_OF_PROCESSES = args.nproc # also number of waffle augmentations
N_WAFFLE = args.n_waffle
NEAREST_NEIGHBORS = args.num_nearest_neighbors
pwd = args.index_directory
DATASET = args.dataset
cache_dir=args.cache_dir

results_file = os.path.join(
               args.tmp_directory, args.dataset,
               'retrieval_results_condensed.list'
           )
labels, sorted_unique_indices, _ = torch.load(results_file)

def _get_num_shards():
    ''' number of .tar files in $TMP_DIR/$1/laion.'''
    tardir = os.path.join(
        args.tmp_directory, 
        args.dataset,
        'laion')
    ntar = 0
    for fn in os.listdir(tardir):
        if fn[-4:] == '.tar':
            ntar += 1
    print('num_shards: ', ntar)
    return ntar

dataset_name = '{}_{}nn_x_{}waffle'.format(args.dataset, NEAREST_NEIGHBORS, N_WAFFLE)
num_shards = _get_num_shards()
classnames = get_dataset_classnames(args.dataset, args)
n_classes = len(classnames)

out_parquet_file = os.path.join(
    args.tmp_directory, 
    args.dataset,
    'nn_sub_combined.parquet'
)
unique_df = pd.read_parquet(out_parquet_file) # original parquet

# downloaded parquet
parquet_list = []
def _mangle(i):
    si = str(i)
    return '0' * (5 - len(si)) + si
for i in range(num_shards): ###
    sub_parquet_file = os.path.join(
        args.tmp_directory, 
        args.dataset,
        'laion',
        '{}.parquet'.format(_mangle(i))
    )
    parquet_list.append(sub_parquet_file)
    
download_df = pd.concat([pd.read_parquet(fn) for fn in parquet_list])

assert len(download_df) == len(unique_df)
for i in tqdm(range(0, len(download_df), 100)):
    assert unique_df.iloc[int(download_df.iloc[i].key)].url == download_df.iloc[i].url

###################################################################################################


datadir = os.path.join(
        args.tmp_directory, 
        args.dataset,
        'laion_jpegs')
filelist = os.listdir(datadir)
counter = 0
result_tup_list = []
key_list = list(download_df.key)
status_list = list(download_df.status)
length = len(download_df)

# some downloads are unsuccessful. Only save the successful downloads
for i in tqdm(range(length)):
    if status_list[i] == 'success':
        str_key = key_list[i]
        jpg_name = str_key + '.jpg'

        label = labels[int(key_list[i])]
        counter += 1
        result_tup_list.append(( os.path.join(datadir,jpg_name), label.item()))
print(counter)

savename = os.path.join(
    args.tmp_directory, 
    args.dataset,
    'result_tup_list.{}'.format(dataset_name)
)
print('saving full retrieved dataset in {}, length {} '.format(savename, len(result_tup_list)))
torch.save(result_tup_list, savename)

###################################################################################################

from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

def _kmeans(y_truth, _label, jpeg_paths, k=16):
    return_list = []
    assert len(y_truth) == len(jpeg_paths)
    label_indices = (y_truth == _label).nonzero().view(-1)
    if len(label_indices) > k:
        ff = image_features[label_indices]

        kmeans = KMeans(n_clusters=k)
        kmeans.fit(ff)
        kmeans_labels = torch.tensor(kmeans.labels_)
        _inds = []
        for c in range(k):
            chosen_inds = (kmeans_labels == c).nonzero().view(-1)
            if len(chosen_inds) > 0:

                # randomly choose a sample from cluster c 
                representative_index = chosen_inds[torch.randperm(len(chosen_inds))[0]]

                assert not representative_index in _inds
                _inds.append(representative_index)

#         assert len(_inds) == k
        for i in range(len(_inds)):
            jpeg = jpeg_paths[label_indices[_inds[i]]]
            return_list.append((jpeg, _label))
    else:
        for iii in label_indices:
            jpeg = jpeg_paths[iii]
            return_list.append((jpeg, _label))
        
    return return_list

### indexing model. Fixed.
modelname='ViT-L-14'
pretrained = 'openai'
d = 768
file_paths, lengths, index_list = torch.load(pwd + 'parquet_lengths.list')

tokenizer = open_clip.get_tokenizer(modelname)
prompt = prompt_strings[DATASET]
classnames = get_dataset_classnames(DATASET, args)
n_classes = len(classnames)
print('CLASSNAMES:')
print(classnames)
print()
text_base = tokenizer(get_text_labels(
    classnames, prompt))

# indexing model
args.visual_prompt_depth = 0
args.visual_prompt_length = 0
args.text_prompt_length = 0
args.text_prompt_depth = 0
args.train_text_encoder = 0
args.train_visual_encoder = 0
model =  MyClip(modelname, pretrained, n_classes, d, 
                temp = 60, args=args, 
                tokenizer=tokenizer,
                tokenized_text_prototypes=None,
                cache_dir=cache_dir)
model = model.cuda()

with torch.no_grad(), torch.cuda.amp.autocast(enabled=True):
    text_base = tokenizer(get_text_labels(classnames, prompt))
    f_text = F.normalize(model.encode_text(text_base.cuda()))

test_xform = get_test_transform()
laion_dset = get_laion_dataset(test_xform, savename)

dl_laion = torch.utils.data.DataLoader(
                laion_dset,
                num_workers=8,
                batch_size=64,
                pin_memory=True,
                shuffle=False
            )


tmp_tup_filename = 'image_features_y_truth.tup'

# results_file = os.path.join(
#                args.tmp_directory, args.dataset,
#                'retrieval_results_condensed.list'
#            )

if tmp_tup_filename in os.listdir(os.path.join(args.tmp_directory, args.dataset)):
    image_features, y_truth = torch.load(os.path.join(args.tmp_directory, args.dataset, tmp_tup_filename))
else:
    with torch.no_grad():
        image_features, y_truth = get_features(
            dl_laion, model, d=768) 
    torch.save((image_features, y_truth), os.path.join(args.tmp_directory, args.dataset, tmp_tup_filename))
    
sims = (F.normalize(image_features.cuda()) * f_text[y_truth]).sum(1)

# pseudolabeling
pseduolabels = (F.normalize(image_features.cuda()) @ f_text.T).argmax(1).cpu()
new_tup_file = savename + '.pseduolabeled'
new_imgs = []
for i, (jpeg, label) in enumerate(laion_dset.imgs):
    new_imgs.append((jpeg, pseduolabels[i].item()))
torch.save(new_imgs, new_tup_file)
print('saved pseudo labeled dataset file in:', new_tup_file)

# filtering unsupervised dataset
threshold = args.filter_sim_threshold
clean_indices = (sims > threshold).nonzero().view(-1).cpu()
print('Number of samples BEFORE cleaning: {}'.format(len(y_truth)))
print('Counts per class, sorted:')
print(y_truth.unique(return_counts=True)[1].sort().values)

# only keep clean indices
y_truth = y_truth[clean_indices]
pseduolabels = pseduolabels[clean_indices]
image_features = image_features[clean_indices, :]
print('Number of samples AFTER cleaning: {}'.format(len(y_truth)))
print('Counts per class, sorted:')
print(y_truth.unique(return_counts=True)[1].sort().values)

agreement = (pseduolabels.cpu() == y_truth.cpu()).float().mean().item() * 100.
print('agreement between PL and orginal labels:', agreement)

laion_imgs = []
for i in clean_indices:
    laion_imgs.append(laion_dset.imgs[i])
    
jpeg_paths = [ii[0] for ii in laion_imgs]

###################################################################################

final_imgs = []
for ci in tqdm(range(n_classes)):
    final_imgs.extend(_kmeans(y_truth, ci, jpeg_paths, k=args.n_clusters))

savename = os.path.join(
    args.tmp_directory, 
    args.dataset,
    'result_tup_list.{}.clustered'.format(dataset_name)
)
print('saving kmeans cluster filtered dataset in {}, length {}'.format(savename, len(final_imgs)))
torch.save(final_imgs, savename)

###################################################################################

# final_imgs = []
# for ci in tqdm(range(n_classes)):
#     final_imgs.extend(_kmeans(pseduolabels, ci, jpeg_paths, k=args.n_clusters))

# savename = os.path.join(
#     args.tmp_directory, 
#     args.dataset,
#     'result_tup_list.{}.clustered.pseduolabeled'.format(dataset_name)
# )
# print('saving kmeans cluster filtered dataset in {}, length {}'.format(savename, len(final_imgs)))
# torch.save(final_imgs, savename)
