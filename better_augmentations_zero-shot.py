import sys
import os
sys.path.append(os.path.abspath(os.getcwd()))

import argparse
import os
import random
from tqdm import tqdm
import torch
import torch.nn.functional as F
import open_clip
from copy import deepcopy

import source.gpt_helpers as gpt_helpers

from source.utils import *
from source.losses import *
from source.samplers import *
from source.transforms import *
from source.models import *
from source.trainer import *
from argparse_parameters import get_arg_parser
import time

parser = get_arg_parser()
args = parser.parse_args()
print(args)

dataset = args.dataset

descriptors = []
word_descriptors = None
good_descriptors = None

if args.descriptor_file != '':
    descriptors = torch.load(args.descriptor_file)
    if args.n_descriptors > 0:
        if args.shuffle_descriptors:
            random.shuffle(descriptors)
        descriptors = descriptors[:args.n_descriptors]
        
else:
    # use default descriptor files
    default_word_descriptor_file = 'cache/word_soup_descriptors_seed{}__{}_{}.list'.format(args.seed, args.modelname, args.pretrained)
    default_desc_descriptor_file = 'cache/good_descriptions_seed{}__{}_{}.list'.format(args.seed, args.modelname, args.pretrained)
    
    word_descriptors = torch.load(default_word_descriptor_file)
    good_descriptors = torch.load(default_desc_descriptor_file)
    
############################ SETUP ############################
base_cfg = argparse.Namespace()
base_cfg.ROOT = args.data_dir
base_cfg.NUM_SHOTS = 16
base_cfg.SEED = args.seed
base_cfg.SUBSAMPLE_CLASSES = 'all'
device = "cuda"
bs = args.bs
modelname = args.modelname
pretrained = args.pretrained
cache_dir = args.cache_dir
d = args.d
epochs = args.n_epochs
iters_per_epoch = args.iters_per_epoch
args.dataset = dataset
dataset = args.dataset

prompt = prompt_strings[args.dataset]
dataset_class = dataset_classes[args.dataset]
####################################################################

############################ TRANSFORMS ############################
train_xform = get_train_transform()
test_xform = get_test_transform()
base_dset = dataset_class(base_cfg)

# unlike the other datasets, when using immagenet,
# following standard procedure, we use the 50,000 validation images
# for testing
dset_base_val = []
dset_base_test, dset_new_test = get_imagenet_val_dataset(
    test_xform,
    imagenet_root = os.path.join(args.data_dir, 'imagenet'))
# for i in range(500):
#     assert base_dset.classnames[i] == get_imagenet_classnames()[i]
    
print('number of base classes: ', len(base_dset.classnames))
####################################################################
     
############################ MODEL #################################
n_classes = len(base_dset.classnames)
        
### Get the zeroshot text prototypes (with prompt)
tokenizer = open_clip.get_tokenizer(modelname)
text_base = tokenizer(get_text_labels(base_dset.classnames, prompt))

model =  MyClip(modelname, pretrained, n_classes, d, 
                  temp = args.temp, args=args, 
                tokenizer=tokenizer,
                tokenized_text_prototypes=text_base,
                cache_dir=cache_dir, descriptors=descriptors)
model = model.cuda()

init_sd = deepcopy(model.cpu().state_dict())
model = model.cuda()

# if maple is true or deep visual and text prompts,
# need to not include that in the initial frozen model
if args.maple or args.visual_prompt_depth or args.text_prompt_depth > 1:
    init_args = deepcopy(args)
    init_args.maple = 0
    init_args.visual_prompt_depth = 0
    init_args.text_prompt_depth = 1
    init_frozen_model =  MyClip(modelname, pretrained, n_classes, d, 
                  temp = args.temp, args=init_args, 
                tokenizer=tokenizer,
                tokenized_text_prototypes=text_base,
                cache_dir=cache_dir, descriptors=descriptors)
    init_frozen_model = init_frozen_model.cuda()
else:
    init_frozen_model = deepcopy(model)
init_frozen_model.eval()
####################################################################

if args.checkpoint != '':
    sd = torch.load(args.checkpoint)
    if 'shallow_prompt.desc_vectors' in sd:
        del sd['shallow_prompt.desc_vectors']
    model.load_state_dict(sd)
    
if args.suffix_string != '':
    model.shallow_prompt.swap_suffix(args.suffix_string, model)

############################ OPTIMIZERS ############################
if not args.eval_only:
    params = get_params(args, model, descriptor_strings=descriptors)
    optimizer, scheduler, scaler = get_optimizer(args, params)
ema_model = deepcopy(model)
####################################################################

lr = args.lr
loss_list = []
accs = []

tokenized_text_with_descriptors = None
if args.train_with_descriptors:
    class_label_strings = get_text_labels(base_dset.classnames, prompt)
    n_classes = len(class_label_strings)
    for cls in range(n_classes):
        assert class_label_strings[cls][-1] == '.'
    tokenized_text_with_descriptors = []
    __descriptors_list = word_descriptors if not word_descriptors is None else descriptors
    for desc in __descriptors_list:
        tokenized_dec = tokenizer([ci[:-1] + ',' + desc for ci in class_label_strings])
        tokenized_text_with_descriptors.append(tokenized_dec)
    tokenized_text_with_descriptors = torch.stack(tokenized_text_with_descriptors).cuda()
    print('tokenized_text_with_descriptors.shape: ',
          tokenized_text_with_descriptors.shape)
    
cfg = argparse.Namespace()
cfg.ROOT = args.data_dir
cfg.NUM_SHOTS = 16
cfg.SEED = 1
cfg.SUBSAMPLE_CLASSES = 'all'
dataset_class = dataset_classes[dataset]
dset = dataset_class(cfg)
test_xform = get_test_transform()

if dataset == 'validation':
    dset_test = dset_base_val
elif dataset in domainnet_datasets or dataset in officehome_datasets or \
    dataset in VLCS_datasets or dataset in PACS_datasets or dataset in TerraInc_datasets:
    dset_test = dset
    dset_test.transform = test_xform
elif not dataset in ['ImageNet']:
    dset_test = dassl_dataset_conversion(dset, test_xform, 'test')
else:
    # unlike the other datasets, when using immagenet,
    # following standard procedure, we use the 50,000 validation images
    # for testing
    dset_test = get_imagenet_val_dataset(
        test_xform,
        imagenet_root = os.path.join(args.data_dir, 'imagenet'),
        split=False)

dl_test = torch.utils.data.DataLoader(
                    dset_test,
                    num_workers=8,
                    batch_size=32,
                    pin_memory=True,
                    drop_last=False,
                    shuffle=False
                )

prompt = prompt_strings[dataset]
tokenizer = open_clip.get_tokenizer(args.modelname)
text = tokenizer(get_text_labels(dset.classnames, prompt))

fn = 'cache/image_features.y_truth.{}{}{}.tup'.format(dataset, modelname, pretrained)
if not os.path.exists(fn):
    model.eval()
    with torch.no_grad():
        image_features, y_truth = get_features(dl_test, model, d=args.d)
    torch.save((image_features, y_truth), fn)
else:
    image_features, y_truth = torch.load(fn)
    
def _evaluate(image_features, text_features, y_truth):
    with torch.no_grad():
        assert image_features.shape[0] == y_truth.shape[0]
        y_truth = y_truth.cuda()
        image_features = F.normalize(image_features.float().cuda())
        probs = image_features @ F.normalize(text_features).T
        y_hat = probs.max(1).indices
    acc = (y_hat == y_truth).float().mean().item() * 100.
    print('acc: ', acc)
    return acc

model.reset_text(text)
with torch.no_grad(), torch.cuda.amp.autocast(enabled=True):
    zs_text_features = model.get_text_prototypes()
_evaluate(image_features, zs_text_features, y_truth)

def _get_descriptor_feature_matrix( model_copy,descriptors,text,token_offsets):
    model_copy.shallow_prompt.reset_descriptors(model_copy, descriptors)
    model_copy.reset_text(text)
    text_features_list = []
    for token_offset in token_offsets:
        for descriptor_index in range(len(model_copy.shallow_prompt.desc_vectors)):
            with torch.no_grad(), torch.cuda.amp.autocast(enabled=True):
                text_features_sub = model_copy.get_text_prototypes(
                    descriptor_index=descriptor_index,
                    token_offset=token_offset
                )
            text_features_list.append(F.normalize(text_features_sub.float()))
    return torch.stack(text_features_list)

def _evaluate_descriptors_scoremean(descriptors, ms = [2, 4, 8,16,32]):
    text_features_stack = _get_descriptor_feature_matrix(
            model, descriptors[:ms[-1]], text,
            token_offsets=[0]
#             token_offsets=[0, 5, 10, 15, 20, 25] ###
        )
    scores = torch.zeros(image_features.shape[0], len(dset.classnames)).cuda()
    print('text_features_stack.shape:', text_features_stack.shape)
    accs = []
    for m in ms:
        for i, c in enumerate(dset.classnames):
            v = F.normalize(text_features_stack[:m, i, :])
#             v = F.normalize(text_features_stack[:, i, :])
            scores[:, i] = (image_features.float().cuda() @ v.T).mean(dim=1)
        _acc = (scores.max(1).indices == y_truth.cuda()).float().mean().item() * 100.
        accs.append(_acc)
    return accs

# def _get_ensemble_acc_list(descriptors):
#     # waffle # descriptors
#     ms = [2, 4, 8,16,32]
#     acc_list = []
# #     descriptors = [',' + k for k in torch.load('cache/descriptions.list')]
#     for _ in range(3):
#         random.shuffle(descriptors)
#         _accs = _evaluate_descriptors_scoremean(descriptors, ms=ms)
#         acc_list.append(_accs)
#     return torch.tensor(acc_list).mean(0).tolist()

# rand_desc_accs = _get_ensemble_acc_list([',' + k for k in torch.load('cache/descriptions.list')])
# waffle_accs = _get_ensemble_acc_list(torch.load('cache/waffle_descriptors_512_count.list'))

fnn = 'cache/{}{}_{}_{}_{}.tensor'.format(
            dataset,
            'description_features',
            '', 
            modelname,
            pretrained
        )

if not os.path.exists(fnn):

    descriptions = torch.load('cache/descriptions.list')
    description_features = torch.zeros(
        len(descriptions), 
        len(dset.classnames), 
        args.d
    )

    prompted_strings = get_text_labels(dset.classnames, prompt)

    for i, desc in tqdm(enumerate(descriptions)):
        dl = [c[:-1] + ',' + desc for c in prompted_strings]

        # print one out for sanity check
        if i == 0:
            print('Example: ', dl[0])

        model.reset_text(tokenizer(dl))
        with torch.no_grad(), torch.cuda.amp.autocast(enabled=True):
            f_text = F.normalize(
                model.get_text_prototypes()
            )
        description_features[i, :, :] = f_text

    torch.save(
        (descriptions,description_features), 
        fnn
    )
    print('saved description features in: {}'.format(fnn))
    
else:
    descriptions,description_features = torch.load(fnn)

assert len(description_features) == len(descriptions)
print('description_features.shape: ', description_features.shape)

import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

def kmeans(embeddings, k=200, device='cuda'):
    '''
    K-means ++ https://en.wikipedia.org/wiki/K-means%2B%2B and 
    https://en.wikipedia.org/wiki/K-means_clustering
    Assume normalized embeddings.
    '''
    cluster_centers = torch.zeros(k, embeddings.shape[1], device=device)
    initial = np.random.choice(embeddings.shape[0]) #Choose one center uniformly at random among the data points.
    cluster_centers[0, :] = embeddings[initial,:]

    for ki in range(1, k):
        # for each data point x not chosen yet, compute D(x), 
        # the distance between x and the nearest center that has already been chosen.
        D_x = (2. - 2. * (embeddings @ cluster_centers.T))[:,:ki] # N x k euclidean distance squared
        D_x, _ =  D_x.min(1)

        # Choose one new data point at random as a new center, 
        # using a weighted probability distribution where a point x is chosen 
        # with probability proportional to D(x)^2.
        D_x = D_x.clamp(min=0.)
        chosen_index = torch.distributions.categorical.Categorical(D_x / D_x.sum()).sample().item()
        cluster_centers[ki, :] = embeddings[chosen_index,:]

    track_list = []
    nearest_clusterid = torch.zeros(embeddings.shape[0], device=device)
    for it in range(100):
        # Assign each observation to the cluster with the nearest mean
        D_x = (2. - 2. * (embeddings @ cluster_centers.T)) # N x k euclidean distance squared
        _, nearest_clusterid_next =  D_x.min(1)

        if (nearest_clusterid != nearest_clusterid_next).sum() == 0:
            # The algorithm has converged when the assignments no longer change. 
            print('K-means converged at iteration {}, exiting ...'.format(it))
            break

        nearest_clusterid = nearest_clusterid_next

        # Recalculate means (centroids) for observations assigned to each cluster.
        mask = nearest_clusterid == torch.arange(k, device=device).unsqueeze(0).T # torch.Size([200, 8054]) bool
        # embeddings.shape is torch.Size([8054, 1024])

        # https://stackoverflow.com/questions/69314108/how-to-do-a-masked-mean-in-pytorch
        denom = torch.sum(mask, -1, keepdim=True)
        cluster_centers_next = torch.sum(embeddings * mask.unsqueeze(-1), dim=1) / denom
        cluster_centers_next = F.normalize(cluster_centers_next)
        track_list.append((cluster_centers_next - cluster_centers).sum().item())
        cluster_centers = cluster_centers_next
    return cluster_centers, nearest_clusterid

def _get_acc(image_features, f_text, y_truth):
    scores = image_features @ f_text.cuda().T
    y_hat = scores.max(1).indices
    acc = (y_hat == y_truth).float().mean().item() * 100.
    return acc

# accs = []
image_features = image_features.cuda()
# description_features = description_features.cuda()
y_truth = y_truth.cuda()

# for i in tqdm(range(len(descriptions))):
#     acc = _get_acc(image_features, description_features[i].cuda(), y_truth)
#     accs.append(acc)

def _get_description_features_without_label_names(model, descriptions):
    t = tokenizer(['a photo' + de for de in descriptions])
    f_t = torch.zeros(len(t), args.d)
    bs = 128
    ptr = 0
    t = t.cuda()
    with torch.no_grad():
        while ptr < len(f_t):
            begin = ptr
            end = min(ptr+bs, len(f_t))
            _f = model.encode_text(t[begin:end])
            f_t[begin:end, :] = _f.cpu()
            ptr = end
    assert end == len(f_t)
    return F.normalize(f_t)

def _get_indices(zs_text_features):
    
    k = 16
    cluster_centers, nearest_clusterid = kmeans(zs_text_features.cuda(), k=k, device='cuda')
    
    varss = []
    vars_master = []
    var_bases = []
    for ki in range(k):
        if (nearest_clusterid == ki).sum() < 2:
            print('not enough points in cluster #{}, continuing ...'.format(ki))
            continue

        varss = []
        for i in range(len(descriptions)):
            d = description_features[i].cuda()[nearest_clusterid == ki]
            var = (d @ d.T).mean().item()
            varss.append(var)

        d = zs_text_features[nearest_clusterid == ki]
        var_base = (d @ d.T).mean().item()
        var_bases.append(var_base)
        vars_master.append(varss)

    vars_master = torch.tensor(vars_master)
    var_bases = torch.tensor(var_bases).unsqueeze(0).T

    indices = (vars_master < var_bases).float().sum(0).sort(descending=True).indices
    return indices

ms = [2, 4, 8, 16]
acc_list = []
for _ in range(3):
    chosen_descriptors = [descriptions[i] for i in _get_indices(zs_text_features)] ### ### ###
    chosen_descriptors = [',' + k for k in chosen_descriptors]
    print(chosen_descriptors[:16])
    _accs = _evaluate_descriptors_scoremean(chosen_descriptors, ms=ms)
    acc_list.append(_accs)
our_accs = torch.tensor(acc_list).mean(0).tolist()

# import matplotlib.pyplot as plt

# plt.figure(figsize=(8, 6))

# ms = [2, 4, 8,16,32]
# zs_acc = _evaluate(image_features, zs_text_features, y_truth)
# p = plt.plot([1] + ms, 
#              [zs_acc] + waffle_accs, 
#              marker='o', label='waffle')
# p = plt.plot([1] + ms, 
#              [zs_acc] + rand_desc_accs, 
#              marker='o', label='rand desc soup')
# p = plt.plot([1] + [2, 4, 8, 16], 
#              [zs_acc] + our_accs, 
#              marker='*', label='ours')

# plt.grid()
# plt.legend()
# # plt.ylim(73.5, 76)
# plt.ylabel('{} ZS {}'.format(dataset, args.modelname))
# plt.xlabel('ensemble size')
# plt.title('Score Mean')

# plt.savefig('figures/{}_{}.png'.format(dataset, args.modelname))


# print('waffle_accs:', waffle_accs)
# print('rand_desc_accs:', rand_desc_accs)
print('our_accs:', our_accs)

fn = 'cache/better_descriptors_sorted_{}_{}.list'.format(modelname, dataset)

torch.save(chosen_descriptors, fn)