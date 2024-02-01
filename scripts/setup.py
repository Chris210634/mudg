import os, sys
sys.path.append(os.getcwd())

import open_clip
import torch
from source.utils import get_text_labels, get_dataset_classnames
from source.trainer import prompt_strings
from source.models import MyClip
from argparse_parameters import get_arg_parser
import argparse
from tqdm import tqdm
import torch.nn.functional as F
import os

parser = get_arg_parser()
### for now we're going to not modify the indexing model
args = parser.parse_args()
args.visual_prompt_depth = 0
args.visual_prompt_length = 0
args.text_prompt_length = 0
args.text_prompt_depth = 0
args.train_text_encoder = 0
args.train_visual_encoder = 0

### Retrieval parameters
NUMBER_OF_PROCESSES = args.nproc
N_WAFFLE = args.n_waffle
NUM_NEAREST_NEIGHBORS = args.num_nearest_neighbors
PWD = args.index_directory
DATASET = args.dataset
cache_dir=args.cache_dir

### indexing model. Fixed.
modelname='ViT-L-14'
pretrained = 'openai'
d = 768

print(args)

# indexing model (OpenAI ViT-L-14)
tokenizer = open_clip.get_tokenizer(modelname)
prompt = prompt_strings[DATASET]
classnames = get_dataset_classnames(DATASET, args)
n_classes = len(classnames)
print('CLASSNAMES:')
print(classnames)
print()

# indexing model
model =  MyClip(modelname, pretrained, n_classes, d, 
                temp = 60, args=args, 
                tokenizer=tokenizer,
                tokenized_text_prototypes=None,
                cache_dir=cache_dir)

# not using yet
if len(args.checkpoint) > 0:
    print('loading checkpoint')
    sd = torch.load(args.checkpoint)
    model.load_state_dict(sd)
model = model.cuda()

# get the Waffle augmentations of text.
if len(args.descriptor_file) > 0:
    gg = torch.load(args.descriptor_file)[:N_WAFFLE]
    for ggi in gg:
        print(ggi)
else:
    gg = ['.']
    # gg = [", which has nKes7, '5yMa.",
    #       ", which has patio lamps.",
    #       ", which has kwfmp, z3Ax0.",
    #       ", which has pxSs2, ie4Hq.",
    #       ", which has EGIBP, dQObI.",
    #       ", which has B-qwo, r/UJA.",
    #       ", which has 5jNPC, lWsrt.",
    #       ''', which has GscIG, S"yJU.''']

_prompts = get_text_labels(classnames, prompt)
print('Unaugmented prompts: ')
for _p in _prompts:
    print(_p)
print()

# calculate augmented text features
fs = []
with torch.no_grad(), torch.cuda.amp.autocast(enabled=True):
    for gi in gg:
        new_prompts = [pi[:-1] + gi for pi in _prompts]
        print('Example new augmented prompt: ', new_prompts[0])
        tokenized_text = tokenizer(new_prompts).cuda()
        f = F.normalize(model.encode_text(tokenized_text)) ### Caution: will not work if shallow prompt has been trained
        fs.append(f.cpu())
        
# save the augmented text features for retrieval
if not os.path.exists(os.path.join(args.tmp_directory, args.dataset)):
    os.makedirs(os.path.join(args.tmp_directory, args.dataset))
torch.save(fs, os.path.join(args.tmp_directory, args.dataset, 'waffle.fs'))