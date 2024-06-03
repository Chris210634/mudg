import random
from tqdm import tqdm
import torch
import torch.nn.functional as F
import open_clip
from copy import deepcopy

from source.datasets import oxford_pets 
from source.datasets import oxford_flowers
from source.datasets import fgvc_aircraft
from source.datasets import dtd
from source.datasets import eurosat
from source.datasets import stanford_cars
from source.datasets import food101
from source.datasets import sun397
from source.datasets import caltech101
from source.datasets import ucf101
from source.datasets import imagenet
from source.datasets import imagenetv2
from source.datasets import imagenet_sketch
from source.datasets import imagenet_a
from source.datasets import imagenet_r
from source.datasets import domainbed

from pytorch_metric_learning import miners, losses

from source.utils import *
from source.losses import *
from source.samplers import *
from source.transforms import *
from source.models import *

import pytorch_metric_learning

number_of_classes = {
    'ImageNet':1000,
    'Caltech101':100,
    'OxfordPets':37,
    'StanfordCars':196,
    'Flowers102':102,
    'Food101':101,
    'FGVCAircraft':100,
    'SUN397':397,
    'DTD':47,
    'EuroSAT':10,
    'UCF101':101,
    'ImageNetV2':1000,
    'ImageNetSketch':1000,
    'ImageNetA':200,
    'ImageNetR':200,
    'DomainNet':345,
    'OfficeHome':65,
    'PACS':7,
    'VLCS':5,
    'TerraInc':10
}

prompt_strings = {
    'ImageNet':'a photo of a {}.',
    'Caltech101':'a photo of a {}.',
    'OxfordPets':'a photo of a {}, a type of pet.',
    'StanfordCars':'a photo of a {}.',
    'Flowers102':'a photo of a {}, a type of flower.',
    'Food101':'a photo of {}, a type of food.',
    'FGVCAircraft':'a photo of a {}, a type of aircraft.',
    'SUN397':'a photo of a {}.',
    'DTD':'a photo of a {} texture.',
    'EuroSAT':'a photo of {}, from a satellite.',
    'UCF101':'a photo of a person doing {}.',
    'ImageNetV2':'a photo of a {}.',
    'ImageNetSketch':'a photo of a {}.',
    'ImageNetA':'a photo of a {}.',
    'ImageNetR':'a photo of a {}.',
    'DomainNet':'a photo of a {}.',
    'DomainNet.clipart':  'a photo of a {}.',
    'DomainNet.infograph':'a photo of a {}.',
    'DomainNet.painting': 'a photo of a {}.',
    'DomainNet.quickdraw':'a photo of a {}.',
    'DomainNet.real':     'a photo of a {}.',
    'DomainNet.sketch':   'a photo of a {}.',
    'OfficeHome': 'a photo of a {}.',
    'OfficeHome.art': 'a photo of a {}.',
    'OfficeHome.clipart': 'a photo of a {}.',
    'OfficeHome.product': 'a photo of a {}.',
    'OfficeHome.real': 'a photo of a {}.',
    'PACS':         'a photo of a {}.',
    'PACS.art':     'a photo of a {}.',
    'PACS.cartoon': 'a photo of a {}.',
    'PACS.photo': 'a photo of a {}.',
    'PACS.sketch':    'a photo of a {}.',
    'VLCS':         'a photo of a {}.',
    'VLCS.caltech': 'a photo of a {}.',
    'VLCS.labelme': 'a photo of a {}.',
    'VLCS.sun':     'a photo of a {}.',
    'VLCS.voc':     'a photo of a {}.',
    'TerraInc':       'a photo of a {}, from a camera trap.',
    'TerraInc.100':   'a photo of a {}, from a camera trap.',
    'TerraInc.38':    'a photo of a {}, from a camera trap.',
    'TerraInc.43':    'a photo of a {}, from a camera trap.',
    'TerraInc.46':    'a photo of a {}, from a camera trap.'
}

test_only_datasets = ['ImageNetV2', 'ImageNetSketch', 'ImageNetA', 'ImageNetR']
domainnet_datasets = ['DomainNet.clipart', 'DomainNet.infograph', 'DomainNet.painting', 
                     'DomainNet.quickdraw', 'DomainNet.real', 'DomainNet.sketch']
officehome_datasets = ['OfficeHome.art', 'OfficeHome.clipart', 'OfficeHome.product', 'OfficeHome.real']
VLCS_datasets = ['VLCS.caltech', 'VLCS.labelme', 'VLCS.sun', 'VLCS.voc']
PACS_datasets = ['PACS.art', 'PACS.cartoon', 'PACS.photo', 'PACS.sketch']
TerraInc_datasets = ['TerraInc.100', 'TerraInc.38', 'TerraInc.43', 'TerraInc.46']

dataset_classes = {
    'ImageNet':imagenet.ImageNet,
    'Caltech101':caltech101.Caltech101,
    'OxfordPets':oxford_pets.OxfordPets,
    'StanfordCars':stanford_cars.StanfordCars,
    'Flowers102':oxford_flowers.OxfordFlowers,
    'Food101':food101.Food101,
    'FGVCAircraft':fgvc_aircraft.FGVCAircraft,
    'SUN397':sun397.SUN397,
    'DTD':dtd.DescribableTextures,
    'EuroSAT':eurosat.EuroSAT,
    'UCF101':ucf101.UCF101,
    'ImageNetV2':imagenetv2.ImageNetV2,
    'ImageNetSketch':imagenet_sketch.ImageNetSketch,
    'ImageNetA':imagenet_a.ImageNetA,
    'ImageNetR':imagenet_r.ImageNetR,
    'DomainNet':domainbed.DomainNetDataset,
    'DomainNet.clipart':  domainbed.DomainNetClipartDataset,
    'DomainNet.infograph':domainbed.DomainNetInfographDataset,
    'DomainNet.painting': domainbed.DomainNetPaintingDataset,
    'DomainNet.quickdraw':domainbed.DomainNetQuickdrawDataset,
    'DomainNet.real':     domainbed.DomainNetRealDataset,
    'DomainNet.sketch':   domainbed.DomainNetSketchDataset,
    'OfficeHome':         domainbed.OfficeHomeDataset,
    'OfficeHome.art':     domainbed.OfficeHomeArtDataset,
    'OfficeHome.clipart': domainbed.OfficeHomeClipartDataset,
    'OfficeHome.product': domainbed.OfficeHomeProductDataset,
    'OfficeHome.real':    domainbed.OfficeHomeRealDataset,
    'PACS':               domainbed.PACSDataset,
    'PACS.art':           domainbed.PACSArtDataset,
    'PACS.cartoon':       domainbed.PACSCartoonDataset,
    'PACS.photo':         domainbed.PACSPhotoDataset,
    'PACS.sketch':        domainbed.PACSSketchDataset,
    'VLCS':               domainbed.VLCSDataset,
    'VLCS.caltech':       domainbed.VLCSCaltechDataset,
    'VLCS.labelme':       domainbed.VLCSLabelmeDataset,
    'VLCS.sun':           domainbed.VLCSSunDataset,
    'VLCS.voc':           domainbed.VLCSVocDataset,
    'TerraInc':           domainbed.TerraIncDataset,
    'TerraInc.100':       domainbed.TerraInc100Dataset,
    'TerraInc.38':        domainbed.TerraInc38Dataset,
    'TerraInc.43':        domainbed.TerraInc43Dataset,
    'TerraInc.46':        domainbed.TerraInc46Dataset,
}

def get_captions_from_file(args, base_dset, test_xform, 
                           model, dl_base_train, tokenizer, get_features=True):
    ''''''
    
    # This is where we retrieve the captions from file and embed the captions.
    caption_texts = []
    img_paths = []
    with open(args.caption_file) as f:
        for line in f:
            if '<eol>' in line:
                ll = line.strip().split('<eol>')[0]
                img_path = ll.split('<sep>')[0]
                caption = ll.split('<sep>')[1]
                img_paths.append(img_path)
                caption_texts.append(caption)

    # check that images used for captions are the same as what using now
    assert len(caption_texts) == len(dl_base_train.dataset.imgs)
    for i in range(len(caption_texts)):
        # just check the image file name and not the rest of the path
        assert dl_base_train.dataset.imgs[i][0].split('/')[-1] == img_paths[i].split('/')[-1]

    # Precalculate caption features
    caption_texts = tokenizer(caption_texts)
    if not get_features:
        return caption_texts
    
    with torch.no_grad(), torch.cuda.amp.autocast():
        text_features = []
        ptr = 0
        while ptr < len(caption_texts):
            c = caption_texts[ptr:min(len(caption_texts),ptr+64)]
            f = model.encode_text(c.cuda()).float().cpu()
            text_features.append(f)
            ptr += f.shape[0]
    text_features = torch.cat(text_features)
    text_features = F.normalize(text_features)
    return text_features
        # dl_base_train.dataset.metadata = text_features
        
def get_captioner(args, caption_prompt_text='a photo of'):
    def _tokenize(x, tokenizer):
        x_tokenized = tokenizer(x).squeeze()
        start_token = 49406
        end_token = 49407
        assert x_tokenized[0] == start_token
        return x_tokenized[:list(x_tokenized).index(end_token)]

    tokenizer = open_clip.get_tokenizer(args.modelname)

    assert open_clip.tokenizer.tokenize == open_clip.get_tokenizer(args.modelname)
    assert open_clip.tokenizer.tokenize == open_clip.get_tokenizer("coca_ViT-L-14")

    caption_model, _, _ = open_clip.create_model_and_transforms(
      model_name="coca_ViT-L-14",
      pretrained="laion2B-s13B-b90k"
    )
    caption_prompt_tokenized=torch.ones(
        (args.bs, 1), 
        device='cuda', 
        dtype=torch.long
    )*_tokenize(caption_prompt_text, tokenizer).cuda()
    caption_model = caption_model.cuda()
    return caption_prompt_tokenized, caption_model

def get_optimizer(args, params):
    if args.optimizer == 'adam':
        optimizer = torch.optim.Adam(params, 
                                     lr=args.lr, weight_decay=args.wd)
    elif args.optimizer == 'sam':
        assert args.autocast == 0
        base_optimizer = torch.optim.SGD
        optimizer = SAM(params, base_optimizer, 
                        lr=args.lr, rho=args.rho, momentum=0.9)
    else:
        assert args.optimizer == 'sgd'
        optimizer = torch.optim.SGD(params, momentum=0.9,
                                     lr=args.lr, weight_decay=args.wd)
        
    scaler = torch.cuda.amp.GradScaler()
#     val_scaler = torch.cuda.amp.GradScaler()

    scheduler_lambda = lambda it: math.exp(
        -args.lr_decay * it/(args.iters_per_epoch*args.n_epochs)
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=scheduler_lambda)
    
    print(' optimizer param shapes ')
    print('------------------------')
    for iii, param_group in enumerate(optimizer.param_groups):
        for p in param_group['params']:
            print(p.shape)
    print('------------------------')

    return optimizer, scheduler, scaler

def get_soft_tokens(model, descriptor_strings):
    descriptor_vectors = []
    for suffix_string in descriptor_strings:
        print('suffix string: ', suffix_string)
        suffix_tokens = model.shallow_prompt.tokenizer([suffix_string]).view(-1)
        print('suffix tokens: ', suffix_tokens)
        suffix_eof = suffix_tokens.argmax(dim=-1)
        suffix_tokens = suffix_tokens[1:suffix_eof]
        suffix_vectors = model.token_embedding(
            suffix_tokens.to(model.text_projection.device)
        ).detach()
        descriptor_vectors.append(suffix_vectors)
    return torch.stack(descriptor_vectors)

def get_params(args, model, descriptor_strings=[]):
    params = []
    if args.train_visual_encoder:
        if args.lora or args.bitfit or args.ssf or args.resblock_adapter:
            for r in model.visual.transformer.resblocks[args.layer_start_v:]:
                for name, p in r.named_parameters():
                    if args.lora and 'lora' in name:
                        print('lora: adding {} to visual encoder optimizer list.'.format(name))
                        params = params + [p]
                    if args.bitfit and 'bias' in name:
                        print('bitfit: adding {} to visual encoder optimizer list.'.format(name))
                        params = params + [p]
                    if args.ssf and 'ssf_' in name:
                        print('SSF: adding {} to visual encoder optimizer list.'.format(name))
                        params = params + [p]
                    elif args.resblock_adapter and 'adapter' in name:
                        print('Adapter: adding {} to visual encoder optimizer list.'.format(name))
                        params = params + [p]
        else:
            params = params + list(model.visual.transformer.resblocks[args.layer_start_v:].parameters()) + [model.visual.proj] + list(model.visual.ln_post.parameters())

    if args.train_text_encoder:
        if args.lora or args.bitfit or args.ssf or args.resblock_adapter:
            for r in model.visual.transformer.resblocks[args.layer_start_v:]:
                for name, p in r.named_parameters():
                    if args.lora and 'lora' in name:
                        print('lora: adding {} to text encoder optimizer list.'.format(name))
                        params = params + [p]
                    if args.bitfit and 'bias' in name:
                        print('bitfit: adding {} to text encoder optimizer list.'.format(name))
                        params = params + [p]
                    if args.ssf and 'ssf_' in name:
                        print('SSF: adding {} to text encoder optimizer list.'.format(name))
                        params = params + [p]
                    elif args.resblock_adapter and 'adapter' in name:
                        print('Adapter: adding {} to text encoder optimizer list.'.format(name))
                        params = params + [p]
        else:
            text_tower_params = list(
                model.transformer.resblocks[args.layer_start_t:].parameters()
            ) + [model.text_projection] + list(model.ln_final.parameters())
            params += text_tower_params
        
    if args.visual_prompt_depth > 0 and not args.maple:
        params += [model.shallow_visual_prompt]
        
    if args.visual_prompt_depth > 1 and not args.maple:
        params += [model.visual.transformer.prompts]
        
    if args.maple:
        assert args.visual_prompt_depth > 0
        params += list(model.maple_projector.parameters())
        
    if args.text_prompt_depth > 1:
        params += [model.transformer.prompts]
        
    if args.adapter:
        assert not model.adapter is None
        params += list(model.adapter.parameters())
        
    if args.text_prompt_depth > 0:
        prompt_params = list(model.shallow_prompt.parameters())
        if args.loss == 'proda':
            model.proda_descriptor_vectors = torch.nn.Parameter(
                get_soft_tokens(model, descriptor_strings).cuda())
            print('model.proda_descriptor_vectors.shape: ', model.proda_descriptor_vectors.shape)
            prompt_params += [model.proda_descriptor_vectors]
        params = [
            {'params':params, 'lr':args.lr},
            {'params':prompt_params, 'lr':args.lr*args.prompt_lr_multi}
        ]
    
    return params

def topk_precision_for_unit_test(f, y, k):
    ''''''
    S = f @ f.T
    for i in range(len(f)):
        S[i,i] = -1.0
    topk_neighbors = S.topk(k).indices
    topk_preds = y[topk_neighbors]
    return (topk_preds == y.unsqueeze(0).T).float().mean().item() * 100.

def topk_precision(f, y, k, splits=32):
    ''''''
    assert len(f) > splits
    m = len(f) // splits
    topk_neighbors = []
    
    i = 0
    for split in range(splits):
        _f = f[i:i+m,:]
        _S = _f @ f.T
        for j in range(m):
            _S[j, j+i] = -1.0 # eliminate sample itself from nearest neighbor
        i += m
        topk_neighbors.append( _S.topk(k).indices ) 
        
    assert i == m * splits
    if i < len(f):
        _f = f[i:,:]
        _S = _f @ f.T
        for j in range(len(f) % splits):
            _S[j, j+i] = -1.0 # eliminate sample itself from nearest neighbor
        topk_neighbors.append( _S.topk(k).indices )
            
    topk_neighbors = torch.cat(topk_neighbors)
    assert len(topk_neighbors) == len(f)
    for i in range(len(f)):
        # double check that I didn't set sample itself as its own nearest neighbor
        assert i != topk_neighbors[i][0]
        
    topk_preds = y[topk_neighbors]
    return (topk_preds == y.unsqueeze(0).T).float().mean().item() * 100.
        
def evaluate(args, model_copy, loader, text):

    model_copy.reset_text(text)
    with torch.no_grad():
        text_features = model_copy.get_text_prototypes()
    model_copy.W = text_features.cuda()

    with torch.no_grad():
        f, y_truth = get_features(loader, model_copy, d=args.d)

    torch.set_num_threads(8)
    with torch.no_grad():
        assert f.shape[0] == y_truth.shape[0]
        y_truth = y_truth.cuda()
        f = F.normalize(f.float().cuda())
        probs = f @ F.normalize(model_copy.W).T
        y_hat = probs.max(1).indices

    # Accuracy
    acc = (y_hat == y_truth).float().mean().item() * 100.
    return acc

def get_descriptor_features(
    tokenized_text_with_descriptors,
    model,
    dim
):
    # populate the initial text feautres matrix
    n_desc, n_classes, token_length = tokenized_text_with_descriptors.shape
    init_text_features = torch.zeros(n_desc, n_classes, dim).cuda()
    for desecriptor_index in range(n_desc):
        with torch.no_grad(), torch.cuda.amp.autocast(enabled=True):
            _t_f = model.encode_text(
                tokenized_text_with_descriptors[desecriptor_index, :, :]
            )
        init_text_features[desecriptor_index, :, :] = F.normalize(_t_f.float())
    return init_text_features.detach()

##############################################################################################
##############################################################################################
##############################################################################################
##############################################################################################
##############################################################################################
##############################################################################################
##############################################################################################
##############################################################################################
##############################################################################################
##############################################################################################
##############################################################################################
##############################################################################################
##############################################################################################
##############################################################################################
##############################################################################################

def train_loop(args, model, ema_model, train_it, 
               n_classes, loss_list,
               scaler, optimizer, scheduler,
               tokenizer=None,
               init_frozen_model=None,
               tokenized_text_with_descriptors=None,
               epoch=0
               # This should be shape [N, C, T]
               # where N is  num of descriptors, C is n classes, T is token length
              ):
    '''
    if tokenized_text_rotation is not None, it must be a list of tensors of shape K x C,
        where K is number of classes and C is context length (usually 77).
        These are like the augmentations of the class prototypes.
        We rotate through the augmentations.
    '''
    ema_val = args.ema
    if args.skip_ema_iters > 0 and epoch==0 :
        print('setting ema_val = 0.0 temporarily')
        ema_val = 0.0
    # ema is the helper for updaing EMA weights
    ema = EMA(beta=ema_val)
    
    if not init_frozen_model is None: init_frozen_model.eval()
    if args.init_lam > 0.0:
        assert not init_frozen_model is None
        if args.train_with_descriptors:
            # populate the initial text feautres matrix
            init_text_features = get_descriptor_features(
                tokenized_text_with_descriptors,
                model=init_frozen_model,
                dim=args.d
            )
        else:
            with torch.no_grad(), torch.cuda.amp.autocast(enabled=True):
                init_text_feature = F.normalize(
                    init_frozen_model.get_text_prototypes().float()
                )

    model.train()

    for it in tqdm(range(args.iters_per_epoch)):
        if ema_val == 0.0 and it >= args.skip_ema_iters:
            ema_val = args.ema
            ema = EMA(beta=ema_val)
            print('restoring ema_val = {}'.format(ema_val))
            ema_model = deepcopy(model)

        x, y = next(train_it)
            
        x = x.cuda()
        y = y.cuda()
        
        init_probs = None
        
        if args.train_with_descriptors:
            # get frozen image features and predictions
            desecriptor_index = it % len(tokenized_text_with_descriptors)
            model.reset_text(tokenized_text_with_descriptors[desecriptor_index, :, :].cpu())
                
        if args.init_lam > 0.0 and args.loss == 'ce':
            assert not init_frozen_model is None
            with torch.no_grad(), torch.cuda.amp.autocast(enabled=True):
                init_image_features = init_frozen_model.encode_image(x)
                init_image_features = F.normalize(init_image_features.float())
            if args.train_with_descriptors:
#                 # get frozen image features and predictions
#                 desecriptor_index = it % len(tokenized_text_with_descriptors)
#                 model.reset_text(tokenized_text_with_descriptors[desecriptor_index, :, :].cpu())
                init_text_feature = init_text_features[desecriptor_index, :, :]
                with torch.no_grad():
                    init_probs = (args.teacher_temp * (init_image_features @ init_text_feature.T)).softmax(1)
            else:
                init_probs = (args.teacher_temp * (init_image_features @ init_text_feature.T)).softmax(1)
        
        def closure():
            ''' Calculate forward pass and backward pass.'''
            nonlocal x, init_probs, it
            if args.loss != 'proda':
                

                efficient_losses = ['clip', 'batched']
                random_prototype_indices = None
                if args.loss == 'batched':
#                     assert False ### ### ###
                    text_batchsize = 500
                    # set random_prototype_indices to be 100 indices, uniformly picked, but must include the ys
                    inds = y.unique()
                    inds_pool = []
                    for _i in range(n_classes):
                        if not _i in inds:
                            inds_pool.append(_i)
                    random.shuffle(inds_pool)
                    random_prototype_indices = torch.cat(
                        (
                            inds, torch.tensor(
                                inds_pool[:(text_batchsize - len(inds))]
                            ).cuda()
                        )
                    )
                    assert len(random_prototype_indices.unique()) == text_batchsize
                    
                    assert init_probs is None 
                    
                    y_prime = torch.zeros_like(y)
                    for __i, yi in enumerate(list(y)):
                        y_prime[__i] = (random_prototype_indices == yi).nonzero().item()
                    
                    y_onehot = get_onehot(y_prime, text_batchsize, label_smoothing=args.label_smoothing)
                
                else:
                    
                    
                    
                    ###
                    if len(y.shape) == 2:
                        assert False  ### ### ###
                        y_onehot = (args.teacher_temp * y).softmax(1)
#                         print(y_onehot)
                    else:
                        y_onehot = get_onehot(y, n_classes, label_smoothing=args.label_smoothing)
                    
                    
                    
                    
                    if not init_probs is None:
                        y_onehot = (1. - args.init_lam) * y_onehot + args.init_lam * init_probs
        
                text_prototype_indices = y if args.loss == 'clip' else random_prototype_indices
                f_image, y_hat, text_prototypes = model(
                    x, return_features=True, 
                    eval_text_features=True,
                    return_text_prototypes=True,
                    autocast=True,
                    text_prototype_indices= text_prototype_indices if args.loss in efficient_losses else None
                )
                f_image = F.normalize(f_image.float())

                #############################################################
                ################## Main finetuning losses ###################
                assert not (args.margin > 0.0 and args.adaptive_margin > 0.0)
                if args.loss == 'ce':
#                     y_hat = y_hat + model.temp * args.adaptive_margin * (
#                         1.0 - text_prototypes.detach()[y] @ text_prototypes.detach().T
#                     )
                    assert args.adaptive_margin == 0.0
                    loss = my_cross_entropy(
                        y_hat, y_onehot, margin=model.temp * args.margin) / args.accum_iter
                elif args.loss == 'supercon':
                    loss_func = pytorch_metric_learning.losses.SupConLoss()
                    supercon_loss = loss_func(f_image, y)
                    ce_loss = my_cross_entropy(
                        y_hat, y_onehot, margin=model.temp * args.margin)
                    loss = args.supercon_lam * supercon_loss + ce_loss
                    loss = loss / args.accum_iter
        
                elif args.loss == 'batched':
                    assert args.adaptive_margin == 0.0
                    loss = my_cross_entropy(
                        y_hat, y_onehot, margin=model.temp * args.margin) / args.accum_iter
                elif args.loss == 'clip':
                    ## CLIP loss
                    loss = multimodal_loss(
                        f_image, text_prototypes,
                        logit_scale=model.temp, 
                        balance=1.0, 
                        margin=args.margin,
                        adaptive_margin=args.adaptive_margin
                    ) / args.accum_iter
                elif args.loss == 'kgcoop':
                    loss = my_cross_entropy(
                        y_hat, get_onehot(y, n_classes, label_smoothing=args.label_smoothing), 
                        margin=0.0) / args.accum_iter
                    loss_kg = 1.0 - (text_prototypes @ init_text_feature.T).mean()
                    loss += args.init_lam * loss_kg / args.accum_iter
                else:
                    assert False
            else:
                assert args.loss == 'proda'
                text_prototype_indices = y
                n_prompt = 4
                with torch.cuda.amp.autocast():
                    f_image = model.encode_image(x)
                f_image = F.normalize(f_image.float())
                
                text_prototypes = torch.zeros(
                    (f_image.shape[0], n_prompt, f_image.shape[-1]), 
                    device=f_image.device
                )
                
                # make sure to define model.proda_descriptor_vectors
                # it should have shape number_of_prompts x number_of_tokens_in_prompt x dim
                assert len(model.proda_descriptor_vectors) % n_prompt == 0
                __prompt_index_start = (it*n_prompt) % len(model.proda_descriptor_vectors)
                for __prompt_index in range(n_prompt):
                    model.shallow_prompt.reset_suffix_vectors(
                        model.proda_descriptor_vectors[__prompt_index + __prompt_index_start, :, :].cpu()
                    )
                    text_prototypes[:, __prompt_index, :] = model.get_text_prototypes(
                        autocast=True, 
                        text_prototype_indices=text_prototype_indices,
                    )
                    
                # memory efficient implementation of proDA loss
                # using contrastive (CLIP) loss instead of CE loss
                # text_prototypes: F.normalized text features
                # shape: n_class x n_prompt x dim
                loss = proda_loss(
                    f_image, text_prototypes,
                    logit_scale=model.temp,
                    margin=args.margin
                ) / args.accum_iter

            scaler.scale(loss).backward()
                
        optimizer.zero_grad()
        closure()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
            
        scheduler.step()
        
        if ema_val > 0.0:
            ema.step_ema(ema_model, model)

    if ema_val == 0.0:
        ema_model = deepcopy(model)
    # always test using the ema model !!!
    return model, ema_model

##############################################################################################
##############################################################################################
##############################################################################################
##############################################################################################
##############################################################################################
##############################################################################################
##############################################################################################
##############################################################################################
##############################################################################################
##############################################################################################
##############################################################################################
##############################################################################################
##############################################################################################
##############################################################################################
##############################################################################################

def train_loop_soft (args, model, ema_model, train_it, 
               n_classes, loss_list,
               scaler, optimizer, scheduler,
               tokenizer=None,
               init_frozen_model=None,
               tokenized_text_with_descriptors=None,
               epoch=0,
               tokenized_text=None
               # This should be shape [N, C, T]
               # where N is  num of descriptors, C is n classes, T is token length
              ):
    '''
    if tokenized_text_rotation is not None, it must be a list of tensors of shape K x C,
        where K is number of classes and C is context length (usually 77).
        These are like the augmentations of the class prototypes.
        We rotate through the augmentations.
    '''
    ema_val = args.ema
    if args.skip_ema_iters > 0 and epoch==0 :
        print('setting ema_val = 0.0 temporarily')
        ema_val = 0.0
    # ema is the helper for updaing EMA weights
    ema = EMA(beta=ema_val)
    
    if not init_frozen_model is None: init_frozen_model.eval()
    if args.init_lam > 0.0:
        assert not init_frozen_model is None
        if args.train_with_descriptors:
            # populate the initial text feautres matrix
            init_text_features = get_descriptor_features(
                tokenized_text_with_descriptors,
                model=init_frozen_model,
                dim=args.d
            )
        else:
            with torch.no_grad(), torch.cuda.amp.autocast(enabled=True):
                init_text_feature = F.normalize(
                    init_frozen_model.get_text_prototypes().float()
                )

    model.train()
    
    #################################################################
    #################################################################
    #################################################################
    teacher_model, _, _ = open_clip.create_model_and_transforms(
            'ViT-L-14', 
            pretrained='openai',
            cache_dir=args.cache_dir
        )
    teacher_model.eval().cuda()
    
    with torch.no_grad(), torch.cuda.amp.autocast(enabled=True):
        teacher_text_feature = F.normalize(
            teacher_model.encode_text(tokenized_text.cuda()).float()
        )
    #################################################################
    #################################################################
    #################################################################

    for it in tqdm(range(args.iters_per_epoch)):
        if ema_val == 0.0 and it >= args.skip_ema_iters:
            ema_val = args.ema
            ema = EMA(beta=ema_val)
            print('restoring ema_val = {}'.format(ema_val))
            ema_model = deepcopy(model)

        x, y = next(train_it)
            
        x = x.cuda()
        y = y.cuda()
        
        init_probs = None
        
        if args.train_with_descriptors:
            # get frozen image features and predictions
            desecriptor_index = it % len(tokenized_text_with_descriptors)
            model.reset_text(tokenized_text_with_descriptors[desecriptor_index, :, :].cpu())
                
        if args.init_lam > 0.0 and args.loss == 'ce':
            assert not init_frozen_model is None
            with torch.no_grad(), torch.cuda.amp.autocast(enabled=True):
                init_image_features = init_frozen_model.encode_image(x)
                init_image_features = F.normalize(init_image_features.float())
            if args.train_with_descriptors:
#                 # get frozen image features and predictions
#                 desecriptor_index = it % len(tokenized_text_with_descriptors)
#                 model.reset_text(tokenized_text_with_descriptors[desecriptor_index, :, :].cpu())
                init_text_feature = init_text_features[desecriptor_index, :, :]
                with torch.no_grad():
                    init_probs = (args.teacher_temp * (init_image_features @ init_text_feature.T)).softmax(1)
            else:
                init_probs = (args.teacher_temp * (init_image_features @ init_text_feature.T)).softmax(1)
                
                
                
                
        #################################################################
        with torch.no_grad(), torch.cuda.amp.autocast(enabled=True):
            teacher_image_features = teacher_model.encode_image(x)
            teacher_image_features = F.normalize(teacher_image_features.float())
        teacher_probs = (args.teacher_temp * (teacher_image_features @ teacher_text_feature.T)).softmax(1)
        #################################################################
                
        
        def closure():
            ''' Calculate forward pass and backward pass.'''
            nonlocal x, init_probs, it, teacher_probs
            if args.loss != 'proda':
                

                efficient_losses = ['clip', 'batched']
                random_prototype_indices = None
                if args.loss == 'batched':
                    assert False ### ### ###
                    text_batchsize = 500
                    # set random_prototype_indices to be 100 indices, uniformly picked, but must include the ys
                    inds = y.unique()
                    inds_pool = []
                    for _i in range(n_classes):
                        if not _i in inds:
                            inds_pool.append(_i)
                    random.shuffle(inds_pool)
                    random_prototype_indices = torch.cat(
                        (
                            inds, torch.tensor(
                                inds_pool[:(text_batchsize - len(inds))]
                            ).cuda()
                        )
                    )
                    assert len(random_prototype_indices.unique()) == text_batchsize
                    
                    assert init_probs is None 
                    
                    y_prime = torch.zeros_like(y)
                    for __i, yi in enumerate(list(y)):
                        y_prime[__i] = (random_prototype_indices == yi).nonzero().item()
                    
                    y_onehot = get_onehot(y_prime, text_batchsize, label_smoothing=args.label_smoothing)
                
                else:
                    
                    
                    
                    ###
                    if len(y.shape) == 2:
                        assert False  ### ### ###
                        y_onehot = (args.teacher_temp * y).softmax(1)
#                         print(y_onehot)
                    else:
                        y_onehot = get_onehot(y, n_classes, label_smoothing=args.label_smoothing)
                    
                    
                    
                    
                    if not init_probs is None:
                        y_onehot = (1. - args.init_lam) * y_onehot + args.init_lam * init_probs
        
                text_prototype_indices = y if args.loss == 'clip' else random_prototype_indices
                f_image, y_hat, text_prototypes = model(
                    x, return_features=True, 
                    eval_text_features=True,
                    return_text_prototypes=True,
                    autocast=True,
                    text_prototype_indices= text_prototype_indices if args.loss in efficient_losses else None
                )
                f_image = F.normalize(f_image.float())

                #############################################################
                ################## Main finetuning losses ###################
                assert not (args.margin > 0.0 and args.adaptive_margin > 0.0)
                if args.loss == 'ce':
#                     y_hat = y_hat + model.temp * args.adaptive_margin * (
#                         1.0 - text_prototypes.detach()[y] @ text_prototypes.detach().T
#                     )
                    assert args.adaptive_margin == 0.0
                    loss = my_cross_entropy(
                        y_hat, teacher_probs, margin=model.temp * args.margin) / args.accum_iter
        
        
#                     print(teacher_probs.max(1))
        
#                     assert False
        
        
        
                elif args.loss == 'batched':
                    assert args.adaptive_margin == 0.0
                    loss = my_cross_entropy(
                        y_hat, y_onehot, margin=model.temp * args.margin) / args.accum_iter
                elif args.loss == 'clip':
                    ## CLIP loss
                    loss = multimodal_loss(
                        f_image, text_prototypes,
                        logit_scale=model.temp, 
                        balance=1.0, 
                        margin=args.margin,
                        adaptive_margin=args.adaptive_margin
                    ) / args.accum_iter
                elif args.loss == 'kgcoop':
                    loss = my_cross_entropy(
                        y_hat, get_onehot(y, n_classes, label_smoothing=args.label_smoothing), 
                        margin=0.0) / args.accum_iter
                    loss_kg = 1.0 - (text_prototypes @ init_text_feature.T).mean()
                    loss += args.init_lam * loss_kg / args.accum_iter
                else:
                    assert False
            else:
                assert args.loss == 'proda'
                text_prototype_indices = y
                n_prompt = 4
                with torch.cuda.amp.autocast():
                    f_image = model.encode_image(x)
                f_image = F.normalize(f_image.float())
                
                text_prototypes = torch.zeros(
                    (f_image.shape[0], n_prompt, f_image.shape[-1]), 
                    device=f_image.device
                )
                
                # make sure to define model.proda_descriptor_vectors
                # it should have shape number_of_prompts x number_of_tokens_in_prompt x dim
                assert len(model.proda_descriptor_vectors) % n_prompt == 0
                __prompt_index_start = (it*n_prompt) % len(model.proda_descriptor_vectors)
                for __prompt_index in range(n_prompt):
                    model.shallow_prompt.reset_suffix_vectors(
                        model.proda_descriptor_vectors[__prompt_index + __prompt_index_start, :, :].cpu()
                    )
                    text_prototypes[:, __prompt_index, :] = model.get_text_prototypes(
                        autocast=True, 
                        text_prototype_indices=text_prototype_indices,
                    )
                    
                # memory efficient implementation of proDA loss
                # using contrastive (CLIP) loss instead of CE loss
                # text_prototypes: F.normalized text features
                # shape: n_class x n_prompt x dim
                loss = proda_loss(
                    f_image, text_prototypes,
                    logit_scale=model.temp,
                    margin=args.margin
                ) / args.accum_iter

            scaler.scale(loss).backward()
                
        optimizer.zero_grad()
        closure()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
            
        scheduler.step()
        
        if ema_val > 0.0:
            ema.step_ema(ema_model, model)

    if ema_val == 0.0:
        ema_model = deepcopy(model)
    # always test using the ema model !!!
    return model, ema_model