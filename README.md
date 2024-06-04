# Multimodal Unsupervised Domain Generalization (MUDG)
-----------------------------------------------------

![](https://github.com/Chris210634/mudg/blob/main/method.gif?raw=true)

Code in this repo uses code from [multimodal prompt learning](https://github.com/muzairkhattak/multimodal-prompt-learning), which in turn uses code from [Co-CoOp and CoOp](https://github.com/KaiyangZhou/CoOp).

## ⏳ Installation
-------------------

### Installing the datasets

* Install dassl library.
```bash
# Instructions borrowed from https://github.com/KaiyangZhou/Dassl.pytorch#installation

# Clone this repo
git clone https://github.com/KaiyangZhou/Dassl.pytorch.git
cd Dassl.pytorch/

# Install dependencies
pip install -r requirements.txt

# Install this library (no need to re-build if the source code is modified)
python setup.py develop
cd ..
```

* Create a directory somewhere called `data/`. Download all 15 zip files from [this shared Google Drive](https://drive.google.com/drive/folders/1kvh5VG4ruGOcSiHKJX9dWJhPAGVgPSZs?usp=drive_link) and unzip them into `data/`. The resulting file tree should look like:
```
data/
|-- caltech-101
|-- dtd
|-- eurosat
|-- fgvc_aircraft
|-- food-101
|-- imagenet
|-- imagenet-adversarial
|-- imagenet-rendition
|-- imagenet-sketch
|-- imagenetv2
|-- oxford_flowers
|-- oxford_pets
|-- stanford_cars
|-- sun397
|-- ucf101
```

Alternatively, follow the download instructions here (some dataset links are stale):
[installing datasets](https://github.com/muzairkhattak/multimodal-prompt-learning/blob/main/docs/DATASETS.md)

* Modify the following two lines in `argparse_parameters.py` to reflect where you have your `data/` dir and where you want the pretrained CLIP weights to be cached (which could be many gigabytes)

```python
parser.add_argument('--cache_dir', default = "data/", type =str) # set to directory where you want large pretrained model weights to be cached
parser.add_argument('--data_dir', default = "data/", type =str)  # set to parent directory of data/
```

* Install the domain generalization datasets using the scripts from DomainBed Github: [here](https://github.com/facebookresearch/DomainBed/tree/main). The resulting file structure should look like:
```
├── domain_net
│   ├── clipart
│   ├── infograph
│   ├── painting
│   ├── quickdraw
│   ├── real
│   └── sketch
├── office_home
│   ├── Art
│   ├── Clipart
│   ├── ImageInfo.csv
│   ├── imagelist.txt
│   ├── Product
│   └── RealWorld
├── PACS
│   ├── art_painting
│   ├── cartoon
│   ├── photo
│   └── sketch
├── terra_incognita
│   ├── location_100
│   ├── location_38
│   ├── location_43
│   └── location_46
└── VLCS
    ├── Caltech101
    ├── LabelMe
    ├── SUN09
    └── VOC2007
```


### Installing the LAION index

* Install the parquets containing the url of the images.

**Run:** `cd laion && sh download_knn_indices.sh`

* Install the pre-made FAISS index files from LAION.

**Run:** `cd laion && sh download_parquets.sh`

* Modify the following line in `argparse_parameters.py` if you downloaded the parquet files and index files into a different directory:

```python
# parent folder of laion-2B-en containing the parquets and indices.
parser.add_argument('--index_directory', default = 'laion/', type=str)
```

The files under the `laion` directory should look like:
```
laion
├── laion-2B-en
│   ├── 0000.parquet
... ... ...
│   ├── 2313.parquet
│   ├── knn.index00
... ... ...
│   ├── knn.index54
├── parquet_lengths.list

```

## ✨ NEW: Paired k-means Index Training
---------------------------------------------

To build LAION indices using the paired k-means training presented in the paper, please follow instructions at [https://github.com/Chris210634/laion-index](https://github.com/Chris210634/laion-index).

This requires a substantial amount of memory and disk space. The instructions here will produce an index split into 58 shards in the format `knn.paired_QT_4bit.index*`. Please copy these into `laion/laion-2B-en`. The resulting directory structure should look like:

```
laion
├── laion-2B-en
│   ├── 0000.parquet
... ... ...
│   ├── 2313.parquet
│   ├── knn.paired_QT_4bit.index00
... ... ...
│   ├── knn.paired_QT_4bit.index57
├── parquet_lengths_paired.list

```

Just remember to use the `parquet_lengths_paired.list` list in the retrieval step. 


## ✨ NEW: Adaptive Label Text Augmentations
---------------------------------------------

To run the adaptive label augmentation presented in the paper:

```bash
for dataset in ImageNet Caltech101 OxfordPets StanfordCars Flowers102 Food101 FGVCAircraft SUN397 DTD EuroSAT UCF101 ImageNetV2 ImageNetSketch ImageNetA ImageNetR DomainNet.clipart OfficeHome.real PACS.art VLCS.caltech TerraInc.100; do
    python better_augmentations_zero-shot.py --dataset $dataset --modelname ViT-L-14 --pretrained openai --d 768
    python better_augmentations_zero-shot.py --dataset $dataset --modelname ViT-B-16 --pretrained openai --d 512
done
cp cache/better_descriptors_sorted_ViT-B-16_DomainNet.clipart.list cache/better_descriptors_sorted_ViT-B-16_DomainNet.list
cp cache/better_descriptors_sorted_ViT-B-16_OfficeHome.real.list   cache/better_descriptors_sorted_ViT-B-16_16_OfficeHome.list
cp cache/better_descriptors_sorted_ViT-B-16_PACS.art.list          cache/better_descriptors_sorted_ViT-B-16_PACS.list
cp cache/better_descriptors_sorted_ViT-B-16_VLCS.caltech.list      cache/better_descriptors_sorted_ViT-B-16_VLCS.list
cp cache/better_descriptors_sorted_ViT-B-16_TerraInc.100.list      cache/better_descriptors_sorted_ViT-B-16_TerraInc.list
cp cache/better_descriptors_sorted_ViT-L-14_DomainNet.clipart.list cache/better_descriptors_sorted_ViT-L-14_DomainNet.list
cp cache/better_descriptors_sorted_ViT-L-14_OfficeHome.real.list   cache/better_descriptors_sorted_ViT-L-14_OfficeHome.list
cp cache/better_descriptors_sorted_ViT-L-14_PACS.art.list          cache/better_descriptors_sorted_ViT-L-14_PACS.list
cp cache/better_descriptors_sorted_ViT-L-14_VLCS.caltech.list      cache/better_descriptors_sorted_ViT-L-14_VLCS.list
cp cache/better_descriptors_sorted_ViT-L-14_TerraInc.100.list      cache/better_descriptors_sorted_ViT-L-14_TerraInc.list
```

The selected augmentations are stored in `cache/better_descriptors_sorted_*.list`.

To train with the selected augmentations, run `run_b16_initreg_paired.sh` and `run_l14_initreg_paired.sh` using the same settings as above (`run_b16.sh` and `run_l14.sh`).


## 🧪 Run Experiments
---------------------------

(1) Retrieve training samples for each dataset. 

```bash
sh run_retrieve.sh ImageNet     16 64 96 cache parquet_lengths.list 8 cache
sh run_retrieve.sh Caltech101   64 64 384 cache parquet_lengths.list 8 cache
sh run_retrieve.sh OxfordPets   64 64 384 cache parquet_lengths.list 8 cache
sh run_retrieve.sh StanfordCars 16 64 96 cache parquet_lengths.list 8 cache
sh run_retrieve.sh Flowers102   64 64 384 cache parquet_lengths.list 8 cache
sh run_retrieve.sh Food101      64 64 384 cache parquet_lengths.list 8 cache
sh run_retrieve.sh FGVCAircraft 64 64 384 cache parquet_lengths.list 8 cache
sh run_retrieve.sh SUN397       16 64 96 cache parquet_lengths.list 8 cache
sh run_retrieve.sh DTD          64 64 384 cache parquet_lengths.list 8 cache
sh run_retrieve.sh EuroSAT      64 64 384 cache parquet_lengths.list 8 cache
sh run_retrieve.sh UCF101       64 64 384 cache parquet_lengths.list 8 cache
sh run_retrieve.sh ImageNetV2   16 64 96 cache parquet_lengths.list 8 cache
sh run_retrieve.sh ImageNetSketch 16 64 96 cache parquet_lengths.list 8 cache
sh run_retrieve.sh ImageNetA    16 64 96 cache parquet_lengths.list 8 cache
sh run_retrieve.sh ImageNetR    16 64 96 cache parquet_lengths.list 8 cache
sh run_retrieve.sh DomainNet    16 64 96 cache parquet_lengths.list 8 cache
sh run_retrieve.sh OfficeHome   64 64 384 cache parquet_lengths.list 8 cache
sh run_retrieve.sh PACS         64 64 384 cache parquet_lengths.list 8 cache
sh run_retrieve.sh VLCS         64 64 384 cache parquet_lengths.list 8 cache
sh run_retrieve.sh TerraInc     64 64 384 cache parquet_lengths.list 8 cache
```

The format is: `bash sh run_retrieve.sh $dataset $m $n $k cache parquet_lengths.list $nprobe cache`.

The first cache can be a different temporary directory if you're worried about disk space. If you're using the paired k-means indices instead of the LAION indices downloaded from their server, use `parquet_lengths_paired.list` instead of `parquet_lengths.list`.

```bash 
sh run_retrieve.sh $dataset $m $n $k cache_paired parquet_lengths_paired.list $nprobe cache_paired
```

The above script queries the index and parquet files, downloads the images and makes a list of training image filenames with pseudo-labels. These files are stored in `cache`. For example, the retrieved aircraft dataset file structure looks like:

```
cache
├── 64nn_x_16waffle
│   └── FGVCAircraft
│       ├── laion
│       ├── laion_jpegs
│       ├── result_tup_list.FGVCAircraft_64nn_x_16waffle
│       ├── result_tup_list.FGVCAircraft_64nn_x_16waffle.clustered
```

(2) Finetune CLIP model.

For CLIP B/16:

```bash
for seed in 1 2 3;
do
    sh run_b16.sh ImageNet     300 16 $seed
    sh run_b16.sh Caltech101   100 64 $seed
    sh run_b16.sh OxfordPets   200 64 $seed
    sh run_b16.sh StanfordCars 1000 16 $seed
    sh run_b16.sh Flowers102   200 64 $seed
    sh run_b16.sh Food101      100 64 $seed
    sh run_b16.sh FGVCAircraft 1000 64 $seed
    sh run_b16.sh SUN397       300 16 $seed
    sh run_b16.sh DTD          200 64 $seed
    sh run_b16.sh EuroSAT      200 64 $seed
    sh run_b16.sh UCF101       200 64 $seed
    sh run_b16.sh ImageNetV2    200 16 $seed
    sh run_b16.sh ImageNetSketch 200 16 $seed
    sh run_b16.sh ImageNetA    200 16 $seed
    sh run_b16.sh ImageNetR    200 16 $seed
    sh run_b16.sh DomainNet    200 16 $seed
    sh run_b16.sh OfficeHome   200 64 $seed
    sh run_b16.sh PACS         100 64 $seed
    sh run_b16.sh VLCS         50 64 $seed
    sh run_b16.sh TerraInc     100 64 $seed
done
```

For CLIP L/14:
```bash
for seed in 1 2 3;
do
    sh run_l14.sh ImageNet     300 16 $seed
    sh run_l14.sh Caltech101   100 64 $seed
    sh run_l14.sh OxfordPets   200 64 $seed
    sh run_l14.sh StanfordCars 1000 16 $seed
    sh run_l14.sh Flowers102   200 64 $seed
    sh run_l14.sh Food101      100 64 $seed
    sh run_l14.sh FGVCAircraft 1000 64 $seed
    sh run_l14.sh SUN397       300 16 $seed
    sh run_l14.sh DTD          200 64 $seed
    sh run_l14.sh EuroSAT      200 64 $seed
    sh run_l14.sh UCF101       200 64 $seed
    sh run_l14.sh ImageNetV2    200 16 $seed
    sh run_l14.sh ImageNetSketch 200 16 $seed
    sh run_l14.sh ImageNetA    200 16 $seed
    sh run_l14.sh ImageNetR    200 16 $seed
    sh run_l14.sh DomainNet    200 16 $seed
    sh run_l14.sh OfficeHome   200 64 $seed
    sh run_l14.sh PACS         100 64 $seed
    sh run_l14.sh VLCS         50 64  $seed
    sh run_l14.sh TerraInc     100 64 $seed
done
```

The format is: `bash sh run_b16.sh $dataset $iterations $m $seed`.


## 🧪 Zero-shot Baselines
---------------------------

```bash
# WaffleCLIP
python main_crossdataset.zs.py --seed 1 --eval_only 1 --soup_eval 1 --descriptor_file cache/waffle_descriptors_512_count.list --modelname ViT-B-16 --pretrained openai --d 512
python main_crossdataset.zs.py --seed 1 --eval_only 1 --soup_eval 1 --descriptor_file cache/waffle_descriptors_512_count.list --modelname ViT-L-14 --pretrained openai --d 768

# Random descriptors
python main_crossdataset.zs.py --seed 1 --eval_only 1 --soup_eval 1 --descriptor_file cache/descriptions.list --modelname ViT-B-16 --pretrained openai --d 512
python main_crossdataset.zs.py --seed 1 --eval_only 1 --soup_eval 1 --descriptor_file cache/descriptions.list --modelname ViT-L-14 --pretrained openai --d 768

# OpenAI Ensemble
python main_crossdataset.zs.py --seed 1 --eval_only 1 --openai_eval 1 --modelname ViT-B-16 --pretrained openai --d 512
python main_crossdataset.zs.py --seed 1 --eval_only 1 --openai_eval 1 --modelname ViT-L-14 --pretrained openai --d 768

# GPT Descriptors (classification by description)
python main_crossdataset.zs.py --seed 1 --eval_only 1 --gpt_centroid_eval 1 --gpt_score_averaging_eval 1 --modelname ViT-B-16 --pretrained openai --d 512
python main_crossdataset.zs.py --seed 1 --eval_only 1 --gpt_centroid_eval 1 --gpt_score_averaging_eval 1 --modelname ViT-L-14 --pretrained openai --d 768
```





