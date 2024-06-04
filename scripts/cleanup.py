import os, torch
import shutil
from tqdm import tqdm
import sys

# folder_path = 'cache/64nn_x_16waffle/ImageNet'
# result_tup_list_path = 'result_tup_list.ImageNet_64nn_x_16waffle.clustered'

folder_path = sys.argv[1]
result_tup_list_path = sys.argv[2]

def _cleanup(folder_path, result_tup_list_path):
    files = os.listdir(folder_path)
    for file in files:
        if file.endswith('.parquet') or file.endswith('.list'):
            os.remove(os.path.join(folder_path, file))

    if 'laion' in files:
        shutil.rmtree(os.path.join(folder_path, 'laion'))
        
    tup_list = torch.load(os.path.join(folder_path, result_tup_list_path))
    jpegs = [i[0].split('/')[-1] for i in tup_list]

    if 'laion_jpegs' in files:
        jpeg_folder = os.path.join(folder_path, 'laion_jpegs')
        jpeg_files = os.listdir(jpeg_folder)
        for file_to_be_deleted in tqdm(jpeg_files):
            filepath = os.path.join(jpeg_folder, file_to_be_deleted)
            if not file_to_be_deleted in jpegs: # keep files in jpegs
                os.remove(filepath)
                
print('begin cleanup of {} {}'.format(folder_path, result_tup_list_path))
_cleanup(folder_path, result_tup_list_path)
print('finished cleanup')