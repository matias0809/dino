import torch
from torchgeo.datasets import Sentinel2
from torchgeo.samplers import GridGeoSampler

from tqdm import tqdm
from pathlib import Path


### THIS FILE TAKES THE .SAFE FILES FROM SENTINEL-2 AND PRELOADS THEM INTO RAM AS TENSORS
### IT IS USED TO SPEED UP THE TRAINING PROCESS BY AVOIDING DISK I/O DURING TRAINING
###RATHER USE "load_patches_as_pt.py". THAT FILE LOADS a .pt FILE PER PATCH!
###THIS FILE SHOULD NOT BE USED!


""" 
### THIS FILE TAKES THE .SAFE FILES FROM SENTINEL-2 AND PRELOADS THEM INTO DISK AS TENSORS
### IT SAVES THE ENTIRE .SAFE AS ONE .PT FILE!!

dataset_raw = Sentinel2(paths="/cluster/home/malovhoi/letmecooknow/dino/data/single_img", res=10)
geosampler = GridGeoSampler(dataset=dataset_raw, size=336, stride=336)
patches = list(geosampler)

all_images = []
for i, bb in enumerate(patches):
    image = dataset_raw[bb]["image"] / 10000
    all_images.append(image)
    if i % 100 == 0:
        print(f"Saved {i} / {len(patches)}")

# Save all tensors to a file
torch.save(all_images, "/cluster/home/malovhoi/letmecooknow/dino/data/preloaded_datasets/single_img/preloaded_dataset_336.pt") """

