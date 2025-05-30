from torch.utils.data import Dataset
from torchgeo.datasets import Sentinel2
from torchgeo.samplers import GridGeoSampler
import torch
import torch.distributed as dist
import os
import torchvision
from specdetr.specdetrutils import MSIBrightnessContrast
from specdetr.specdetr_custom_datamodules import SentinelPatchDataset
from specdetr.specdetr_custom_datamodules import DataAugmentationSpecDetr

import torchvision.transforms as T
import torch.nn.functional as F
import kornia.augmentation as K
import kornia.filters as KF
import torch.nn as nn
from PIL import Image
from datetime import datetime
from torchvision import datasets, transforms
import utils

class DataAugmentationDINO(object):
    def __init__(self, global_crops_scale, local_crops_scale, local_crops_number):
        flip_and_color_jitter = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply(
                [transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)],
                p=0.8
            ),
            transforms.RandomGrayscale(p=0.2),
        ])
        normalize = transforms.Compose([
            transforms.ToTensor(),
            #transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

        # first global crop
        self.global_transfo1 = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=global_crops_scale, interpolation=Image.BICUBIC),
            flip_and_color_jitter,
            utils.GaussianBlur(1.0),
            normalize,
        ])
        # second global crop
        self.global_transfo2 = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=global_crops_scale, interpolation=Image.BICUBIC),
            flip_and_color_jitter,
            utils.GaussianBlur(0.1),
            utils.Solarization(0.2),
            normalize,
        ])
        # transformation for the local small crops
        self.local_crops_number = local_crops_number
        self.local_transfo = transforms.Compose([
            transforms.RandomResizedCrop(96, scale=local_crops_scale, interpolation=Image.BICUBIC),
            flip_and_color_jitter,
            utils.GaussianBlur(p=0.5),
            normalize,
        ])

    def __call__(self, image):
        crops = []
        crops.append(self.global_transfo1(image))
        crops.append(self.global_transfo2(image))
        for _ in range(self.local_crops_number):
            crops.append(self.local_transfo(image))
        return crops
    


os.environ['MASTER_ADDR'] = '127.0.0.1'
os.environ['MASTER_PORT'] = '29500'


dist.init_process_group(
    backend="nccl",
    init_method="env://",
    world_size=1,
    rank=0,
    )


    
    


    
    
 
###TEST    



transform = DataAugmentationSpecDetr()  # Example transform, can be replaced with any other transform
transformdino = DataAugmentationDINO((0.4,1.0), (0.05, 0.4), 8) # Example transform, can be replaced with any other transform

dataset = Sentinel2(
    paths="/home/mnachour/master/dino/data",  # R, G, B at 10m resolution
    res=10,                       # force 10m resolution
    #transforms=ApplyTransformToImage(transform),  # apply the transform to the image
)

geo_sampler = GridGeoSampler(
    dataset=dataset,
    size=256,  # patch size in pixels
    stride=256
)

patches = list(geo_sampler) 


""" idx = patches[200]
img_raw = dataset[idx]["image"]  # This is a tensor of shape [13, 256, 256]
img_raw_rgb = img_raw[[3, 2, 1], :, :]    #NEED TO EXTRACT RGB CORRECTLY


# Save raw RGB image (normalized to [0, 1] assumed via transform)
save_tensor_as_image(img_raw_rgb / 10000.0, os.path.join(output_dir, "raw_rgb.png"))

# Apply full transform (will normalize inside)
crops = transform(img_raw)  # 10 crops, all [13, H, W]

# Save RGB channels of each crop
for i, crop in enumerate(crops):
    crop_rgb = crop[[3, 2, 1], :, :]  # Extract RGB
    save_tensor_as_image(crop_rgb, os.path.join(output_dir, f"crop_{i:02d}_rgb.png")) """



#################################
"""
dataset_new = SentinelPatchDataset(
    base_dataset=dataset,
    index_list=patches,
    transform=transform 
)

sampler_new = torch.utils.data.DistributedSampler(dataset_new, shuffle=True)  # Totally compatible now!

data_loader = torch.utils.data.DataLoader(
    dataset_new,
    sampler=sampler_new,
    batch_size=8,
    num_workers=10,
    pin_memory=True,
    drop_last=True,
)

imgs, labels = (next(iter(data_loader)))  # Should print a batch of images and None labels

print("NÅ")
print(len(imgs))
for i in range(len(imgs)):
    print(imgs[i].shape)  # Should print the shape of each image tensor in the batch


print(imgs[0]) """

    
if dist.is_available() and dist.is_initialized():
    dist.destroy_process_group() 