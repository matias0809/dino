from torch.utils.data import Dataset
import torch
import kornia.augmentation as K
import time
from pathlib import Path

from specdetr.specdetrutils import MSIBrightnessContrast

###Denne klassen henta fra .SAFE filer, så igjen slet med bottlenecken av å loade bilder fra jp2 filer
""" class SentinelPatchDataset(Dataset):
    def __init__(self, base_dataset, index_list, transform=None):
        self.base_dataset = base_dataset           # e.g., Sentinel2
        self.index_list = index_list               # list of BoundingBox objects
        self.transform = transform

    def __len__(self):
        return len(self.index_list)

    def __getitem__(self, idx):
        print(f"Getting item {idx}")
        bounding_box = self.index_list[idx]
        #print(f"\nIndex {idx}, BBox: {bounding_box}")
        #print(f"Base dataset CRS: {self.base_dataset.crs}") 
        t0 = time.time()
        sample = self.base_dataset[bounding_box]
        print(f"Loaded image in {time.time() - t0:.2f}s")   # This works! Sentinel2 supports this
        image = sample["image"]                    # Tensor shape [13, 256, 256]
        if self.transform:
            image = self.transform(image)
        return image, 0   """


###Denne klassen bruke .pt fil, MEN antok at en .pt fil hadde alle patches. Så self.samples ble ALT for stor, og vi gikk tom for RAM!
""" class PreloadedPatchDataset(torch.utils.data.Dataset):
    ###Class used for loading dataset that already has been divided by 10000, and loaded as tensors into RAM.
    ###It loads a .pt file which contains a list of tensors, each representing a preprocessed image patch.
    def __init__(self, tensor_file_path):
        print("Loading preprocessed .pt dataset into RAM...")
        start = time.time()
        self.samples = torch.load(tensor_file_path)
        print(f"Loaded {len(self.samples)} samples in {time.time() - start:.2f}s")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx], 0   """
    


class PreloadedPatchDatasetv2(torch.utils.data.Dataset):
    def __init__(self, patch_dir, transform=None):
        self.patch_dir = Path(patch_dir)
        print("Loading preprocessed .pt dataset into RAM...")
        start = time.time()
        self.paths = sorted(self.patch_dir.glob("patch_*.pt"))
        print(f"Loaded {len(self.paths)} samples in {time.time() - start:.2f}s")
        self.transform = transform
    
    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        #print("GETTIN ITEM", idx)
        image = torch.load(self.paths[idx])
        #print(f"Loaded image from {self.paths[idx]}")
        if self.transform:
            image = self.transform(image)
        return image, 0  # Dummy label
    

###Denne klassen lasta inn patches fra .SAFE filene. Bottleneck, så funka ikke!
""" class SentinelPatchDataset(torch.utils.data.Dataset):
    ###Used for loading Sentinel-2 patches from .SAFE files. Has to load patches every time we start a training run.
    def __init__(self, base_dataset, index_list):
        self.samples = []
        print("Preloading image patches into RAM...")
        start = time.time()
        for i, bb in enumerate(index_list):
            self.samples.append(base_dataset[bb]["image"])
            if i % 100 == 0:
                print(f"  Loaded {i} / {len(index_list)}")
        print(f"Done loading {len(self.samples)} patches in {time.time() - start:.2f}s")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx], 0  """
    

class DataAugmentationSpecDetr:
    def __init__(self, global_crops_scale=(0.4, 1.0), local_crops_scale=(0.05, 0.4), local_crops_number=8, num_channels=13):
        self.local_crops_number = local_crops_number
        self.num_channels = num_channels 

        self.global_transfo1 = torch.nn.Sequential(
            K.RandomResizedCrop((224, 224), scale=global_crops_scale), #spesifisere bicubic?
            K.RandomHorizontalFlip(p=0.5),
            MSIBrightnessContrast(brightness=0.4, contrast=0.4, p_brightness=0.8, p_contrast=0.8),
            K.RandomGaussianBlur((23, 23), sigma=(0.1, 1.0), p=1.0),
        )

        self.global_transfo2 = torch.nn.Sequential(
            K.RandomResizedCrop((224, 224), scale=global_crops_scale), #SPEISFISERE BICUBIC?
            K.RandomHorizontalFlip(p=0.5),
            MSIBrightnessContrast(brightness=0.4, contrast=0.4, p_brightness=0.8, p_contrast=0.8),  # Initialize brightness/contrast jitter
            K.RandomGaussianBlur((23, 23), sigma=(0.1, 1.0), p=0.1),
        )

        self.local_transfo = torch.nn.Sequential(
            K.RandomResizedCrop((96, 96), scale=local_crops_scale), #spesifisere bicubic?
            K.RandomHorizontalFlip(p=0.5),
            MSIBrightnessContrast(brightness=0.4, contrast=0.4, p_brightness=0.8, p_contrast=0.8),
            K.RandomGaussianBlur((23, 23), sigma=(0.1, 1.0), p=0.5),
        )

    def __call__(self, batch):
        # image shape: [C, H, W] → Kornia expects [B, C, H, W]
        """
        This was the original code, when we sent transform inside SentinelPatchDataset. 
        Then transforms were applied to each image patch individually. 
        Now we recieve a batch from the dataloader of (B, C, H, W) shape!"""
        """
        device = image.device
        print(f"Kornia transforms running on: {device}")
        image = image/10000  ##HUSK Å SKRU DENNE AV/PÅ BASERT PÅ OM DET ER SENTINEL FRA .SAFE ELLER IKKE!!
        image = image.unsqueeze(0)
        crops = []
        crops.append(self.global_transfo1(image).squeeze(0))
        crops.append(self.global_transfo2(image).squeeze(0))
        for _ in range(self.local_crops_number):
            crops.append(self.local_transfo(image).squeeze(0))
        return crops """
        #device = batch.device
        #print(f"Kornia transforms running on: {device}")
        #batch = batch/10000  #IMPORTANT, UNCOMMENT IF USING .SAFE FILES! BUT, .pt FILES ARE ALREADY NORMALIZED!
        crops = []
        crops.append(self.global_transfo1(batch))
        crops.append(self.global_transfo2(batch))
        for _ in range(self.local_crops_number):
            crops.append(self.local_transfo(batch))
        #print(f"Number of crops generated: {len(crops)}")
        return crops 