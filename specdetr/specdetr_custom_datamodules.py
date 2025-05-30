from torch.utils.data import Dataset
import torch
import kornia.augmentation as K

from specdetr.specdetrutils import MSIBrightnessContrast

class SentinelPatchDataset(Dataset):
    def __init__(self, base_dataset, index_list, transform=None):
        self.base_dataset = base_dataset           # e.g., Sentinel2
        self.index_list = index_list               # list of BoundingBox objects
        self.transform = transform

    def __len__(self):
        return len(self.index_list)

    def __getitem__(self, idx):
        bounding_box = self.index_list[idx]
        sample = self.base_dataset[bounding_box]   # This works! Sentinel2 supports this
        image = sample["image"]                    # Tensor shape [13, 256, 256]
        if self.transform:
            image = self.transform(image)
        return image, 0  
    

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

    def __call__(self, image):
        # image shape: [C, H, W] → Kornia expects [B, C, H, W]
        image = image/10000  ##HUSK Å SKRU DENNE AV/PÅ BASERT PÅ OM DET ER SENTINEL FRA .SAFE ELLER IKKE!!
        image = image.unsqueeze(0)
        crops = []
        crops.append(self.global_transfo1(image).squeeze(0))
        crops.append(self.global_transfo2(image).squeeze(0))
        for _ in range(self.local_crops_number):
            crops.append(self.local_transfo(image).squeeze(0))
        return crops