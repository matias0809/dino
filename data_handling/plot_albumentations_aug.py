import os
import numpy as np
import rasterio
import cv2
import matplotlib.pyplot as plt
import albumentations as A
import fiftyone as fo
from torch.utils.data import Dataset
from albumentations.pytorch import ToTensorV2



"""
This file takes in the MSI data, and does augmentation with our Albumentations pipeline.
The goal is to remove the center-problem
The output images are saved to outputs/albumentations_aug

NOTE that using border_mode=cv2.BORDER_REFLECT_101 gives some ships a "ghost" effect.
That means, the ship is reflected at the border, but does not get a bounding box. So we introduce false negatives
Still , this is better than having all ships in the center I believe?
"""

# Paths
TIFF_DIR = "/cluster/home/malovhoi/letmecooknow/dino/data_finetune/sentinel2/data_msi"
SAVE_DIR = "/cluster/home/malovhoi/letmecooknow/dino/outputs/albumentations_aug"
os.makedirs(SAVE_DIR, exist_ok=True)

# Load training samples from FiftyOne
dataset = fo.load_dataset("sentinel2")
train_samples = dataset.match_tags("train")

# Albumentations pipeline to diversify ship positions
transform = A.Compose([
    A.RandomResizedCrop(height=336, width=336, scale=(0.4, 1.0), ratio=(0.75, 1.33), p=1.0),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.ShiftScaleRotate(shift_limit=0.20, scale_limit=0.2, rotate_limit=30, border_mode=cv2.BORDER_REFLECT_101, p=1.0),
], bbox_params=A.BboxParams(format="albumentations", label_fields=["class_labels"], min_visibility=0.3))



# Custom Dataset
class ShipDatasetWithAlbumentations(Dataset):
    def __init__(self, samples, tiff_dir, transform):
        self.samples = list(samples)
        self.tiff_dir = tiff_dir
        self.transform = transform

    def __getitem__(self, idx):
        sample = self.samples[idx]
        png_path = sample.filepath
        stem = os.path.basename(png_path).replace("_TCI.png", "")
        tiff_path = os.path.join(self.tiff_dir, f"{stem}_MSI.tif")

        with rasterio.open(tiff_path) as src:
            img = src.read().transpose(1, 2, 0)  # (HWC)
            H, W = img.shape[:2]

        bboxes = []
        for det in sample.gt_bounding_boxes.detections:
            x, y, w, h = det.bounding_box
            #x1 = max(x * W, 0)
            #y1 = max(y * H, 0)
            #x2 = min((x + w) * W, W - 1)
            #y2 = min((y + h) * H, H - 1)
            x2 = min(x+w, 1.0)
            y2 = min(y+h, 1.0)
            bboxes.append([x, y, x2, y2])

        if not bboxes:
            bboxes = [[0, 0, 1, 1]]
            labels = ["no_ship"]
        else:
            labels = ["ship"] * len(bboxes)

        augmented = self.transform(image=img, bboxes=bboxes, class_labels=labels)
        return augmented["image"], augmented["bboxes"]

    def __len__(self):
        return len(self.samples)



# Apply and save 50 examples
dataset_aug = ShipDatasetWithAlbumentations(train_samples, TIFF_DIR, transform)

for i in range(50):
    image, bboxes = dataset_aug[i]
    H, W = image.shape[:2]
    image = np.clip(image / 3000.0, 0, 1)
    
    rgb = image[:, :, [3, 2, 1]]
    rgb = (rgb * 255).astype(np.uint8)

    fig, ax = plt.subplots()
    ax.imshow(rgb)


    for box in bboxes:
        x1, y1, x2, y2 = box
        rect = plt.Rectangle((x1*W, y1*H), x2*W - x1*W, y2*H - y1*H, edgecolor='red', facecolor='none', linewidth=1)
        ax.add_patch(rect)

    ax.set_title(f"Augmented {i}")
    ax.axis("off")
    plt.savefig(os.path.join(SAVE_DIR, f"aug_img_{i}.png"), bbox_inches="tight", pad_inches=0)
    plt.close()