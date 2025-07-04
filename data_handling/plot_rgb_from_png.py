from PIL import Image
import os
import torch
import torchvision
import torchvision.transforms as T
from specdetr.specdetr_custom_datamodules import DataAugmentationSpecDetr
from main_dino import DataAugmentationDINO ##WILL SOON BE FROM main_dino!!

def process_png_with_dino_transform(image_path, transform, output_dir):
    """
    This works for PNG image from sentinel2, NOT for .SAFE file!
    It then plots the original RGB image, as well as the 2 global crops and 8 local crops.
    The DINO-transform expects a PIL-image, WHILE the SpecDetr transform expects a tensor.
    save_tensor_as_image expects a tensor in [0, 1] range.

    """
    os.makedirs(output_dir, exist_ok=True)

    # Load the PNG image as a 3-channel tensor [3, H, W] in [0, 1]
    pil_img = Image.open(image_path).convert("RGB")
    rgb_tensor = T.ToTensor()(pil_img)  # shape [3, H, W]


    # Save original RGB
    save_tensor_as_image(rgb_tensor, os.path.join(output_dir, "original_rgb.png"))

    # Apply your original transform (assuming you've commented out /10000 inside)
    if isinstance(transform, DataAugmentationDINO):
        crops = transform(pil_img)
    if isinstance(transform, DataAugmentationSpecDetr):
        ###TODO Denne ble laget da vi sendte inn enkelt-bilder. Nå sender vi batches, så kanskje vi må squeeze
        crops = transform(rgb_tensor) #REMEMBER TO CHECK IF WE DIVIDE BY 10000 IN DataAugmentationSpecDetr! SHOULD NOT BE DIVIDING!


    # Save RGB channels from each crop
    for i, crop in enumerate(crops):
        save_tensor_as_image(crop, os.path.join(output_dir, f"crop_{i:02d}_rgb.png"))

    print(f"✅ Saved original and {len(crops)} crops to {output_dir}")

def save_tensor_as_image(tensor, path):
    """
    Save a 3-channel tensor image (C, H, W) normalized in [0,1] to a file path as PNG.
    Assumes tensor is already on CPU.
    """
    torchvision.utils.save_image(tensor, path)

###################
# Example usage
transform_specdetr = DataAugmentationSpecDetr(
    global_crops_scale=(0.4, 1.0),
    local_crops_scale=(0.05, 0.4),
    local_crops_number=8,
)
transform_dino = DataAugmentationDINO(
    global_crops_scale=(0.4, 1.0),
    local_crops_scale=(0.05, 0.4),
    local_crops_number=8,
)
# An image from the png-data from prosjektoppave:
image_path = "/cluster/home/malovhoi/letmecooknow/dino/data_finetune/sentinel2/data/6dfb1a6c-b457-4a25-b5df-33920545ddbb_TCI.png"

output_dir_specdetr = "/cluster/home/malovhoi/letmecooknow/dino/outputs/specdetr_aug_views"
output_dir_dino = "/cluster/home/malovhoi/letmecooknow/dino/outputs/dino_aug_views"

process_png_with_dino_transform(image_path, transform_specdetr, output_dir_specdetr)
process_png_with_dino_transform(image_path, transform_dino, output_dir_dino)