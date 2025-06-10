import os
import torch
from torchvision.utils import save_image
from specdetr.specdetr_custom_datamodules import DataAugmentationSpecDetr
from main_dino import DataAugmentationDINO
from torchvision.transforms import ToPILImage

# RGB band indices in Sentinel-2 [13-band] images
BAND_RED = 3
BAND_GREEN = 2
BAND_BLUE = 1

def save_rgb_augmented_patches(
    pt_file_path: str,
    output_dir: str,
    augmentor,
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
):
    """
    Loads one image from a pre-saved .pt file, applies augmentations using the provided augmentor,
    and saves the original image + 2 global + 8 local crops as RGB .png files.
    
    Args:
        pt_file_path (str): Path to the .pt file containing a list of [13, H, W] tensors.
        output_dir (str): Directory to store the RGB images.
        augmentor (callable): Augmentation object, e.g., instance of DataAugmentationSpecDetr.
        index (int): Index of the image patch to load from the .pt file.
        device (str): "cuda" or "cpu"
    """
    os.makedirs(output_dir, exist_ok=True)

    # Load .pt file and get one image
    image = torch.load(pt_file_path).unsqueeze(0).to(device)

    # Save original RGB image
    rgb_original = image[0][[BAND_RED, BAND_GREEN, BAND_BLUE], :, :]
    save_image(rgb_original, os.path.join(output_dir, "original_rgb.png"))

    # Generate augmented crops
    if isinstance(augmentor, DataAugmentationSpecDetr):
        crops = augmentor(image)
    elif isinstance(augmentor, DataAugmentationDINO):
        pil_image = ToPILImage()(rgb_original)
        crops = augmentor(pil_image)

    # Save each crop as RGB
    for i, crop in enumerate(crops):
        if isinstance(augmentor, DataAugmentationDINO):
            rgb_crop = crop
        else:
            rgb_crop = crop[0][[BAND_RED, BAND_GREEN, BAND_BLUE], :, :]
        save_image(rgb_crop, os.path.join(output_dir, f"crop_{i+1}.png"))

    print(f"Saved original + {len(crops)} crops to {output_dir}")

# Example usage

pt_file_path = "/cluster/home/malovhoi/letmecooknow/dino/data_extra/single_img_pt_336/patch_001000.pt"
transform_specdetr = DataAugmentationSpecDetr()
transform_dino = DataAugmentationDINO(global_crops_scale=(0.4,1.0),local_crops_scale=(0.05,0.4) ,local_crops_number=8) ##NEED TO HAVE PIL FILES TO MAKE THIS WORK!

save_rgb_augmented_patches(
    pt_file_path=pt_file_path,
    output_dir="/cluster/home/malovhoi/letmecooknow/dino/outputs/specdetr_aug_views/from_pt",
    augmentor=transform_specdetr,
)

###HUSK!!! NORMALISERING MÅ KOMMENTERES UT I DataAugmentationDINO NÅR DENNE BRUKES!!!
save_rgb_augmented_patches(
    pt_file_path= pt_file_path,
    output_dir="/cluster/home/malovhoi/letmecooknow/dino/outputs/dino_aug_views/from_pt",
    augmentor=transform_dino,
) 