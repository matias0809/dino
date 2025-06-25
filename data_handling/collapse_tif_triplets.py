import os
import torch
import rasterio
import numpy as np
import torch.nn.functional as F


#####
"""
This code takes the 6000 raw .tif files, and combines them into a dataset of 2000 .tif files, each of 13 bands. Red is band 3, Green is band 2, Blue is band 1 (0-indexed)
The stored .tif files are NOT yet normalized. Needs normalization when loaded to a dataloader. Seems like dividing by 3000 is good!
"""


# ──────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────
input_dir = "/cluster/home/malovhoi/letmecooknow/dino/data_finetune/sentinel2/data_raw"
output_dir = "/cluster/home/malovhoi/letmecooknow/dino/data_finetune/sentinel2/data_msi"
os.makedirs(output_dir, exist_ok=True)

# ──────────────────────────────────────────────────────────────
# Functions
# ──────────────────────────────────────────────────────────────

def read_tif(file):
    with rasterio.open(file) as src:
        img = src.read()
    return torch.tensor(img.astype(np.float32))

def upsample(img, target_size):
    img = img.unsqueeze(0)  # [1, C, H, W]
    return F.interpolate(img, size=target_size, mode="bilinear", align_corners=False).squeeze(0)

def load_and_stack_tif(path_10m, path_20m, path_60m):
    img_10m = read_tif(path_10m)
    img_20m = read_tif(path_20m)
    img_60m = read_tif(path_60m)

    target_size = img_10m.shape[1:]
    img_20m_up = upsample(img_20m, target_size)
    img_60m_up = upsample(img_60m, target_size)

    stacked = torch.cat([img_10m, img_20m_up, img_60m_up], dim=0)

    reorder_idx = [
        10,  # B01
        2,   # B02
        1,   # B03
        0,   # B04
        4,   # B05
        5,   # B06
        6,   # B07
        3,   # B08
        7,   # B8A
        11,  # B09
        12,  # B10
        8,   # B11
        9    # B12
    ]

    return stacked[reorder_idx]

def save_stacked_tif(output_tensor, reference_path_10m, output_path):
    with rasterio.open(reference_path_10m) as ref:
        profile = ref.profile.copy()

    profile.update({
        "count": 13,
        "dtype": "float32",
        "driver": "GTiff",
        "compress": "lzw"
    })

    with rasterio.open(output_path, "w", **profile) as dst:
        dst.write(output_tensor.numpy())

# ──────────────────────────────────────────────────────────────
# Main Loop
# ──────────────────────────────────────────────────────────────

# Collect all filenames
all_files = sorted([f for f in os.listdir(input_dir) if f.endswith(".tif")])

# Check we have exactly 6000 tif files (3 per image)
assert len(all_files) == 6000, f"Expected 6000 .tif files, got {len(all_files)}"

# Loop over every triplet
for i in range(0, len(all_files), 3):
    triplet = all_files[i:i+3]
    
    # Match files by suffix
    try:
        stem = triplet[0].replace("_10m.tif", "")
        path_10m = os.path.join(input_dir, f"{stem}_10m.tif")
        path_20m = os.path.join(input_dir, f"{stem}_20m.tif")
        path_60m = os.path.join(input_dir, f"{stem}_60m.tif")
        output_path = os.path.join(output_dir, f"{stem}_MSI.tif")

        # Check that all three files exist
        if not (os.path.exists(path_10m) and os.path.exists(path_20m) and os.path.exists(path_60m)):
            print(f"⚠️ Skipping {stem}: one or more files missing.")
            continue

        # Process and save
        stacked_tensor = load_and_stack_tif(path_10m, path_20m, path_60m)
        save_stacked_tif(stacked_tensor, path_10m, output_path)
        print(f"✅ [{i//3 + 1:04d}/2000] Saved: {output_path}")

    except Exception as e:
        print(f"❌ Error on triplet starting at {triplet[0]}: {e}")