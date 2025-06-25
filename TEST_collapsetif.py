import rasterio
from rasterio.transform import from_origin
import torch
import torch.nn.functional as F
import numpy as np
import os

def load_and_stack_tif(path_10m, path_20m, path_60m):
    def read_tif(file):
        with rasterio.open(file) as src:
            img = src.read()  # shape: (bands, height, width)
        return torch.tensor(img.astype(np.float32), dtype=torch.float32)

    img_10m = read_tif(path_10m)  # [4, H, W]
    img_20m = read_tif(path_20m)  # [6, H//2, W//2]
    img_60m = read_tif(path_60m)  # [3, H//6, W//6]

    def upsample(img, target_size):
        img = img.unsqueeze(0)  # [1, C, H, W]
        img_up = F.interpolate(img, size=target_size, mode='bilinear', align_corners=False)
        return img_up.squeeze(0)  # [C, H, W]

    target_size = img_10m.shape[1:]  # (H, W)
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

    reordered = stacked[reorder_idx]
    return reordered

def save_stacked_tif(output_tensor, reference_path_10m, output_path):
    # output_tensor shape: [13, H, W]
    with rasterio.open(reference_path_10m) as ref:
        profile = ref.profile.copy()

    profile.update({
        'count': 13,
        'dtype': 'float32',
        'driver': 'GTiff',
        'compress': 'lzw'
    })

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with rasterio.open(output_path, 'w', **profile) as dst:
        dst.write(output_tensor.numpy())

    print(f"✅ Saved stacked image to: {output_path}")

# ─── Input paths ─────────────────────────────────────────────────────────────

base = "/cluster/home/malovhoi/letmecooknow/dino/data_finetune/sentinel2/data_msi/"
stem = "0a2d943f-dfab-4156-84ae-96e8280d0552"

path_10m = os.path.join(base, f"{stem}_10m.tif")
path_20m = os.path.join(base, f"{stem}_20m.tif")
path_60m = os.path.join(base, f"{stem}_60m.tif")

output_path = f"/cluster/home/malovhoi/letmecooknow/dino/data_extra/single_img_tif/{stem}_MSI.tif"

# ─── Combine and Save ────────────────────────────────────────────────────────

stacked_tensor = load_and_stack_tif(path_10m, path_20m, path_60m)
save_stacked_tif(stacked_tensor, path_10m, output_path)