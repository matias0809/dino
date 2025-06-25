""" import torch
from torchgeo.datasets import Sentinel2
from torchgeo.samplers import GridGeoSampler

from tqdm import tqdm
from pathlib import Path

This script should be used to make a .pt dataset from .SAFE files of Sentinel-2 images.
It will save each patch as a separate .pt file in the specified directory.
The first version was "load_patches_to_ram.py", which loaded all patches into one pt file. That killed RAM when trying to load the dataset!



dataset_raw = Sentinel2(paths="/cluster/home/malovhoi/letmecooknow/dino/data_safe", res=10, crs="EPSG:3857")
geosampler = GridGeoSampler(dataset=dataset_raw, size=336, stride=336)
patches = list(geosampler)

save_dir = Path("/cluster/home/malovhoi/letmecooknow/dino/data_pt_336")
save_dir.mkdir(parents=True, exist_ok=True)

for i, bb in tqdm(enumerate(patches), total=len(patches)):
    sample = dataset_raw[bb]["image"] / 10000  # Normalize
    torch.save(sample, save_dir / f"patch_{i:06d}.pt") """

from torchgeo.datasets import Sentinel2
from torchgeo.samplers import GridGeoSampler
from pathlib import Path
from tqdm import tqdm
import torch

data_root = Path("/cluster/home/malovhoi/letmecooknow/dino/data_safe")
save_dir = Path("/cluster/home/malovhoi/letmecooknow/dino/data_pt_336")
save_dir.mkdir(parents=True, exist_ok=True)

patch_counter = 0
tile_counter = 0
tile_patch_counts = {}

all_tile_dirs = sorted([p for p in data_root.iterdir() if p.is_dir()])

for idx, tile_dir in enumerate(tqdm(all_tile_dirs), start=1):
    try:
        dataset = Sentinel2(paths=str(tile_dir), res=10)
        sampler = GridGeoSampler(dataset=dataset, size=336, stride=336)
        patch_count = len(sampler)

        print(f"[{idx}/{len(all_tile_dirs)}] Tile: {tile_dir.name} → {patch_count} patches")

        tile_patch_counts[tile_dir.name] = patch_count
        tile_counter += 1

        for i, bb in enumerate(sampler):
            sample = dataset[bb]["image"] / 10000  # Normalize reflectance
            torch.save(sample, save_dir / f"patch_{patch_counter:06d}.pt")
            patch_counter += 1

    except Exception as e:
        print(f"⚠️ Skipping {tile_dir.name} due to error: {e}")

# ✅ Summary
print("\n--- Summary ---")
print(f"Total tiles processed: {tile_counter}")
print(f"Total patches saved: {patch_counter}\n")

for tile, count in tile_patch_counts.items():
    print(f"{tile}: {count} patches")