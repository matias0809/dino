import rasterio
import numpy as np
import matplotlib.pyplot as plt
import os

# Path to your upsampled 13-band .tif
tif_path = "/cluster/home/malovhoi/letmecooknow/dino/data_finetune/sentinel2/data_msi/0a2d943f-dfab-4156-84ae-96e8280d0552_MSI.tif"

# Load the .tif
with rasterio.open(tif_path) as src:
    img = src.read()  # shape: (13, H, W)

# Extract B04 (R), B03 (G), B02 (B)
# Indices: B04 = 3, B03 = 2, B02 = 1
rgb = img[[3, 2, 1], :, :].astype(np.float32)

# Normalize
rgb /= 3000.0
rgb = np.clip(rgb, 0, 1)

# Transpose to HWC for saving
rgb = np.transpose(rgb, (1, 2, 0))

# Output path (same folder, same stem)
output_path = tif_path.replace("_MSI.tif", "_RGBpreview.png")

# Save with matplotlib
plt.imsave(output_path, rgb)

print(f"✅ RGB preview saved to: {output_path}")