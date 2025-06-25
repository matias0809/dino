import fiftyone as fo
import torch
import os
import numpy as np
import matplotlib.pyplot as plt
import PIL.Image as Image
"""
SAVE_DIR = "/cluster/home/malovhoi/letmecooknow/dino/data_extra/single_img_tif"
os.makedirs(SAVE_DIR, exist_ok=True)

def draw_gaussian(heatmap, center, sigma=1):
    x, y = center
    H, W = heatmap.shape[-2:]

    tmp_size = int(3 * sigma)
    mu_x = int(x)
    mu_y = int(y)

    x0 = max(mu_x - tmp_size, 0)
    x1 = min(mu_x + tmp_size + 1, W)
    y0 = max(mu_y - tmp_size, 0)
    y1 = min(mu_y + tmp_size + 1, H)

    x_range = torch.arange(x0, x1, device=heatmap.device)
    y_range = torch.arange(y0, y1, device=heatmap.device)
    yy, xx = torch.meshgrid(y_range, x_range, indexing='ij')
    gaussian = torch.exp(-((xx - x)**2 + (yy - y)**2) / (2 * sigma**2))

    heatmap[0, y0:y1, x0:x1] = torch.maximum(heatmap[0, y0:y1, x0:x1], gaussian)

# --- Load dataset
dataset = fo.load_dataset("sentinel2")
sample = list(dataset)[0]  # Take first sample

# --- Get image shape
img = Image.open(sample.filepath)
W, H = img.size  # Note: PIL gives (W, H)

# --- Make empty heatmap
heatmap = torch.zeros((1, H, W))

# --- Draw Gaussian on each ship
for det in sample.gt_bounding_boxes.detections:
    if det.label == "ship":
        x, y, w, h = det.bounding_box
        cx = (x + w / 2) * W
        cy = (y + h / 2) * H
        draw_gaussian(heatmap, (cx, cy), sigma=2)

# --- Save heatmap
heatmap_np = heatmap.squeeze(0).cpu().numpy()
heatmap_img = (heatmap_np * 255).astype(np.uint8)
save_path = os.path.join(SAVE_DIR, "heatmap_sample0.png")
plt.imsave(save_path, heatmap_img, cmap="hot")
print(f"✅ Saved heatmap to {save_path}")"""

import fiftyone as fo
import torch
import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

SAVE_DIR = "/cluster/home/malovhoi/letmecooknow/dino/data_extra/single_img_tif"
os.makedirs(SAVE_DIR, exist_ok=True)

def draw_gaussian2(heatmap, center, bbox_size, min_sigma=1):
    x, y = center
    w, h = bbox_size
    H, W = heatmap.shape[-2:]

    sigma = max(min_sigma, 0.15 * ((w + h) / 2))

    tmp_size = int(3 * sigma)
    mu_x = int(x)
    mu_y = int(y)

    x0 = max(mu_x - tmp_size, 0)
    x1 = min(mu_x + tmp_size + 1, W)
    y0 = max(mu_y - tmp_size, 0)
    y1 = min(mu_y + tmp_size + 1, H)

    x_range = torch.arange(x0, x1, device=heatmap.device)
    y_range = torch.arange(y0, y1, device=heatmap.device)
    yy, xx = torch.meshgrid(y_range, x_range, indexing='ij')
    gaussian = torch.exp(-((xx - x)**2 + (yy - y)**2) / (2 * sigma**2))

    heatmap[0, y0:y1, x0:x1] = torch.maximum(heatmap[0, y0:y1, x0:x1], gaussian)

# --- Load dataset
dataset = fo.load_dataset("sentinel2")
sample = list(dataset)[0]

# --- Load RGB image
img = Image.open(sample.filepath).convert("RGB")
img_np = np.array(img)
H, W = img_np.shape[:2]

# --- Create heatmap
heatmap = torch.zeros((1, H, W))
for det in sample.gt_bounding_boxes.detections:
    if det.label == "ship":
        x, y, w, h = det.bounding_box
        cx = (x + w / 2) * W
        cy = (y + h / 2) * H
        bw = w * W
        bh = h * H
        draw_gaussian2(heatmap, (cx, cy), (bw, bh))

# --- Normalize heatmap for visualization
heatmap_np = heatmap.squeeze(0).cpu().numpy()
heatmap_np = (heatmap_np - heatmap_np.min()) / (heatmap_np.max() - heatmap_np.min() + 1e-8)

# --- Overlay and save
fig, ax = plt.subplots()
ax.imshow(img_np)
ax.imshow(heatmap_np, cmap="hot", alpha=0.5)
ax.axis("off")

save_path = os.path.join(SAVE_DIR, "overlay_rgb_heatmap_sample0.png")
plt.savefig(save_path, bbox_inches="tight", pad_inches=0)
plt.close()
print(f"✅ Saved RGB + heatmap overlay to {save_path}")