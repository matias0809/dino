import torch
import rasterio
import numpy as np
from specdetr.specdetrwrapper import SpecDetrWrapper
from mmengine import Config
from collections import OrderedDict

cfg = Config.fromfile("specdetr/specdetrconfig.py")

# 1. Path to your image
tif_path = "/cluster/home/malovhoi/letmecooknow/dino/data_extra/single_img_tif/0a2d943f-dfab-4156-84ae-96e8280d0552_MSI.tif"

# 2. Load .tif multispectral image (13 channels) – rasterio loads channels first
with rasterio.open(tif_path) as src:
    img_np = src.read()  # shape: (C, H, W)

# 3. Normalize (optional – depends on your DINO pretraining)
img_tensor = torch.from_numpy(img_np).float()
img_tensor = img_tensor / 3000.0  # if values are in 0–10000 range

# 4. Add batch dimension -> shape: (1, C, H, W)
img_tensor = img_tensor.unsqueeze(0)
print("Tensor shape: ", img_tensor.shape)

# 5. Instantiate the model
model = SpecDetrWrapper(**cfg['model'])

# 6. Load pre-trained DINO weights (optional)
ckpt = torch.load("/cluster/home/malovhoi/letmecooknow/dino/outputs/dino_outputs_embed64/checkpoint0100.pth", map_location="cpu")

# Extract student weights (MultiCropWrapper)
student_state = ckpt["student"]

# Extract only the backbone (which is your SpecDetrWrapper)
backbone_state = OrderedDict()
for k, v in student_state.items():
    if k.startswith("module.backbone."):
        backbone_state[k.replace("module.backbone.", "")] = v

missing, unexpected = model.load_state_dict(backbone_state, strict=False)
print("Done loading")
# 7. Send to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
img_tensor = img_tensor.to(device)

# 8. Forward pass
model.eval()
with torch.no_grad():
    output = model(img_tensor)

# 9. Print output shape
print("Output shape:", output.shape)  # Expecting (1, embed_dim)