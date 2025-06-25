import os
import torch
import rasterio
import numpy as np
from PIL import Image
from tqdm import tqdm
from torchvision.models import vit_b_16, ViT_B_16_Weights
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from mmengine import Config
from specdetr.specdetrwrapper import SpecDetrWrapper
from collections import OrderedDict
import fiftyone as fo


"""
This file performs KNN and PCA on SpecDETR and a pre-trained ViT. REMEMBER to pool the output embeddings of SpecDETR encoder if this script is going to get used.
PCA plots are saved to outputs/pca_outputs

"""

###############33


# ─────────────────────────────────────────────────────────────────────────
# Step 1: Load dataset and labels from FiftyOne
# ─────────────────────────────────────────────────────────────────────────
dataset = fo.load_dataset("sentinel2")

samples_png = []
labels = []

for sample in dataset:
    classifications = sample.gt_classifications.classifications
    has_ship = any(cls.label == "ship" for cls in classifications)
    labels.append(1 if has_ship else 0)
    samples_png.append(sample.filepath)  # PNG filepaths

# ─────────────────────────────────────────────────────────────────────────
# Step 2: Create matching list of MSI (.tif) paths
# ─────────────────────────────────────────────────────────────────────────
def png_to_tif(png_path):
    fname = os.path.basename(png_path).replace("_TCI.png", "_MSI.tif")
    return os.path.join("/cluster/home/malovhoi/letmecooknow/dino/data_finetune/sentinel2/data_msi", fname)

samples_tif = [png_to_tif(p) for p in samples_png]

# ─────────────────────────────────────────────────────────────────────────
# Step 3: Dataset class for both RGB and MSI modes
# ─────────────────────────────────────────────────────────────────────────
class SentinelDataset(Dataset):
    def __init__(self, paths, mode="rgb", transform=None):
        self.paths = paths
        self.mode = mode  # "rgb" or "msi"
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        if self.mode == "rgb":
            img = Image.open(path).convert("RGB")
            return self.transform(img)
        elif self.mode == "msi":
            with rasterio.open(path) as src:
                img = src.read().astype(np.float32) / 3000.0
            return torch.from_numpy(img)
        else:
            raise ValueError("Mode must be 'rgb' or 'msi'")

# ─────────────────────────────────────────────────────────────────────────
# Step 4: Feature extraction
# ─────────────────────────────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- ViT setup
weights = ViT_B_16_Weights.DEFAULT
vit_transform = weights.transforms()
vit_model = vit_b_16(weights=weights).to(device)
vit_model.eval()

def extract_vit_features(paths):
    dataset = SentinelDataset(paths, mode="rgb", transform=vit_transform)
    loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=4)

    feats = []
    with torch.no_grad():
        for batch in tqdm(loader, desc="Extracting ViT features"):
            batch = batch.to(device)
            output = vit_model(batch)
            feats.append(output.cpu())
    return torch.cat(feats).numpy()

# --- SpecDETR setup
cfg = Config.fromfile("specdetr/specdetrconfig.py")
specdetr_model = SpecDetrWrapper(**cfg['model'])

# Load DINO student backbone only
ckpt = torch.load("/cluster/home/malovhoi/letmecooknow/dino/outputs/dino_outputs_embed256/checkpoint0100.pth", map_location="cpu")
student_state = ckpt["student"]
backbone_state = OrderedDict()
for k, v in student_state.items():
    if k.startswith("module.backbone."):
        backbone_state[k.replace("module.backbone.", "")] = v
specdetr_model.load_state_dict(backbone_state, strict=False)
specdetr_model.to(device).eval()

def extract_specdetr_features(paths):
    dataset = SentinelDataset(paths, mode="msi")
    loader = DataLoader(dataset, batch_size=16, shuffle=False, num_workers=2)

    feats = []
    with torch.no_grad():
        for batch in tqdm(loader, desc="Extracting SpecDETR features"):
            batch = batch.to(device)
            out = specdetr_model(batch)
            feats.append(out.cpu())
    return torch.cat(feats).numpy()

# ─────────────────────────────────────────────────────────────────────────
# Step 5: kNN + PCA utilities
# ─────────────────────────────────────────────────────────────────────────
def run_knn_and_pca(features, labels, name):
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"✅ kNN accuracy ({name}): {acc:.4f}")

    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(features)

    colors = ['red' if y == 1 else 'blue' for y in labels]
    plt.figure(figsize=(8, 6))
    plt.scatter(X_pca[:, 0], X_pca[:, 1], c=colors, alpha=0.6, edgecolor='k', s=40)
    plt.title(f"PCA of {name} Features")
    plt.xlabel("PC 1")
    plt.ylabel("PC 2")
    plt.grid(True)
    plt.tight_layout()

    os.makedirs("/cluster/home/malovhoi/letmecooknow/dino/outputs/pca_outputs", exist_ok=True)
    plot_path = f"/cluster/home/malovhoi/letmecooknow/dino/outputs/pca_outputs/test/pca_{name}.png"
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"✅ PCA plot saved: {plot_path}")

# ─────────────────────────────────────────────────────────────────────────
# Step 6: Run everything
# ─────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("🚀 Running ViT pipeline...")
    vit_feats = extract_vit_features(samples_png)
    run_knn_and_pca(vit_feats, labels, "vit")

    print("\n🚀 Running SpecDETR pipeline...")
    specdetr_feats = extract_specdetr_features(samples_tif)
    run_knn_and_pca(specdetr_feats, labels, "specdetr")

    