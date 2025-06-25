from torchvision.models import vit_b_16, ViT_B_16_Weights
from torchvision import transforms
import torch
from torch.utils.data import DataLoader, Dataset
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import fiftyone as fo
import fiftyone.zoo as foz
from PIL import Image

# --- Step 1: Load FiftyOne dataset ---
dataset = fo.load_dataset("sentinel2")

# --- Step 2: Assign labels (1 if at least one ship, else 0) ---
samples = []
labels = []

for sample in dataset:
    classifications = sample.gt_classifications.classifications
    has_ship = any(cls.label == "ship" for cls in classifications)
    labels.append(1 if has_ship else 0)
    samples.append(sample.filepath)

# --- Step 3: Define image transform and model ---
device = "cuda" if torch.cuda.is_available() else "cpu"

weights = ViT_B_16_Weights.DEFAULT
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])
transform=weights.transforms()
model = vit_b_16(weights=weights).to(device)
model.eval()

# --- Step 4: Create PyTorch dataset ---
class Sentinel2Dataset(Dataset):
    def __init__(self, filepaths, transform):
        self.filepaths = filepaths
        self.transform = transform

    def __len__(self):
        return len(self.filepaths)

    def __getitem__(self, idx):
        img = Image.open(self.filepaths[idx]).convert("RGB")
        return self.transform(img)

dataset_torch = Sentinel2Dataset(samples, transform)
loader = DataLoader(dataset_torch, batch_size=32, shuffle=False, num_workers=4)

# --- Step 5: Extract features with ViT ---
features = []

with torch.no_grad():
    for batch in loader:
        batch = batch.to(device)
        feats = model._modules["heads"].register_forward_hook(lambda m, i, o: o)  # skip classification head
        out = model(batch)
        features.append(out.cpu())

features = torch.cat(features).numpy()

# --- Step 6: kNN classification (train/test split) ---
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

y_pred = knn.predict(X_test)
acc = accuracy_score(y_test, y_pred)

print(f"✅ kNN classification accuracy: {acc:.4f}")

"""
labels = []

for sample in tqdm(dataset):
    classifications = sample.gt_classifications.classifications
    has_ship = any(cls.label == "ship" for cls in classifications)
    labels.append(1 if has_ship else 0)
for i in range(30):
    print(labels[i])"""

import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import os

# --- Step 7: PCA projection to 2D ---
pca = PCA(n_components=2)
X_pca = pca.fit_transform(features)

# --- Step 8: Plotting ---
os.makedirs("/cluster/home/malovhoi/letmecooknow/dino/outputs/pca_outputs/test", exist_ok=True)
output_path = "/cluster/home/malovhoi/letmecooknow/dino/outputs/pca_outputs/test/pca_knn.png"

colors = ['red' if label == 1 else 'blue' for label in labels]

plt.figure(figsize=(8, 6))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=colors, alpha=0.6, edgecolor='k', s=40)
plt.title("PCA of ViT Features: Red = Ship, Blue = No Ship")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.grid(True)
plt.tight_layout()
plt.savefig(output_path, dpi=300)
plt.close()

print(f"✅ PCA scatter plot saved to {output_path}")

print(dataset.first().filepath)
for i in range(30):
    print(samples[i])