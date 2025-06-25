# specdetr_ship_detector.py

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import rasterio
import numpy as np
from tqdm import tqdm
import fiftyone as fo

from specdetr.specdetrwrapper import SpecDetrWrapper
from mmengine import Config

# ---------------------------- CONFIG ----------------------------------

BATCH_SIZE = 4
NUM_EPOCHS = 10
LEARNING_RATE = 1e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DATASET_DIR = "/cluster/home/malovhoi/letmecooknow/dino/data_finetune/sentinel2"
TIFF_DIR = os.path.join(DATASET_DIR, "data_msi")
PNGS_DIR = os.path.join(DATASET_DIR, "data")

# ------------------------ CUSTOM DATASET ------------------------------

class ShipDetectionDataset(Dataset):
    def __init__(self, fiftyone_dataset, tiff_dir):
        self.samples = list(fiftyone_dataset)
        self.tiff_dir = tiff_dir

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        png_path = sample.filepath
        stem = os.path.basename(png_path).replace("_TCI.png", "")
        tiff_path = os.path.join(self.tiff_dir, f"{stem}_MSI.tif")

        with rasterio.open(tiff_path) as src:
            img = src.read()
        img_tensor = torch.from_numpy(img).float() / 3000.0

        boxes = []
        labels = []

        for det in sample.gt_bounding_boxes.detections:
            x, y, w, h = det.bounding_box
            if det.label == "ship":
                boxes.append([x, y, x + w, y + h])  # xyxy normalized
                labels.append(1)  # ship = 1

        target = {
            "boxes": torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4),
            "labels": torch.tensor(labels, dtype=torch.long).reshape(-1),
        }
        #print("Boxes shape: ", target["boxes"].shape, "Labels shape: ", target["labels"].shape)

        return img_tensor, target


# ---------------------- COLLATE FN -----------------------------------

def custom_collate_fn(batch):
    images, targets = zip(*batch)
    images = torch.stack(images, dim=0)
    return images, list(targets)


# ---------------------- DETECTION HEAD -------------------------------

class DetectionHead(nn.Module):
    def __init__(self, embed_dim, num_classes=2):
        super().__init__()
        self.num_classes = num_classes
        self.class_head = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(embed_dim, num_classes, 1)
        )
        self.box_head = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(embed_dim, 4, 1),
            nn.Sigmoid()  # box coordinates between 0 and 1
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        # x: [B, N, C] --> reshape to [B, C, H, W]
        B, N, C = x.shape
        H = W = int(N ** 0.5)
        x = x.transpose(1, 2).reshape(B, C, H, W)
        logits = self.class_head(x)  # [B, num_classes, H, W]
        bboxes = self.box_head(x)   # [B, 4, H, W]
        return logits, bboxes

# ---------------------- LOSS FUNCTIONS -------------------------------

class DetectionLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.cls_loss = nn.CrossEntropyLoss()
        self.reg_loss = nn.L1Loss()

    def forward(self, pred_logits, pred_boxes, targets):
        B, _, H, W = pred_logits.shape
        total_cls_loss = 0
        total_box_loss = 0
        valid_batches = 0
        
        for i in range(B):
            target = targets[i]
            tgt_boxes = target['boxes']
            tgt_labels = target['labels']

            if tgt_labels.numel() == 0:
                continue

            # Center of boxes --> pixel grid
            centers = tgt_boxes[:, :2] + (tgt_boxes[:, 2:] - tgt_boxes[:, :2]) / 2
            centers = (centers * H).long().clamp(0, H-1)

            pred_cls = pred_logits[i].permute(1, 2, 0)[centers[:,1], centers[:,0]]
            pred_box = pred_boxes[i].permute(1, 2, 0)[centers[:,1], centers[:,0]]

            total_cls_loss += self.cls_loss(pred_cls, tgt_labels.to(pred_cls.device))
            total_box_loss += self.reg_loss(pred_box, tgt_boxes.to(pred_box.device))
            
            valid_batches += 1
        
        if valid_batches == 0:
            return None, None

        return total_cls_loss / valid_batches, total_box_loss / valid_batches

        

# ---------------------- MAIN TRAINING LOOP ---------------------------

def train():
    print("🔁 Loading FiftyOne dataset...")
    dataset = fo.load_dataset("sentinel2")
    torch_dataset = ShipDetectionDataset(dataset, TIFF_DIR)
    loader = DataLoader(torch_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, collate_fn=custom_collate_fn)

    print("🔧 Initializing model and head...")
    cfg = Config.fromfile("specdetr/specdetrconfig.py")
    backbone = SpecDetrWrapper(**cfg['model'])

    ckpt = torch.load("/cluster/home/malovhoi/letmecooknow/dino/outputs/dino_outputs_embed64/checkpoint0100.pth", map_location="cpu")
    student_state = ckpt["student"]
    from collections import OrderedDict
    backbone_state = OrderedDict()
    for k, v in student_state.items():
        if k.startswith("module.backbone."):
            backbone_state[k.replace("module.backbone.", "")] = v
    backbone.load_state_dict(backbone_state, strict=False)
    backbone.to(DEVICE)
    backbone.eval()  # freeze backbone

    head = DetectionHead(embed_dim=64).to(DEVICE)
    optimizer = torch.optim.AdamW(head.parameters(), lr=LEARNING_RATE)
    criterion = DetectionLoss()

    print("🚀 Starting training...")
    for epoch in range(NUM_EPOCHS):
        head.train()
        pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}")
        epoch_cls_loss = 0
        epoch_box_loss = 0

        for imgs, targets in pbar:
            imgs = imgs.to(DEVICE)
            with torch.no_grad():
                feats = backbone(imgs)
                feats = feats[:, :-1]  # remove global token

            logits, boxes = head(feats)
            cls_loss, box_loss = criterion(logits, boxes, targets)

            if cls_loss is None or box_loss is None:
                continue

            loss = cls_loss + box_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_cls_loss += cls_loss.item()
            epoch_box_loss += box_loss.item()
            pbar.set_postfix({"ClsLoss": cls_loss.item(), "BoxLoss": box_loss.item()})

        print(f"✅ Epoch {epoch+1}: ClsLoss = {epoch_cls_loss:.4f}, BoxLoss = {epoch_box_loss:.4f}")

if __name__ == "__main__":
    train()