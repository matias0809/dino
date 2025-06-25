

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
import rasterio
import numpy as np
from tqdm import tqdm
import fiftyone as fo

from sklearn.metrics import average_precision_score
from specdetr.specdetrwrapper import SpecDetrWrapper
from mmengine import Config
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2

import json
import tempfile
from collections import OrderedDict
# ---------------------------- CONFIG ----------------------------------

BATCH_SIZE = 16
NUM_EPOCHS = 200
LEARNING_RATE = 1e-5*4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DATASET_DIR = "/cluster/home/malovhoi/letmecooknow/dino/data_finetune/sentinel2"
TIFF_DIR = os.path.join(DATASET_DIR, "data_msi")
PNGS_DIR = os.path.join(DATASET_DIR, "data")

CHECKPOINT_DIR = "/cluster/home/malovhoi/letmecooknow/dino/outputs/detectionhead_downsample/embed256_8_aug"
LOG_FILE = os.path.join(CHECKPOINT_DIR, "training_log.txt")


# -------------------------- TRANSFORM ------------------------------

# Albumentations pipeline to diversify ship positions
transform = A.Compose([
    A.RandomResizedCrop(height=336, width=336, scale=(0.4, 1.0), ratio=(0.75, 1.33), p=1.0),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.ShiftScaleRotate(shift_limit=0.20, scale_limit=0.2, rotate_limit=30, border_mode=cv2.BORDER_REFLECT_101, p=1.0),
], bbox_params=A.BboxParams(format="albumentations", label_fields=["class_labels"], min_visibility=0.3))


# ------------------------ CUSTOM DATASET ------------------------------

class ShipDetectionDataset(Dataset):
    def __init__(self, fiftyone_dataset, tiff_dir, transform=None):
        self.samples = list(fiftyone_dataset)
        self.tiff_dir = tiff_dir
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        png_path = sample.filepath
        stem = os.path.basename(png_path).replace("_TCI.png", "")
        tiff_path = os.path.join(self.tiff_dir, f"{stem}_MSI.tif")

        with rasterio.open(tiff_path) as src:
            img = src.read()
        boxes = []
        labels = []
        raw_boxes = []
        for det in sample.gt_bounding_boxes.detections:
            x1, y1, w, h = det.bounding_box
            if det.label == "ship":
                x2 = min(x1+w, 1.0)
                y2 = min(y1+h, 1.0)
                raw_boxes.append([x1, y1, x2, y2])  # xyxy normalized
        if self.transform and raw_boxes:
            augmented = self.transform(image=img.transpose(1,2,0), bboxes=raw_boxes, class_labels=["ship" for _ in range(len(raw_boxes))])
            img = augmented["image"].transpose(2, 0, 1)  # Convert to CHW format
            if augmented["bboxes"]: 
                boxes = [list(t) for t in augmented["bboxes"]] ##VIDERE: Mekk så alle augs gjøres etter for loop. Deretter gjør loss kompatibel med x1y1x2y2. Deretter sjekk om metoden fungerer, med test scriptet. Bruk albumentations format!
                labels += [1 for _ in range(len(boxes))]  # ship = 1
        elif not self.transform and raw_boxes:
            boxes = raw_boxes
            labels += [1 for _ in range(len(boxes))]  # ship = 1

        img_tensor = torch.from_numpy(img).float() / 3000.0
        boxes = torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        labels = torch.tensor(labels, dtype=torch.long).reshape(-1)

        target = {"boxes": boxes, "labels": labels, "id": idx}
        return img_tensor, target

# ---------------------- COLLATE FN -----------------------------------

def custom_collate_fn(batch):
    images, targets = zip(*batch)
    images = torch.stack(images, dim=0)
    return images, list(targets)

# ---------------------- DETECTION HEAD -------------------------------

class DetectionHead(nn.Module):
    def __init__(self, embed_dim, num_classes=1):
        super().__init__()
        self.num_classes = num_classes
        self.shared = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(embed_dim, embed_dim, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(embed_dim, embed_dim, kernel_size=3, stride=2, padding=1),
            nn.ReLU()
        )
        self.class_head = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv2d(embed_dim, embed_dim, kernel_size=7, padding=3),
            nn.ReLU(),
            nn.Conv2d(embed_dim, num_classes, 1)
        )
        self.box_head = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim, kernel_size=5, padding=2),  # larger local context
            nn.ReLU(),
            nn.Conv2d(embed_dim, embed_dim, kernel_size=7, padding=3),  # even larger context
            nn.ReLU(),
            nn.Conv2d(embed_dim, 2, kernel_size=1),
            nn.Sigmoid()
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        B, N, C = x.shape
        H = W = int(N ** 0.5)
        x = x.transpose(1, 2).reshape(B, C, H, W)

        x = self.shared(x) ##FOR DOWNSAMPLING!
        logits = self.class_head(x)
        bboxes = self.box_head(x)
        return logits, bboxes

# ---------------------- GAUSSIAN UTILS -------------------------------

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

# ---------------------- LOSS FUNCTIONS -------------------------------


def modified_focal_loss(pred, gt, alpha=2, beta=4):
    pos_inds = gt.eq(1).float()
    neg_inds = gt.lt(1).float()

    pos_loss = -((1 - pred) ** alpha) * pos_inds * torch.log(pred.clamp(min=1e-6))
    neg_loss = -((1 - gt) ** beta) * (pred ** alpha) * neg_inds * torch.log((1 - pred).clamp(min=1e-6))

    num_pos = pos_inds.sum().clamp(min=1.0)
    num_neg = neg_inds.sum().clamp(min=1.0)
    loss = pos_loss.sum() / num_pos + neg_loss.sum() / num_neg

    return loss

class DetectionLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.reg_loss = nn.L1Loss()

    def forward(self, pred_logits, pred_boxes, targets):
        B, _, H, W = pred_logits.shape
        total_cls_loss = 0
        total_box_loss = 0
        valid_box_batches = 0

        for i in range(B):
            target = targets[i]
            tgt_boxes = target['boxes'].to(pred_logits.device)
            tgt_labels = target['labels'].to(pred_logits.device)

            heatmap = torch.zeros((1, H, W), device=pred_logits.device)
            if tgt_labels.numel() == 0:
                #print("\n\nNO SHIP")
                cls_loss = modified_focal_loss(torch.sigmoid(pred_logits[i]), heatmap)
                #print("CLS loss when no ships: ", cls_loss)
                total_cls_loss += cls_loss
                continue
            

            scaling = tgt_boxes.new_tensor([W, H])
            centers = tgt_boxes[:, :2] + (tgt_boxes[:, 2:] - tgt_boxes[:, :2]) / 2
            #centers = tgt_boxes[:, :2]
            centers = centers * scaling
            
            sizes = tgt_boxes[:, 2:] - tgt_boxes[:, :2]
            #sizes = tgt_boxes[:, 2:] * scaling
            sizes = sizes * scaling
            centers[:, 0] = centers[:, 0].clamp(0, W - 1)
            centers[:, 1] = centers[:, 1].clamp(0, H - 1)
            centers = centers.long()

            #print("HERE IS CENTERS: ", centers)
            #print("These are sizes: ", sizes)
            for c, s in zip(centers, sizes):
                draw_gaussian2(heatmap, c, s)
            #print("\n\nTHERE EXIST SHIP")
            cls_loss = modified_focal_loss(torch.sigmoid(pred_logits[i]), heatmap)
            pred_box = pred_boxes[i].permute(1, 2, 0)[centers[:,1], centers[:,0]]
            #print("predicted boxes: ", pred_box)
            #box_loss = self.reg_loss(pred_box, tgt_boxes[:, 2:].to(pred_box.device))
            box_loss = self.reg_loss(pred_box, (tgt_boxes[:, 2:] - tgt_boxes[:, :2]).to(pred_box.device))
            #print("CLS loss when ship: ", cls_loss)
            #print("Box loss when ship: ", box_loss)
            total_cls_loss += cls_loss
            total_box_loss += box_loss
            valid_box_batches += 1

        #if valid_batches == 0:
            #return None, None
        avg_box_loss = total_box_loss / valid_box_batches if valid_box_batches > 0 else torch.tensor(0.0, dtype=torch.float32, device=pred_logits.device)


        return total_cls_loss / B, avg_box_loss

# ---------------------- MAIN TRAINING LOOP ---------------------------

def train():
    print("🔁 Loading FiftyOne dataset...")
    dataset = fo.load_dataset("sentinel2")

    train_view = dataset.match_tags("train")
    val_view = dataset.match_tags("val")
    test_view = dataset.match_tags("test")

    # Convert to PyTorch datasets
    train_dataset = ShipDetectionDataset(train_view, TIFF_DIR, transform=transform)
    val_dataset   = ShipDetectionDataset(val_view, TIFF_DIR)
    test_dataset  = ShipDetectionDataset(test_view, TIFF_DIR)

    ###MÅ ENNÅ FIKSE COCO FORMAT PÅ EVAL DATA! Så dette er ikke helt riktig enda

    # Use them in your dataloaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, collate_fn=custom_collate_fn)
    val_loader   = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, collate_fn=custom_collate_fn)
    test_loader  = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, collate_fn=custom_collate_fn)


    print("🔧 Initializing model and head...")
    cfg = Config.fromfile("specdetr/specdetrconfig.py")
    backbone = SpecDetrWrapper(**cfg['model'])

    ckpt = torch.load("/cluster/home/malovhoi/letmecooknow/dino/outputs/dino_outputs_embed256/checkpoint0100.pth", map_location="cpu")
    student_state = ckpt["student"]
    
    backbone_state = OrderedDict()
    for k, v in student_state.items():
        if k.startswith("module.backbone."):
            backbone_state[k.replace("module.backbone.", "")] = v
    backbone.load_state_dict(backbone_state, strict=False)
    backbone.to(DEVICE)
    backbone.eval()

    head = DetectionHead(embed_dim=256).to(DEVICE)
    optimizer = torch.optim.AdamW(head.parameters(), lr=LEARNING_RATE)
    criterion = DetectionLoss()

    print("🚀 Starting training...")
    for epoch in range(NUM_EPOCHS):
        head.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}")
        epoch_cls_loss = 0
        epoch_box_loss = 0

        for imgs, targets in pbar:
            imgs = imgs.to(DEVICE)
            with torch.no_grad():
                feats = backbone(imgs)
                feats = feats[:, :-1]

            logits, boxes = head(feats)
            cls_loss, box_loss = criterion(logits, boxes, targets)


            loss = cls_loss + box_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_cls_loss += cls_loss.item()*BATCH_SIZE
            epoch_box_loss += box_loss.item()*BATCH_SIZE
            pbar.set_postfix({"ClsLoss": cls_loss.item(), "BoxLoss": box_loss.item()})

        with open(LOG_FILE, 'a') as f:
            f.write(json.dumps({
                "class_loss": epoch_cls_loss,
                "bbox_loss":  epoch_box_loss,
                "total_loss": epoch_cls_loss + epoch_box_loss,
                "epoch": epoch
            }) + "\n")

        print(f"✅ Epoch {epoch+1}: ClsLoss = {epoch_cls_loss:.4f}, BoxLoss = {epoch_box_loss:.4f}")

        if (epoch + 1) % 20 == 0:
            ckpt_path = os.path.join(CHECKPOINT_DIR, f"checkpoint{(epoch+1):04}.pth")
            torch.save(head.state_dict(), ckpt_path)

"""
        head.eval()
        image_id = 0
        coco_gt = {"info": {}, "licenses": [], "images": [], "annotations": [], "categories": [{"id": 1, "name": "ship"}]}
        coco_dt = []
        ann_id = 0
        with torch.no_grad():
            for imgs, targets in tqdm(test_loader, desc="loader"):
                imgs = imgs.to(DEVICE)
                print("hola new batch")
                feats = backbone(imgs)[:, :-1]
                logits, boxes = head(feats)
                probs = torch.sigmoid(logits).squeeze(1)
                B, H, W = probs.shape
                boxes = boxes.permute(0, 2, 3, 1)  # BxHxWx4
"""
"""
                for b in range(B):
                    gt_boxes = targets[b]['boxes']
                    coco_gt["images"].append({"id": image_id})
                    for box in gt_boxes:
                        x1, y1, x2, y2 = box.tolist()
                        coco_gt["annotations"].append({
                            "id": ann_id,
                            "image_id": image_id,
                            "category_id": 1,
                            "bbox": [x1 * W, y1 * H, (x2 - x1) * W, (y2 - y1) * H],
                            "area": (x2 - x1) * (y2 - y1) * W * H,
                            "iscrowd": 0
                        })
                        ann_id += 1
                    for i in range(H):
                        for j in range(W):
                            score = probs[b, i, j].item()
                            if score > 0.3:
                                px, py = j, i
                                pred_box = boxes[b, i, j].cpu()
                                cx, cy, bw, bh = px / W, py / H, pred_box[2].item(), pred_box[3].item()
                                coco_dt.append({
                                    "image_id": image_id,
                                    "category_id": 1,
                                    "bbox": [cx * W - bw * W / 2, cy * H - bh * H / 2, bw * W, bh * H],
                                    "score": score
                                })
                    if image_id % 100 == 0:
                        print(f"Processed {image_id} images...")
                    image_id += 1"""
"""
                threshold = 0.3
                for b in range(B):
                    print("hola new image")
                    gt_boxes = targets[b]['boxes']
                    coco_gt["images"].append({"id": image_id})
                    
                    for box in gt_boxes:
                        x1, y1, x2, y2 = box.tolist()
                        coco_gt["annotations"].append({
                            "id": ann_id,
                            "image_id": image_id,
                            "category_id": 1,
                            "bbox": [x1 * W, y1 * H, (x2 - x1) * W, (y2 - y1) * H],
                            "area": (x2 - x1) * (y2 - y1) * W * H,
                            "iscrowd": 0
                        })
                        ann_id += 1

                    # 👉 Vectorized part
                    score_map = probs[b]
                    high_score_indices = (score_map > threshold).nonzero(as_tuple=False)  # shape: (N, 2)
                    if high_score_indices.numel() == 0:
                        image_id += 1
                        continue
                    pred_boxes = boxes[b][high_score_indices[:, 0], high_score_indices[:, 1]]
                    print("before loop")
                    for idx, (py, px) in enumerate(high_score_indices):
                        score = score_map[py, px].item()
                        cx, cy = px.item() / W, py.item() / H
                        bw, bh = pred_boxes[idx][2].item(), pred_boxes[idx][3].item()
                        coco_dt.append({
                            "image_id": image_id,
                            "category_id": 1,
                            "bbox": [cx * W - bw * W / 2, cy * H - bh * H / 2, bw * W, bh * H],
                            "score": score
                        })

                    image_id += 1

        with tempfile.NamedTemporaryFile(mode='w+', delete=False) as gt_file, tempfile.NamedTemporaryFile(mode='w+', delete=False) as dt_file:
            json.dump(coco_gt, gt_file)
            json.dump(coco_dt, dt_file)
            gt_file.flush()
            dt_file.flush()
            coco_gt_obj = COCO(gt_file.name)
            coco_dt_obj = coco_gt_obj.loadRes(dt_file.name)
            coco_eval = COCOeval(coco_gt_obj, coco_dt_obj, 'bbox')
            coco_eval.evaluate()
            coco_eval.accumulate()
            coco_eval.summarize()
"""

if __name__ == "__main__":
    train()