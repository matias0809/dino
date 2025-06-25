import torch
import json
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from DetectionHead import (
    DetectionHead, SpecDetrWrapper, ShipDetectionDataset, custom_collate_fn,
    TIFF_DIR, DEVICE, Config, fo, DataLoader
)
from collections import OrderedDict
from tqdm import tqdm
import torchvision.ops as ops
import numpy as np



CHECKPOINT_PATH ="/cluster/home/malovhoi/letmecooknow/dino/outputs/detectionhead_downsample/embed64_8_aug/checkpoint0200.pth"
ANNOTATION_PATH = "/cluster/home/malovhoi/letmecooknow/dino/data_finetune/sentinel2/coco_annotations/instances_val.json"
PREDICTIONS_PATH = "predictions.json"

# Load dataset
dataset = fo.load_dataset("sentinel2")
val_view = dataset.match_tags("val")
val_dataset = ShipDetectionDataset(val_view, TIFF_DIR)
val_loader = DataLoader(val_dataset, batch_size=16, collate_fn=custom_collate_fn, shuffle=False)

# Load model
cfg = Config.fromfile("specdetr/specdetrconfig.py")
backbone = SpecDetrWrapper(**cfg['model'])

ckpt = torch.load("/cluster/home/malovhoi/letmecooknow/dino/outputs/dino_outputs_embed64/checkpoint0100.pth", map_location="cpu")
student_state = ckpt["student"]

backbone_state = OrderedDict()
for k, v in student_state.items():
    if k.startswith("module.backbone."):
        backbone_state[k.replace("module.backbone.", "")] = v
backbone.load_state_dict(backbone_state, strict=False)
backbone.to(DEVICE)
backbone.eval()

head = DetectionHead(embed_dim=64).to(DEVICE)
head.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=DEVICE))
head.eval()

# Run inference
preds = []
image_id=1
id_map = {s.filepath: s.id for s in val_view}
with torch.no_grad():
    for imgs, targets in tqdm(val_loader):
        imgs = imgs.to(DEVICE)
        feats = backbone(imgs)[:, :-1]
        logits, box_preds = head(feats)

        B, _, H, W = logits.shape
        for i in range(B):
            prob = torch.sigmoid(logits[i, 0])
            mask = prob > 0.9
            coords = mask.nonzero(as_tuple=False)
            scores = prob[coords[:, 0], coords[:, 1]]
            wh = box_preds[i, :, coords[:, 0], coords[:, 1]].T
            wh_pixel = wh * torch.tensor([W, H], device=DEVICE)

            boxes = []
            for (y, x), (w, h) in zip(coords, wh_pixel):
                cx, cy = float(x), float(y)
                x1 = cx - w.item() / 2
                y1 = cy - h.item() / 2
                x2 = cx + w.item() / 2
                y2 = cy + h.item() / 2
                boxes.append([x1, y1, x2, y2])
            if boxes:
                boxes = torch.tensor(boxes, device=DEVICE)
                scores = scores.to(DEVICE)

                keep_idxs = ops.nms(boxes, scores, iou_threshold=0.1)

                for idx in keep_idxs:
                    x1, y1, x2, y2 = boxes[idx]
                    w = x2 - x1
                    h = y2 - y1
                    preds.append({
                        "image_id": image_id,
                        "category_id": 1,
                        "bbox": [x1.item()*8, y1.item()*8, w.item()*8, h.item()*8],
                        "score": scores[idx].item()
                    })
            image_id += 1
# Save predictions
with open(PREDICTIONS_PATH, "w") as f:
    json.dump(preds, f)

# Evaluate
coco_gt = COCO(ANNOTATION_PATH)
coco_dt = coco_gt.loadRes(PREDICTIONS_PATH)
evaluator = COCOeval(coco_gt, coco_dt, iouType="bbox")
evaluator.params.iouThrs = [0.1, 0.2, 0.3, 0.4, 0.5]  # 👈 Only evaluate at IoU = 0.3
#evaluator.params.maxDets = [1, 10, 1000] #forsøk på å få precision til å gi mening, men hjalp ikke

evaluator.evaluate()
evaluator.accumulate()
evaluator.summarize()

from faster_coco_eval import COCO as FasterCOCO
from faster_coco_eval.extra import PreviewResults
from faster_coco_eval.extra import Curves

# Load annotations and predictions directly from the JSON paths you already used
gt_data = FasterCOCO.load_json(ANNOTATION_PATH)
dt_data = FasterCOCO.load_json(PREDICTIONS_PATH)

# Initialize FasterCOCO objects
faster_gt = FasterCOCO(gt_data)
faster_dt = faster_gt.loadRes(dt_data)

cur = Curves(faster_gt, faster_dt, iou_tresh=0.3, iouType="bbox", useCats=False)
cur.plot_pre_rec()
cur.plot_f1_confidence()

# Set up PreviewResults for detection confusion analysis
results = PreviewResults(
    faster_gt,
    faster_dt,
    iou_tresh=0.1,
    iouType="bbox",
    useCats=False
)
results.display_matrix()
"""
results = PreviewResults(
    faster_gt,
    faster_dt,
    iou_tresh=0.3,
    iouType="bbox",
    useCats=True
)

results.display_tp_fp_fn(data_folder="/cluster/home/malovhoi/letmecooknow/dino/data_finetune/sentinel2/data", image_ids=[4])
"""

"""
Right now: Things make sense. Only thing is that if we lower the threshold for what we consider true positive, our prediction list becomes around 20000 predictions (with threshold 0.1). 
However, the COCO precision evaluation gives a relatively high precision. That should not be the case with so many false positives.
Dont know why this happens. 

Otherwise, lowering the threshold gives a smoother precision-recall functions. Thats because a lower threshold opens up for more true positives. 
That gives a higher recall, so we get points in the curve of higher recall values. 

Check out this link: https://github.com/cocodataset/cocoapi/issues/56 
I think it might explain the precision issue. AP isnt precision, but rather area under precision-recall curve. Can get some un-intuituve effects from that.
"""