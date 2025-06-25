from DetectionHead import ShipDetectionDataset, DetectionHead
from specdetr.specdetrwrapper import SpecDetrWrapper
import os
import torch
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import fiftyone as fo
import rasterio
from torch.utils.data import DataLoader
import torchvision.ops as ops


from mmengine import Config

# -------------------- CONFIG -------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATASET_NAME = "sentinel2"
TIFF_DIR = "/cluster/home/malovhoi/letmecooknow/dino/data_finetune/sentinel2/data_msi"
CHECKPOINT_PATH = "/cluster/home/malovhoi/letmecooknow/dino/outputs/detectionhead_downsample/embed64_2/checkpoint0200.pth"
SAVE_DIR = "/cluster/home/malovhoi/letmecooknow/dino/data_extra/single_img_tif"
os.makedirs(SAVE_DIR, exist_ok=True)

def custom_collate_fn(batch):
    images, targets = zip(*batch)
    images = torch.stack(images, dim=0)
    return images, list(targets)



# ------------------ VISUALIZE ------------------------
def run_visualization():
    dataset = fo.load_dataset(DATASET_NAME)
    val_view = dataset.match_tags("val")
    torch_dataset = ShipDetectionDataset(val_view, TIFF_DIR)
    loader = DataLoader(torch_dataset, batch_size=1, shuffle=False, collate_fn = custom_collate_fn)

    cfg = Config.fromfile("specdetr/specdetrconfig.py")
    backbone = SpecDetrWrapper(**cfg["model"])

    ckpt = torch.load("/cluster/home/malovhoi/letmecooknow/dino/outputs/dino_outputs_embed64/checkpoint0100.pth", map_location="cpu")
    backbone_state = {k.replace("module.backbone.", ""): v for k, v in ckpt["student"].items() if k.startswith("module.backbone.")}
    backbone.load_state_dict(backbone_state, strict=False)
    backbone.to(DEVICE).eval()

    head = DetectionHead(embed_dim=64).to(DEVICE)
    head.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=DEVICE))
    head.eval()

    for i, (imgs, targets) in enumerate(loader):
        print(i)
        if i >= 40:
            break
        imgs = imgs.to(DEVICE)
        with torch.no_grad():
            feats = backbone(imgs)[:, :-1]
            logits, boxes = head(feats)
            probs = torch.sigmoid(logits[0, 0])  # (H, W)
            pred_boxes = boxes[0].permute(1, 2, 0)  # (H, W, 2)

            threshold = 0.93
            mask = probs > threshold

            pred_coords = pred_boxes[mask]
            grid_y, grid_x = torch.where(mask)
            W, H = probs.shape[::-1]

            filepath = list(val_view)[i].filepath  # 👈 Get the original filepath from the FiftyOne sample
            img = np.array(Image.open(filepath).convert("RGB"))
            fig, ax = plt.subplots()
            ax.imshow(img)
            boxes = []
            scores = []

            for (bx, by), box in zip(zip(grid_x, grid_y), pred_coords):
                pw, ph = box[0].item(), box[1].item()  # normalized width and height
                # Use grid_x and grid_y directly as centers in pixel space
                cx = bx.item()
                cy = by.item()

                # Compute top-left corner in normalized format
                x1 = (cx / W) - (pw / 2)
                y1 = (cy / H) - (ph / 2)
                x2= (cx / W) + (pw / 2)
                y2 = (cy / H) + (ph / 2)

                boxes.append([x1, y1, x2, y2])
                scores.append(probs[by, bx].item())


            if boxes:
                boxes = torch.tensor(boxes)
                scores = torch.tensor(scores)

                # --- Step 2: Run NMS ---
                keep = ops.nms(boxes, scores, iou_threshold=0.1)

                # --- Step 3: Draw only the kept boxes ---
                for idx in keep:
                    x1, y1, x2, y2 = boxes[idx]
                    pw = x2 - x1
                    ph = y2 - y1

                    rect = plt.Rectangle(
                        (x1 * W*8, y1 * H*8),
                        pw * W*8,
                        ph * H*8,
                        edgecolor='red',
                        facecolor='none',
                        linewidth=1,
                    )
                    ax.add_patch(rect)
                    print(f"  pred box: x1={x1:.3f}, y1={y1:.3f}, w={pw:.3f}, h={ph:.3f}")

            gt_boxes = targets[0]["boxes"].cpu()
            print("✅ Ground truth boxes:")
            for box in gt_boxes:
                x1, y1, x2, y2 = box.tolist()
                w = x2 - x1
                h = y2 - y1
                print(f"  [{x1:.3f}, {y1:.3f}, {w:.3f}, {h:.3f}]")
                rect = plt.Rectangle(
                    (x1*W*8, y1*H*8), w*W*8, h* H*8,
                    edgecolor='green', facecolor='none', linewidth=1
                )
                ax.add_patch(rect)

            ax.set_title(f"Predicted boxes (img {i})")
            ax.axis("off")
            plt.savefig(os.path.join(SAVE_DIR, f"pred_{i}.png"), bbox_inches="tight", pad_inches=0)
            plt.close()

    print(f"✅ Saved predictions to {SAVE_DIR}")


if __name__ == "__main__":
    run_visualization()

