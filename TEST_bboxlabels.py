import fiftyone as fo
import fiftyone.utils.annotations as foua
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import tifffile
import numpy as np

"""
# Load the dataset
dataset = fo.load_dataset("sentinel2")

# Inspect first few samples
for i, sample in enumerate(dataset):
    print(f"\n🔍 Sample {i}:")
    print(sample)  # full FiftyOne Sample object

    print("\n📦 Sample fields:")
    for field in sample.field_names:
        print(f" - {field}: {type(sample[field])}")

    if i >= 2:
        break
"""
"""
import fiftyone as fo
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os
import imageio.v3 as iio  # modern way to load PNG
import numpy as np

# --- Configuration ---
output_dir = "/cluster/home/malovhoi/letmecooknow/dino/data_extra/single_img_tif"
os.makedirs(output_dir, exist_ok=True)
save_path = os.path.join(output_dir, "bbox_overlay.png")

# --- Load dataset ---
dataset = fo.load_dataset("sentinel2")
i=0

# --- Find one sample with bounding boxes ---
for sample in dataset:
    print("iteration:", i)
    detections = sample.gt_bounding_boxes.detections
    if detections:  # at least one bbox
        print(f"✅ Found sample: {sample.filepath}")
        img = iio.imread(sample.filepath)

        # Safety: if grayscale, expand to RGB
        if img.ndim == 2:
            img = np.stack([img] * 3, axis=-1)

        h, w = img.shape[:2]
        fig, ax = plt.subplots()
        ax.imshow(img)

        for det in detections:
            label = det.label
            x, y, bw, bh = det.bounding_box  # all normalized
            x_pixel = x * w
            y_pixel = y * h
            bw_pixel = bw * w
            bh_pixel = bh * h

            rect = patches.Rectangle(
                (x_pixel, y_pixel),
                bw_pixel,
                bh_pixel,
                linewidth=2,
                edgecolor='red',
                facecolor='none'
            )
            ax.add_patch(rect)
            ax.text(
                x_pixel,
                y_pixel - 5,
                label,
                color='white',
                fontsize=8,
                bbox=dict(facecolor='red', alpha=0.5, pad=1)
            )

        ax.set_title("Detected bounding boxes")
        ax.axis('off')
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
        print(f"📸 Saved image with bboxes to: {save_path}")
    else:
        print("ONLY BACKGROUND!")
    i+=1
    if i==2:
        break
"""

import fiftyone as fo

# Load your dataset
dataset = fo.load_dataset("sentinel2")

# Convert to a list
samples_list = list(dataset)

# Check type and length
print(f"Type: {type(samples_list)}")  # should be <class 'list'>
print(f"Number of samples: {len(samples_list)}")

# Try indexing
print("\nFirst sample:")
print(samples_list[0])

print("\nFilepath of sample 5:")
print(samples_list[5].filepath)

print("\nLabel of first classification in sample 5:")
print(samples_list[5].gt_classifications.classifications[0].label)