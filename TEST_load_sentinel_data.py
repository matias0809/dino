from torchgeo.datasets import Sentinel2
from torchgeo.samplers import GridGeoSampler
from torch.utils.data import DataLoader
from torchgeo.datasets.utils import stack_samples
import torch
from torch.utils.data import RandomSampler
import os
import torch.distributed as dist


class DummyMultiCrop:
    def __call__(self, image):
        list = [image.clone() for _ in range(3)]  # 3 identical crops
        list.append(torch.zeros(13, 96 ,96))
        return list
    
class ApplyTransformToImage:
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, sample):
        sample["image"] = self.transform(sample["image"])
        return sample
    
transform = DummyMultiCrop()
dataset = Sentinel2(
    paths="/home/mnachour/master/dino/data",  # R, G, B at 10m resolution
    res=10,                       # force 10m resolution
    #transforms=ApplyTransformToImage(transform),  # apply the transform to the image
)

#print("Dataset length:", len(dataset))
"""
def image_and_label_collate(batch):
    images = torch.stack([b["image"] for b in batch])
    labels = torch.tensor([0 for _ in batch])  # dummy label
    return images, labels
"""
def image_and_label_collate(batch):
    # batch[i]["image"] is a list of tensors
    # Transpose the list of image lists to group by view
    crops_per_view = list(zip(*[b["image"] for b in batch]))  # shape: [3][B]
    images = [torch.stack(view) for view in crops_per_view]   # 3 tensors of [B, C, H, W]
    labels = torch.tensor([0 for _ in batch])
    return images, labels


sampler = GridGeoSampler(
    dataset=dataset,
    size=256,  # patch size in pixels
    stride=256
)


data_loader = torch.utils.data.DataLoader(
    dataset,
    sampler=sampler,
    batch_size=8,
    num_workers=10,
    pin_memory=True,
    drop_last=True,
    collate_fn=image_and_label_collate  # use custom collate function
    )

#print(list(sampler))

###TESTE WRAPPER FOR DDP OG DET PISSET

""" patches = list(sampler)
bbox = patches[0]
sample = dataset[bbox]
image = sample["image"]
print("Image len:", len(image))
print(image.shape)
print(len(patches)) """


""" sample_batch, lbl= next(iter(data_loader)) #her gjøres transformasjonene
imgs = (sample_batch[0])


print(next(iter(data_loader))[1].dtype)
print(imgs.shape) """



""" if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
    print("OKAY1")
elif 'SLURM_PROCID' in os.environ:
    print("OKAY2")
elif torch.cuda.is_available():   
    print("OKAY3")
else:       
    print("OKAY4")

print(dist.get_rank())
print(dist.get_world_size()) """



"""
for it, (images, _) in enumerate(data_loader):
        # update weight decay and learning rate according to their schedule
          # global training iteration

        # move images to gpu
        images = images.cuda(non_blocking=True)
        # teacher and student forward passes + compute dino loss
        print(images)
        print(images.shape)
        print("NEWWWWW", it)

"""
