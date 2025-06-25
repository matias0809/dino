###SCRIPT BY VAKE FOR INITIALIZNG DATASET
from pathlib import Path

import fiftyone as fo
from fiftyone.types import FiftyOneDataset

location = Path("/cluster/home/malovhoi/letmecooknow/dino/data_finetune")
fiftyone_name = "sentinel2"
d: fo.Dataset = fo.Dataset.from_dir(
    dataset_dir=(location / fiftyone_name).as_posix(),
    dataset_type=FiftyOneDataset,
    name=fiftyone_name,
)
d.persistent = True
d.save()