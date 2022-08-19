import os
from fiftyone import Dataset as COCODataset
from fiftyone.types import COCODetectionDataset
from fiftyone.core.dataset import delete_datasets
from fiftyone.utils.splits import random_split

# Hand decoding
import json
import codecs


# Delete datasets
delete_datasets("*")

# Execute once for Dataset creation
DATASET_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "dataset", "HomerCompTraining")
DATASET_ANNOTATION = "HomerCompTrainingReadCoco.json"
# _Copy.json"

# kwargs = {
#     "encoding": "utf-8"
# }

coco_dataset = COCODataset.from_dir(
    dataset_type=COCODetectionDataset,
    data_path=DATASET_PATH,
    labels_path=os.path.join(DATASET_PATH, DATASET_ANNOTATION),
    include_id=True,
    label_field="annotations",
    name="ICFHR2022_train"
)

random_split(coco_dataset, {"eval": 0.1, "train": 0.9})

coco_dataset.persistent = True

# Create a space for an artificial dataset
artificial_dataset = COCODataset("ICFHR2022_artificial")

# Copy its properties
artificial_dataset.media_type = coco_dataset.media_type
artificial_dataset.default_classes = coco_dataset.default_classes
artificial_dataset.info["categories"] = coco_dataset.info["categories"]

artificial_dataset.persistent = True
