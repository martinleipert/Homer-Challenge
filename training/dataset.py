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

"""
dataset = COCODataset.from_json(
    os.path.join(DATASET_PATH, DATASET_ANNOTATION),
    dataset_type=COCODetectionDataset,
    rel_dir=DATASET_PATH,
    name="ICFHR2022_train")
"""

"""
dataset = COCODataset.from_dir(
    dataset_dir=DATASET_PATH,
    # data_path=os.path.join(DATASET_PATH),
    dataset_type=COCODetectionDataset,
    name="ICFHR2022_train",
    labels_path=os.path.join(DATASET_PATH, DATASET_ANNOTATION)
)
"""

"""
train_dataset = COCODataset.from_json(
    os.path.join(DATASET_PATH, DATASET_ANNOTATION),
    "ICFHR2022_train"
)
train_dataset = COCODataset.from_archive(
    f"{DATASET_PATH}.zip",
    COCODetectionDataset,
    "images",
    DATASET_ANNOTATION,
    "ICFHR2022_train"
)

train_dataset = COCODataset.from_labeled_images()
"""
