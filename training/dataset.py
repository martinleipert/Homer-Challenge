# Path handling
import os

# Fiftyone => Datset handling
from fiftyone import Dataset as COCODataset
from fiftyone.types import COCODetectionDataset
from fiftyone.core.dataset import delete_datasets
from fiftyone.utils.splits import random_split

# Delete datasets
# Hard reset
delete_datasets("*")

# First dataset is the provided training dataset
DATASET_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "dataset", "HomerCompTraining")
DATASET_ANNOTATION = "HomerCompTrainingReadCoco.json"

coco_dataset = COCODataset.from_dir(
    dataset_type=COCODetectionDataset,
    data_path=DATASET_PATH,
    labels_path=os.path.join(DATASET_PATH, DATASET_ANNOTATION),
    include_id=True,
    label_field="annotations",
    name="ICFHR2022_train"
)

# We need validation data and therefore perform a split
random_split(coco_dataset, {"eval": 0.1, "train": 0.9})

coco_dataset.persistent = True

# Load the provided dataset for testing
TESTING_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "dataset", "HomerCompTesting")
TESTING_ANNOTATION = "HomerCompTestingReadCoco.json"

testing_dataset = COCODataset.from_dir(
    dataset_type=COCODetectionDataset,
    data_path=TESTING_PATH,
    labels_path=os.path.join(TESTING_PATH, TESTING_ANNOTATION),
    include_id=True,
    label_field="annotations",
    name="ICFHR2022_test"
)

testing_dataset.persistent = True

# Create a space for an artificial dataset
# The dataset will be filled after this in CreateArtificialPapyri
artificial_dataset = COCODataset("ICFHR2022_artificial")

# Copy the properties of the coco dataset
artificial_dataset.media_type = coco_dataset.media_type
artificial_dataset.default_classes = coco_dataset.default_classes
artificial_dataset.info["categories"] = coco_dataset.info["categories"]

artificial_dataset.persistent = True
