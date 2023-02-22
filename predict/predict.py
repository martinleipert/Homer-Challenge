# Python standard libraries
import os
import time
import numpy as np

# Dataset Handling
from fiftyone import load_dataset

from fiftyone.utils.coco import add_coco_labels

# Pytorch imports
import torch
from torch.utils.data import DataLoader
from torchvision.models.detection import retinanet_resnet50_fpn_v2

# Imports from own project
from training.PytorchDatasetFromFO import FiftyOneTorchDataset
from fiftyone.types import COCODetectionDataset

from PIL import Image


# Define the batch size
BATCH_SIZE = 1

DATASET = "ICFHR2022_test"
TRAIN_DATASET = "ICFHR2022_train"
MODEL_PATH = "final_models/model_with_pretraining.pth"

FINAL_ANNOTATIONS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "dataset", "HomerCompTesting",
                                      "HomerCompTestingReadCoco_Annoations.json")


def main():

    state_dict = torch.load(MODEL_PATH)

    model = retinanet_resnet50_fpn_v2(num_classes=26)
    model.load_state_dict(state_dict)
    model.eval()

    train_dataset = load_dataset(TRAIN_DATASET)
    train_dataset = FiftyOneTorchDataset(train_dataset)

    test_dataset = load_dataset(DATASET)
    # test_dataset = FiftyOneTorchDataset(test_dataset, classes=train_dataset.classes)

    classes = train_dataset.classes
    classes.append('.')

    classes_reverse = dict(list(enumerate(classes)))

    new_annotations = []

    for image_idx, item in enumerate(test_dataset):
        image = Image.open(item.filepath)
        image = np.array(image) / 255
        image = image.astype(np.float32)
        image = np.moveaxis(image, 2, 0)
        image = torch.tensor(image)
        image = torch.unsqueeze(image, 0)

        pred = model(image)[0]

        boxes = np.array(pred['boxes'].cpu().detach().numpy())
        labels = np.array(pred['labels'].cpu().detach().numpy())
        scores = np.array(pred['scores'].cpu().detach().numpy())

        for idx, label in enumerate(labels):

            new_label = classes_reverse[label]
            box = list(np.round(boxes[idx]).astype(np.int32))

            new = {
                "id": idx,
                "image_id": item.id,
                "category_id": label,
                "bbox": box,
                "prediction": new_label,

                # optional
                "score": scores[idx],
            }

            new_annotations.append(new)

    add_coco_labels(test_dataset, "predictions", new_annotations, classes, "detections", "id")

    test_dataset.export(
        dataset_type=COCODetectionDataset,
        labels_path=FINAL_ANNOTATIONS_FILE,
        label_field="predictions",
        encoding="utf-8"
    )
    pass


if __name__ == "__main__":
    main()
