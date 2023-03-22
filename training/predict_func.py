from argparse import ArgumentParser
import pathlib
import yaml

# Python standard libraries
import os
import time
import numpy as np
from PIL import Image

# Dataset Handling
from fiftyone import load_dataset
import fiftyone as fo

# Pytorch imports
import torch
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR
from torchvision.models.detection import RetinaNet

# Imports from own project
from PytorchDatasetFromFO import FiftyOneTorchDataset
from Augmentation import augmentation_func
from omegaconf import DictConfig, OmegaConf

import logging
import hydra
from hydra.utils import instantiate, call

# A logger for this file
log = logging.getLogger(__name__)

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True


BASE_PATH = "predictions"


def main(config_path):

    config_path = pathlib.Path(config_path)

    relative = config_path.relative_to(os.path.abspath("./outputs/"))

    new_path = os.path.join(BASE_PATH, relative.parent.parent)
    new_path = pathlib.Path(new_path)

    # Make rekursively
    os.makedirs(new_path, exist_ok = True)

    with open(config_path, "r") as openconfig:
        config_dict = yaml.full_load(openconfig)

    cfg = DictConfig(config_dict)

    model = instantiate(cfg.model.torch_module)

    model.eval()

    # Find best model
    state_path = os.path.join(config_path.parent.parent, "model_best.pth")
    state_dict = torch.load(state_path)
    model.load_state_dict(state_dict)
    model.to("cuda")

    # Load COCO dataset from disk
    # dataset = load_dataset(cfg.dataset_test.name)
    dataset = load_dataset(cfg.dataset.name)

    classes = dataset.default_classes

    for iteration, (sample) in enumerate(dataset):

        image = Image.open(sample.filepath)
        image = np.array(image)

        detections = []

        h, w, c = image.shape

        image = np.moveaxis(image, 2, 0)
        image = torch.tensor(image).float() / 255
        image = torch.unsqueeze(image, 0)
        image = image.to("cuda")
        # annotations = annotations
        result = model(image)

        for index, item in enumerate(result):
            # Store the results of the evaluation
            boxes = item["boxes"].detach().cpu().numpy()
            scores = item["scores"].detach().cpu().numpy()
            labels = item["labels"].detach().cpu().numpy()

            # Convert detections to FiftyOne format
            for label, score, box in zip(labels, scores, boxes):
                # Convert to [top-left-x, top-left-y, width, height]
                # in relative coordinates in [0, 1] x [0, 1]
                x1, y1, x2, y2 = box
                rel_box = [x1 / w, y1 / h, (x2 - x1) / w, (y2 - y1) / h]

                detections.append(
                    fo.Detection(
                        label=classes[label],
                        bounding_box=rel_box,
                        confidence=score
                    )
                )

        # Save predictions to dataset
        sample["retina_net"] = fo.Detections(detections=detections)
        sample.save()

    out_file = os.path.join(new_path, "prediction.json")

    dataset.export(dataset_type=fo.types.COCODetectionDataset, labels_path=out_file, label_field="ground_truth",
                   abs_paths=True)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("config", type=str)

    args = parser.parse_args()

    main(args.config)
