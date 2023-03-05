# Python standard libraries
import os
import time
import numpy as np

# Dataset Handling
from fiftyone import load_dataset

# Pytorch imports
import torch
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR
from torchvision.models.detection import retinanet_resnet50_fpn_v2

# Imports from own project
from PytorchDatasetFromFO import FiftyOneTorchDataset
from Augmentation import augmentation_func
from omegaconf import DictConfig, OmegaConf

import logging
from hydra.utils import instantiate

# A logger for this file
log = logging.getLogger(__name__)

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True


@hydra.main(config_path="config.yaml")
def training_func(cfg: DictConfig):
    """
    Standardized training function
    :param dataset_name: Name of the dataset to use
    :param learning_rate: Initial learning rate
    :param model_dir: Directory to store the model states
    :param model_name: name of the model, also used for storing
    :param epochs: Number of epochs used in training
    :param initialization: Path to a stored model used for initializing
    :param decay: LR Decay epochs
    :param decay_factor: Decay factor
    :return:
    """

    # region Initialization
    # Eval store used for saving the best parameters
    best_eval_score = 0

    # Storage
    model_base, file_ending = cfg.model.name.split(".")

    # Load COCO dataset from disk
    dataset = load_dataset(cfg.dataset.name)

    # Load model
    model = instantiate(cfg.model.torch_module)
    # Initialize model
    # if initialization is not None:
    #     model.load_state_dict(torch.load(initialization))
    model = model.to("cuda")

    # Learning rate and optimzation
    if cfg.training.optimizer == "SGD":
        optimizer = torch.optim.SGD(model.parameters(), lr=cfg.training.learning_rate)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg.training.learning_rate)
    lr_sheduler = StepLR(optimizer, step_size=cfg.training.decay_epochs, gamma=cfg.training.decay_factor)

    coco_dataset = FiftyOneTorchDataset(
        dataset,
        transforms=augmentation_func
    )

    coco_dataset_eval = FiftyOneTorchDataset(
        dataset,
        split_tag="eval"
    )

    coco_loader = DataLoader(
        coco_dataset,
        batch_size=cfg.training.batch_size,
        collate_fn=FiftyOneTorchDataset.collate_fn,
        shuffle=cfg.dataset.shuffle,
    )

    coco_eval_loader = DataLoader(
        coco_dataset_eval,
        batch_size=cfg.training.batch_size,
        collate_fn=FiftyOneTorchDataset.collate_fn,
        shuffle=cfg.dataset.shuffle,
    )

    n_samples_eval = len(coco_eval_loader)
    # endregion Initialization

    for epoch in range(cfg.training.epochs):

        print("Epoch - {} Started".format(epoch))
        start_time = time.time()

        model.train()

        epoch_loss = []

        # region Training
        for iteration, (image, annotations) in enumerate(coco_loader):

            image = image.float().to("cuda") / 255
            annotations = annotations
            result = model(image, annotations)

            class_loss = result["classification"]
            regress_loss = result["bbox_regression"]

            batch_loss = class_loss + regress_loss

            if bool(batch_loss == 0):
                continue

            # Calculating Gradients
            batch_loss.backward()

            # Clip gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)

            # Updating Weights
            optimizer.step()

            # Epoch Loss
            epoch_loss.append(float(batch_loss))

            now = time.strftime("%m/%d/%Y, %H:%M:%S")

            # Iteration result
            log.info(
                f'{now} - Epoch: {epoch} | Iteration: {iteration} | Classification loss: {class_loss:1.5f} | '
                f'Regression loss: {regress_loss:1.5f} | Running loss: {np.mean(epoch_loss):1.5f}'
            )

        end_time = time.time()
        log.info(f"Epoch - {epoch} Ended, Learning Rate {lr_sheduler.get_lr()[0]}, Duration: {end_time - start_time}")
        # endregion Training

        # region Evaluation
        model.eval()
        eval_score = 0

        for iteration, (image, annotations) in enumerate(coco_eval_loader):
            image = image.float().to("cuda") / 255
            annotations = annotations
            result = model(image, annotations)

            for index, item in enumerate(result):

                # Store the results of the evaluation
                scores = item["scores"].detach().cpu().numpy()
                sample_score = scores.mean()
                eval_score += sample_score / n_samples_eval

                # Iteration result
                now = time.strftime("%m/%d/%Y, %H:%M:%S")

                log.info(
                    f'{now} - Epoch: {epoch} | Eval Iteration: {iteration} | Eval Score: {sample_score:1.5f}'
                )
            pass

        log.info(f"Epoch - Eval Score: {eval_score}")

        # Store model if it's the current best
        if eval_score > best_eval_score:
            best_eval_score = eval_score
            model_file_name = f"{model_base}_best.{file_ending}"
            torch.save(model.state_dict(), model_file_name)

        if epoch % 10 == 0:
            model_file_name = f"{model_base}_e{epoch}.{file_ending}"
            torch.save(model.state_dict(), model_file_name)

        lr_sheduler.step()
        # endregion Evaluation

    # Save the final model
    final_model = model.state_dict()
    torch.save(final_model, os.path.join(model_dir, model_name))


if __name__ == "__main__":
    main()
