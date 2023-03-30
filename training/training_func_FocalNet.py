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
from torchvision.models.detection import RetinaNet, MaskRCNN, FasterRCNN, FasterRCNN_ResNet50_FPN_V2_Weights
import torch.nn as nn
from torchvision.transforms import RandomCrop

from torchvision.models.detection.faster_rcnn import FastRCNNConvFCHead, RPNHead, AnchorGenerator
from torchvision.models.detection.mask_rcnn import MaskRCNNHeads
from torchvision.models.detection.rpn import RPNHead, RegionProposalNetwork

from focalnet.FocalNet import FocalNet
from focalnet.model_config import model

# Imports from own project
from PytorchDatasetFromFO import FiftyOneTorchDataset
from Augmentation import augmentation_func
from omegaconf import DictConfig, OmegaConf

from timm import create_model

import logging
import hydra
from hydra.utils import instantiate, call

# A logger for this file
log = logging.getLogger(__name__)

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True


@hydra.main(config_name="config.yaml")
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
    # model = instantiate(cfg.model.torch_module)

    """
    TYPE: focalnet_huge_fl4
    NAME: focalnet_huge_fl4
    PRETRAINED: True
    NUM_CLASSES: 21842
    DROP_PATH_RATE:
    FOCAL:
    EMBED_DIM: 352
    DEPTHS: 
    FOCAL_LEVELS: [4, 4, 4, 4]
    FOCAL_WINDOWS:
    USE_CONV_EMBED: True
    USE_POSTLN: True
    USE_LAYERSCALE: True
    USE_POSTLN_IN_MODULATION:
    """
    """
    "focal",
    pretrained = False,
    img_size = 224,
    num_classes = 26,
    drop_path_rate = 0.5,
    focal_levels = [4, 4, 4, 4],
    focal_windows = [3, 3, 3, 3],
    use_conv_embed = True,
    use_layerscale = True,
    use_postln = True,
    use_postln_in_modulation = True,
    normalize_modulator = True,
    """

    # model = create_model(model)

    backbone = FocalNet(
        pretrain_img_size=1333,
        patch_size=4,
        in_chans=3,
        embed_dim=128,
        depths=(2, 2, 4, 4, 2),
        mlp_ratio=4.,
        drop_rate=0.,
        drop_path_rate=0.2,
        norm_layer=nn.LayerNorm,
        patch_norm=True,
        out_indices=(0, 1, 2, 3, 4),
        frozen_stages=-1,
        focal_levels=[4, 4, 4, 4, 4],
        focal_windows=[3, 3, 3, 3, 3],
        use_conv_embed=True,
        use_checkpoint=False,
        use_layerscale=True
        )

    backbone.out_channels = 128

    # anchor_generator = AnchorGenerator(sizes=((32, 64, 128, 256, 512),),
    #                                   aspect_ratios=((0.5, 1.0, 2.0),))

    # rpn = RegionProposalNetwork()

    """
    box_head = FastRCNNConvFCHead(
        (backbone.out_channels[3], 7, 7), backbone.out_channels, [1024], norm_layer=nn.BatchNorm2d
    )
    mask_head = MaskRCNNHeads(backbone.out_channels[3], backbone.out_channels, 1, norm_layer=nn.BatchNorm2d)

    rpn_head = RPNHead(backbone.out_channels, 32)
    """

    model = FasterRCNN(backbone, num_classes=26) #, rpn_anchor_generator=anchor_generator)

    # Initialize model
    # if initialization is not None:
    #     model.load_state_dict(torch.load(initialization))
    model = model.to("cuda")

    # Learning rate and optimzation
    if cfg.training.optimizer == "SGD":
        optimizer = torch.optim.SGD(model.parameters(), lr=cfg.training.learning_rate)
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.training.learning_rate)
    lr_sheduler = StepLR(optimizer, step_size=cfg.training.decay_epochs, gamma=cfg.training.decay_factor)

    # transforms = RandomCrop(1280, pad_if_needed=True)

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
        batch_size=1, # cfg.training.batch_size,
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

            image = image.float().to("cuda")
            annotations = annotations

            result = model(image, annotations)

            class_loss = result["loss_classifier"]
            regress_loss = result["loss_box_reg"]
            objectness_loss = result["loss_objectness"]
            loss_rpn_regress = result["loss_rpn_box_reg"]

            # if epoch > 50:
            #     batch_loss = class_loss + regress_loss + objectness_loss + loss_rpn_regress
            # else:
            batch_loss = class_loss + regress_loss + objectness_loss + loss_rpn_regress

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

            # Epoch result
            log.info(
                f'{now} - Epoch: {epoch} | Iteration: {iteration} | Classification loss: {class_loss:1.5f} | '
                f'Objectness loss: {objectness_loss:1.5f} | RPN Regression loss: {loss_rpn_regress:1.5f} |'
                f'Regression loss: {regress_loss:1.5f} | Running loss: {np.mean(epoch_loss):1.5f}'
            )

        end_time = time.time()
        log.info(f"Epoch - {epoch} Ended, Learning Rate {lr_sheduler.get_lr()[0]}, Duration: {end_time - start_time}")
        # endregion Training

        # region Evaluation
        model.eval()
        eval_score = 0

        for iteration, (image, annotations) in enumerate(coco_eval_loader):
            image = image.float().to("cuda")
            annotations = annotations
            result = model(image, annotations)

            for index, item in enumerate(result):

                # Store the results of the evaluation
                scores = item["scores"].detach().cpu().numpy()
                sample_score = np.nan_to_num(scores.mean(), 0)
                eval_score += sample_score / n_samples_eval

                log.info(
                    f'Epoch: {epoch} | Eval Iteration: {iteration} | Eval Score: {sample_score:1.5f}'
                )

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
    model_file_name = f"{model_base}_final.{file_ending}"
    torch.save(final_model, model_file_name)


if __name__ == "__main__":
    training_func()
