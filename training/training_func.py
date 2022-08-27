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

# Define the batch size
BATCH_SIZE = 1


def training_func(dataset_name="ICFHR2022_train", learning_rate=1e-4, model_dir="model_store",
                  model_name="model.pth", epochs=50, initialization=None, decay=20, decay_factor=0.7):
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
    model_base, file_ending = model_name.split(".")
    model_dir = os.path.join(os.path.dirname(__file__), model_dir)
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)

    # Load COCO dataset from disk
    dataset = load_dataset(dataset_name)

    # Load model
    model = retinanet_resnet50_fpn_v2(num_classes=26)
    # Initialize model
    if initialization is not None:
        model.load_state_dict(torch.load(initialization))
    model = model.to("cuda")

    # Learning rate and optimzation
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    lr_sheduler = StepLR(optimizer, step_size=decay, gamma=decay_factor)

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
        batch_size=BATCH_SIZE,
        collate_fn=FiftyOneTorchDataset.collate_fn,
        shuffle=True,
    )

    coco_eval_loader = DataLoader(
        coco_dataset_eval,
        batch_size=BATCH_SIZE,
        collate_fn=FiftyOneTorchDataset.collate_fn,
        shuffle=True,
    )

    n_samples_eval = len(coco_eval_loader)
    # endregion Initialization

    for epoch in range(epochs):

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
            print(
                f'{now} - Epoch: {epoch} | Iteration: {iteration} | Classification loss: {class_loss:1.5f} | '
                f'Regression loss: {regress_loss:1.5f} | Running loss: {np.mean(epoch_loss):1.5f}'
            )

        end_time = time.time()
        print(f"Epoch - {epoch} Ended, Learning Rate {lr_sheduler.get_lr()[0]}, Duration: {end_time - start_time}")
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

                print(
                    f'{now} - Epoch: {epoch} | Eval Iteration: {iteration} | Eval Score: {sample_score:1.5f}'
                )
            pass

        print(f"Epoch - Eval Score: {eval_score}")

        # Store model if it's the current best
        if eval_score > best_eval_score:
            best_eval_score = eval_score
            model_file_name = f"{model_base}_best.{file_ending}"
            modeL_out_path = os.path.join(model_dir, model_file_name)
            torch.save(model.state_dict(), modeL_out_path)

        if epoch % 10 == 0:
            model_file_name = f"{model_base}_e{epoch}.{file_ending}"
            modeL_out_path = os.path.join(model_dir, model_file_name)
            torch.save(model.state_dict(), modeL_out_path)

        lr_sheduler.step()
        # endregion Evaluation

    # Save the final model
    final_model = model.state_dict()
    torch.save(final_model, os.path.join(model_dir, model_name))


if __name__ == "__main__":
    main()
