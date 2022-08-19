import os
from fiftyone import load_dataset
from torchvision.models.detection import RetinaNet, retinanet_resnet50_fpn_v2, RetinaNet_ResNet50_FPN_V2_Weights
from torchvision.models import resnet50, ResNet50_Weights
from PIL import Image
import fiftyone
# from dataset import coco_dataset
from torch.utils.data import DataLoader
import torch
import time
import numpy as np
from fiftyone.utils.splits import random_split

from PytorchDatasetFromFO import FiftyOneTorchDataset
from Augmentation import augmentation_func


DATASETS_AVAILABLE = ["ICFHR2022_train", "ICFHR2022_artificial"]
epochs = 50

PATH = "model_pretraining.pth"
BATCH_SIZE = 1


def training_func(dataset_name="ICFHR2022_train", learning_rate=1e-4, model_dir="model_store",
                  model_name="model.pth", epochs=50, initialization=None):

    best_eval_score = 1
    model_base, file_ending = model_name.split(".")

    model_dir = os.path.join(os.path.dirname(__file__), model_dir)
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)

    # Load COCO dataset from disk
    dataset = load_dataset(dataset_name)

    # splitted = random_split(dataset, {"train": 0.9, "val": 0.1})
    # train_dataset = splitted["train"]
    # validation_dataset = splitted["val"]
    # resnet = resnet50(ResNet50_Weights.IMAGENET1K_V2, out_channels=24)
    # model = RetinaNet(backbone=resnet, num_classes=24)

    model = retinanet_resnet50_fpn_v2(num_classes=26)

    if initialization is not None:
        model.load_state_dict(torch.load(initialization))

    #   , weights=RetinaNet_ResNet50_FPN_V2_Weights.COCO_V1)
    model = model.to("cuda")

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

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
        # num_workers=4
    )

    coco_eval_loader = DataLoader(
        coco_dataset_eval,
        batch_size=BATCH_SIZE,
        collate_fn=FiftyOneTorchDataset.collate_fn,
        shuffle=True,
    )

    n_samples_eval = len(coco_eval_loader)
    # session = fiftyone.launch_app(coco_dataset)

    for epoch in range(epochs):

        print("Epoch - {} Started".format(epoch))
        start_time = time.time()

        model.train()

        epoch_loss = []

        for iteration, (image, annotations) in enumerate(coco_loader):

            # annot = torch.concat([annotations["boxes"], annotations["labels"].unsqueeze(-1)], 2)
            image = image.float().to("cuda") / 255
            annotations = annotations
            result = model(image, annotations)

            class_loss = result["classification"]
            regress_loss = result["bbox_regression"]

            # class_loss = class_loss.mean()
            # regress_loss = regress_loss.mean()

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
        print(f"Epoch - {epoch} Ended, Duration: {end_time - start_time}")

        # region Evaluation
        model.eval()
        eval_score = 0

        for iteration, (image, annotations) in enumerate(coco_eval_loader):
            image = image.float().to("cuda") / 255
            annotations = annotations
            result = model(image, annotations)

            # class_loss = result["classification"]
            # regress_loss = result["bbox_regression"]

            # eval_loss += (class_loss + regress_loss) / n_samples_eval

            for index, item in enumerate(result):

                idx = iteration * BATCH_SIZE + index
                bboxes = item["boxes"].detach().cpu().numpy()
                scores = item["scores"].detach().cpu().numpy()
                labels = item["labels"].detach().cpu().numpy()

                sample_score = scores.mean()
                eval_score += sample_score / n_samples_eval

                # Iteration result
                now = time.strftime("%m/%d/%Y, %H:%M:%S")

                print(
                    f'{now} - Epoch: {epoch} | Eval Iteration: {iteration} | Eval score: {sample_score:1.5f}'
                )
            pass

        print(f"Epoch - Eval Score: {eval_score}")

        if eval_score > best_eval_score:
            best_eval_score = eval_score
            model_file_name = f"{model_base}_best.{file_ending}"
            modeL_out_path = os.path.join(model_dir, model_file_name)
            torch.save(model.state_dict(), modeL_out_path)

        if epoch % 10 == 0:
            model_file_name = f"{model_base}_e{epoch}.{file_ending}"
            modeL_out_path = os.path.join(model_dir, model_file_name)
            torch.save(model.state_dict(), modeL_out_path)
        # endregion Evaluation

    torch.save(model.state_dict(), os.path.join(model_dir, model_name))


if __name__ == "__main__":
    main()
