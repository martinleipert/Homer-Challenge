import torch
import fiftyone.utils.coco as fouc
import numpy
from PIL import Image, ImageFile
from Augmentation import no_augmentation_func
ImageFile.LOAD_TRUNCATED_IMAGES = True


class FiftyOneTorchDataset(torch.utils.data.Dataset):
    """A class to construct a PyTorch dataset from a FiftyOne dataset.

    Args:
        fiftyone_dataset: a FiftyOne dataset or view that will be used for training or testing
        transforms (None): a list of PyTorch transforms to apply to images and targets when loading
        gt_field ("ground_truth"): the name of the field in fiftyone_dataset that contains the
            desired labels to load
        classes (None): a list of class strings that are used to define the mapping between
            class names and indices. If None, it will use all classes present in the given fiftyone_dataset.
    """

    def __init__(
            self,
            fiftyone_dataset,
            transforms=None,
            gt_field="annotations_detections",
            classes=None,
            # "train", "val" or "test"
            split_tag="train"
    ):
        self.split_tag = split_tag
        self.samples = fiftyone_dataset.match_tags(self.split_tag, bool=True)
        self.transforms = transforms
        self.gt_field = gt_field

        self.img_paths = self.samples.values("filepath")

        self.classes = classes
        if not self.classes:
            # Get list of distinct labels that exist in the view
            self.classes = self.samples.distinct(
                "%s.detections.label" % gt_field
            )

        if self.classes[0] != "background":
            self.classes = ["background"] + self.classes

        self.labels_map_rev = {c: i for i, c in enumerate(self.classes)}

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        sample = self.samples[img_path]
        metadata = sample.metadata
        img = Image.open(img_path).convert("RGB")
        img = numpy.array(img)
        img = numpy.moveaxis(img, 2, 0)

        boxes = []
        labels = []
        area = []
        iscrowd = []
        detections = sample[self.gt_field].detections

        # new_target = list()

        for det in detections:
            category_id = self.labels_map_rev[det.label]
            coco_obj = fouc.COCOObject.from_label(
                det, metadata, category_id=category_id,
            )
            x, y, w, h = coco_obj.bbox
            boxes.append([x, y, w, h])
            labels.append(coco_obj.category_id)
            area.append(coco_obj.area)
            iscrowd.append(coco_obj.iscrowd)

        target = {}
        target["bboxes"] = torch.as_tensor(boxes, dtype=torch.float32).to("cuda")
        target["labels"] = torch.as_tensor(labels, dtype=torch.int64).to("cuda")
        target["image_id"] = torch.as_tensor([idx]).to("cuda")
        target["area"] = torch.as_tensor(area, dtype=torch.float32).to("cuda")
        target["iscrowd"] = torch.as_tensor(iscrowd, dtype=torch.int64).to("cuda")

        if self.transforms is not None:
            img, target = self.transforms(img, target)
        else:
            img, target = no_augmentation_func(img, target)

        return img, target

    def __len__(self):
        return len(self.img_paths)

    def get_classes(self):
        return self.classes

    @classmethod
    def collate_fn(cls, batch):
        """
        Since each image may have a different number of objects, we need a collate function (to be passed to the DataLoader).
        This describes how to combine these tensors of different sizes. We use lists.
        Note: this need not be defined in this Class, can be standalone.
        :param batch: an iterable of N sets from __getitem__()
        :return: a tensor of images, lists of varying-size tensors of bounding boxes, labels, and difficulties
        """

        images = list()
        targets = list()

        for image, target in batch:
            images.append(torch.Tensor(image))
            targets.append(target)

        images = torch.stack(images, dim=0)

        return images, targets
