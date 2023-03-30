
from albumentations.augmentations import RandomBrightnessContrast, HueSaturationValue, ShiftScaleRotate, GaussianBlur, \
    ISONoise, Sharpen, ImageCompression, GridDistortion, RandomShadow, RandomSnow, RandomCrop, BBoxSafeRandomCrop, \
    RandomSizedCrop, PadIfNeeded, Normalize
from albumentations import OneOf, Compose, BboxParams
import numpy as np
import torch

standard_augmentation = Compose([

    PadIfNeeded(1280, 1280),
    RandomSizedCrop([1280, 1280], 1280, 1280, always_apply=True),
    # Effekte durch Dokumentenlage
    ShiftScaleRotate(rotate_limit=3, scale_limit=0.1, p=0.7),
    # Effekte durch Dokument
    # Farbverschiebung
    HueSaturationValue(15, 20, 15, p=0.3),
    # Kontrast und Helligkeit
    RandomBrightnessContrast(0.1, 0.2, p=0.5),
    # Effekte durch Altertung
    # Wasserschaden
    RandomShadow(shadow_dimension=50, p=0.2),
    # Löcher
    RandomSnow(p=0.2),
    # Effekte durch die Aufnahme bedingt
    # Rauschen
    OneOf([
        ISONoise(intensity=(5e-2, 1e-1), p=0.2),
        # Blur
        GaussianBlur((3, 5), p=0.5),
        # Schärfe
        Sharpen(p=0.3)
    ], p=0.4),
    # Kompression
    ImageCompression(quality_lower=60),
    Normalize()

    # Distorsion
    # Not active due to bboxes
    # GridDistortion(distort_limit=5e-2, p=0.2)
], bbox_params=BboxParams(format='coco', label_fields=['labels'], min_area=250, min_visibility=0.3))

no_aug = Compose([PadIfNeeded(1280, 1280),
                  RandomSizedCrop([1280, 1280], 1280, 1280, always_apply=True),
                    Normalize()],
                 bbox_params=BboxParams(format='coco', label_fields=['labels'], min_area=250, min_visibility=0.3))

def augmentation_func(image, target):
    bboxes = target["boxes"].cpu().numpy()

    image = np.moveaxis(image, 0, 2)
    shape = image.shape

    bboxes = trafo_bboxes_from_coco(bboxes)
    bboxes = clip_boxes(bboxes, shape)
    bboxes = trafo_bboxes_to_coco(bboxes)

    transformed = standard_augmentation(image=image, bboxes=bboxes, labels=target["labels"])
    transformed_image = transformed['image']
    transformed_bboxes = np.array(transformed['bboxes'])

    transformed_bboxes = trafo_bboxes_from_coco(transformed_bboxes)

    target["boxes"] = torch.tensor(transformed_bboxes.astype(np.float32)).to("cuda")

    transformed_image = np.moveaxis(transformed_image, 2, 0)

    if target["boxes"].shape == torch.Size([0]):
        target["boxes"] = torch.empty((0, 4)).to("cuda")
        target["labels"] = torch.empty((0, ), dtype=torch.int64).to("cuda")
        target["area"] = torch.empty((0, )).to("cuda")
        target["iscrowd"] = torch.empty((0, )).to("cuda")

    return transformed_image, target


def no_augmentation_func(image, target):

    image = np.moveaxis(image, 0, 2)
    shape = image.shape

    bboxes = target["boxes"].cpu().numpy()

    bboxes = trafo_bboxes_from_coco(bboxes)
    bboxes = clip_boxes(bboxes, image.shape)
    bboxes = trafo_bboxes_to_coco(bboxes)

    transformed = no_aug(image=image, bboxes=bboxes, labels=target["labels"])

    transformed_image = transformed['image']

    transformed_image = np.moveaxis(transformed_image, 2, 0)

    transformed_bboxes = np.array(transformed['bboxes'])

    transformed_bboxes = trafo_bboxes_from_coco(transformed_bboxes)

    target["boxes"] = torch.tensor(transformed_bboxes.astype(np.float32)).to("cuda")

    if target["boxes"].shape == torch.Size([0]):
        target["boxes"] = torch.empty((0, 4)).to("cuda")
        target["labels"] = torch.empty((0, ), dtype=torch.int64).to("cuda")
        target["area"] = torch.empty((0, )).to("cuda")
        target["iscrowd"] = torch.empty((0, )).to("cuda")

    return transformed_image, target


def clip_boxes(bboxes, im_shape):
    """
    Clip the bounding boxes to fit them into the image
    :param bboxes: bboxes in x, y, w, h format (coco)
    :param im_shape:
    :return:
    """
    if bboxes.size == 0:
        return bboxes
    if len(np.shape(bboxes)) == 1:
        bboxes[0] = np.clip(bboxes[0], 0, im_shape[1])
        bboxes[1] = np.clip(bboxes[1], 0, im_shape[0])
        bboxes[2] = np.clip(bboxes[2], 0, im_shape[1])
        bboxes[3] = np.clip(bboxes[3], 0, im_shape[0])
    else:
        bboxes[:, 0] = np.clip(bboxes[:, 0], 0, im_shape[1])
        bboxes[:, 1] = np.clip(bboxes[:, 1], 0, im_shape[0])
        bboxes[:, 2] = np.clip(bboxes[:, 2], 0, im_shape[1])
        bboxes[:, 3] = np.clip(bboxes[:, 3], 0, im_shape[0])
    return bboxes


def trafo_bboxes_from_coco(bboxes):
    """
    Transform bounding boxes from coco x,y,w,h format to x1,y1,x2,y2 format
    :param bboxes: coco bboxes as numpy array
    :return: transformed bboxes as numpy array
    """
    if bboxes.size == 0:
        return bboxes
    if len(np.shape(bboxes)) == 1:
        bboxes[2:4] = np.add(bboxes[0:2], bboxes[2:4])
    else:
        bboxes[:, 2:4] = np.add(bboxes[:, 0:2], bboxes[:, 2:4])
    return bboxes


def trafo_bboxes_to_coco(bboxes):
    """
    Transform bounding boxes from x1,y1,x2,y2 format to coco x,y,w,h format to
    :param bboxes: coco bboxes as numpy array
    :return: transformed bboxes as numpy array
    """
    if bboxes.size == 0:
        return bboxes
    if len(np.shape(bboxes)) == 1:
        bboxes[2:4] = np.subtract(bboxes[2:4], bboxes[0:2])
    else:
        bboxes[:, 2:4] = np.subtract(bboxes[:, 2:4], bboxes[:, 0:2])
    return bboxes
