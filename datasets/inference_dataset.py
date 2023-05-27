# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
COCO dataset which returns image_id for evaluation.

Mostly copy-paste from https://github.com/pytorch/vision/blob/13b35ff/references/detection/coco_utils.py
"""
from PIL import Image
import torch
import numpy as np
import torch.utils.data

from os import listdir
from os.path import isfile, isdir, join, split
import datasets.transforms as T

class InferenceDataset(torch.utils.data.Dataset):
    def __init__(self, img, transforms):
        super().__init__()
        self._transforms = transforms
        if isdir(img):
            self.folder = img
            self.imgs = [f for f in listdir(img) if isfile(join(img,f)) and (f.endswith(".png") or f.endswith(".jpg"))]
        elif isfile(img) and (img.endswith(".png") or img.endswith(".jpg")):
            self.folder, self.imgs = split(img)
            self.imgs = [self.imgs]
        else:
            raise ValueError("The given path should be a folder containing all the images for which the model should make predictions.")

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        img_name = self.imgs[idx]
        filename = join(self.folder, img_name)
        img = Image.open(filename)
        target = {
                    'image_id': torch.as_tensor([idx]), 'annotations': None,
                    'image': np.array(img), 'filename':img_name,
                    'orig_size': torch.as_tensor([img.height, img.width]),
                    'size': torch.as_tensor([img.height, img.width]),
                 }

        if self._transforms is not None:
            img, target = self._transforms(img, target)

        return img, target


def transform(size):
    max_size = 512
    normalize = T.Compose([
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    return T.Compose([
        T.RandomResize([size], max_size=max_size),
        normalize,
    ])


def build_inference(args):
    dataset = InferenceDataset(img=args.image, 
        transforms=transform(args.input_image_resize)
        )
    return dataset