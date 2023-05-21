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
from os.path import isfile, isdir, join
import datasets.transforms as T

class InferenceDataset(torch.utils.Dataset):
    def __init__(self, img, transforms):
        super().__init__()
        self._transforms = transforms
        if isdir(img):
            self.imgs = [f for f in listdir(img) if isfile(join(img,f)) and (f.endsWith(".png") or f.endsWith(".jpg"))]
        elif isfile(img) and (img.endsWith(".png") or img.endsWith(".jpg")):
            self.imgs = [img]
        else:
            raise ValueError("The given path should be a folder containing all the images for which the model should make predictions.")

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        img_name = self.imgs[idx]
        filename = join(self.img_folder,img_name)
        img = Image.open(filename)
        target = {
                    'image_id': idx, 'annotations': {},
                    'image': np.array(img), 'filename':img_name,
                    'orig_size': torch.as_tensor([img.height, img.width]),
                    'size': torch.as_tensor([img.height, img.width]),
                 }

        if self._transforms is not None:
            img, target = self._transforms(img, target)

        return img, target


def transform(size):
    normalize = T.Compose([
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    return T.Compose([
        T.Resize(size),
        normalize,
    ])


def build_inference(image_set, args):
    dataset = InferenceDataset(img_folder=args.image_folder, 
        transforms=transform(args.input_image_resize)
        )
    return dataset