# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
'''
COCO dataset which returns image_id for evaluation.

Mostly copy-paste from https://github.com/pytorch/vision/blob/13b35ff/references/detection/coco_utils.py
'''
from pathlib import Path

import albumentations as al
import cv2
import numpy as np
import torch
import torch.utils.data
import torchvision
from PIL import Image
from pycocotools import mask as coco_mask
import os
import random
from .occlusion_augmentation import apply_grid_masking, apply_blur_masking

import datasets.transforms as T


class CocoDetection(torchvision.datasets.CocoDetection):
    def __init__(self, img_folder, ann_file, transforms, al_transforms, apply_augm, occlusion_augmentation_transforms, apply_occlusion_augmentation, segmentation_folder, split):
        super(CocoDetection, self).__init__(img_folder, ann_file)
        self._transforms = transforms
        self.al_transforms = al_transforms
        self.prepare = ConvertCocoPolysToMask(split)
        self.apply_augm = apply_augm
        self.split = split
        self.occlusion_augmentation_transforms = occlusion_augmentation_transforms
        self.segmentation_folder = segmentation_folder
        self.apply_occlusion_augmentation = apply_occlusion_augmentation and (segmentation_folder is not None) and os.path.isdir(segmentation_folder) and split == 'train'

    def __getitem__(self, idx):
        img, target = super(CocoDetection, self).__getitem__(idx)
        image_id = self.ids[idx]
        img_copy = np.array(img.copy())
        target = { 'image_id': image_id, 'annotations': target }
        file_name = self.coco.loadImgs(image_id)[0]['file_name']

        img, target = self.prepare(img, target)

        if self._transforms is not None:
            img, target = self._transforms[0](img, target)

        if self.occlusion_augmentation_transforms is not None and self.apply_occlusion_augmentation:
            img = np.array(img)
            seg = np.zeros(img.shape[:2])
            offset = 10
            for anotation in target['boxes']:
                a,b,c,d = anotation.numpy()
                a = max(int(a - offset), 0)
                b = max(int(b - offset), 0)
                c = min(int(c + offset), img.shape[1])
                d = min(int(d + offset), img.shape[0])
                seg[b:d,a:c] = 1

            for mod in self.occlusion_augmentation_transforms:
                if(random.random() < 0.25):
                    img = mod(img, seg)
            img = Image.fromarray(img)

        if self.al_transforms is not None and self.apply_augm:
            img = np.array(img)
            img = self.al_transforms(image=img)['image']
            
        if self._transforms is not None:
            img, target = self._transforms[1](img, target)

        target['image'] = img_copy
        target['filename'] = file_name
        target['labels'] -= 1
        return img, target


def convert_coco_poly_to_mask(segmentations, height, width):
    masks = []
    for polygons in segmentations:
        rles = coco_mask.frPyObjects(polygons, height, width)
        mask = coco_mask.decode(rles)
        if len(mask.shape) < 3:
            mask = mask[..., None]
        mask = torch.as_tensor(mask, dtype=torch.uint8)
        mask = mask.any(dim=2)
        masks.append(mask)
    if masks:
        masks = torch.stack(masks, dim=0)
    else:
        masks = torch.zeros((0, height, width), dtype=torch.uint8)
    return masks


class ConvertCocoPolysToMask(object):
    def __init__(self, split):
        self.split = split

    def __call__(self, image, target):
        w, h = image.size

        img_array = np.array(image)
        if len(img_array.shape) == 2:
            img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
            image = Image.fromarray(img_array)

        image_id = target['image_id']
        image_id = torch.tensor([image_id])

        anno = target['annotations']

        anno = [obj for obj in anno if ('iscrowd' not in obj or obj['iscrowd'] == 0) and (obj['num_keypoints']!=0)]

        boxes = [obj['bbox'] for obj in anno]
        # guard against no boxes via resizing
        boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        boxes[:, 2:] += boxes[:, :2]
        boxes[:, 0::2].clamp_(min=0, max=w)
        boxes[:, 1::2].clamp_(min=0, max=h)

        classes = [obj['category_id'] for obj in anno]
        classes = torch.tensor(classes, dtype=torch.int64)


        keypoints = None
        if anno and 'keypoints' in anno[0]:
            keypoints = [obj['keypoints'] for obj in anno]
            keypoints = torch.as_tensor(keypoints, dtype=torch.float32)
            num_keypoints = keypoints.shape[0]
            if num_keypoints:
                keypoints = keypoints.view(num_keypoints, -1, 3)

        keep = (boxes[:, 3] >= boxes[:, 1]) & (boxes[:, 2] >= boxes[:, 0])

        target = {}
        target['boxes'] = boxes[keep]
        target['labels'] = classes[keep]
        target['image_id'] = image_id
        if keypoints is not None:
            target['keypoints'] = keypoints[keep]

        # for conversion to coco api
        area = torch.tensor([obj['area'] for obj in anno])
        iscrowd = torch.tensor([obj['iscrowd'] if 'iscrowd' in obj else 0 for obj in anno])
        target['area'] = area[keep]
        target['iscrowd'] = iscrowd[keep]

        target['orig_size'] = torch.as_tensor([int(h), int(w)])
        target['size'] = torch.as_tensor([int(h), int(w)])
        
        return image, target

def make_coco_transforms(image_set):

    normalize = T.Compose([
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    #TODO ADAPTE
    # scales = [480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800]
    scales = [480, 500]
    val_scale = [scales[-1]]
    max_size = 512

    if image_set == 'train':
        return (T.Compose([
            T.RandomHorizontalFlip(),
            T.RandomSelect(
                T.Compose([
                T.RandomResize(scales, max_size=max_size),
                ]),
                T.Compose([
                    T.RandomResize([400, 500]),
                    T.RandomSizeCrop(384, 600),
                    T.RandomResize(scales, max_size=max_size),
                ])
            ),
        ]), normalize)

    if image_set == 'val'or image_set == 'test':
        return (T.Compose([
            T.RandomResize(val_scale, max_size=max_size),
        ]), normalize)

    raise ValueError(f'unknown {image_set}')

def albumentations_transform(image_set):
    if image_set == 'train': 
        return al.Compose([
        al.ColorJitter(0.4, 0.4, 0.5, 0.2, p=0.6),
        al.RandomBrightnessContrast(p=0.5),
        al.ToGray(p=0.01),
        al.FancyPCA(p=0.2),
        al.ImageCompression(50, 80,p=0.1),
        al.RandomSunFlare(p=0.05),
        al.Solarize(p=0.05),
        al.GaussNoise(var_limit=(1.0,30.0), p=0.1),
      ])
    return None
    
def occlusion_augmentation_transform(image_set):
    if image_set == 'train': 
        return [apply_grid_masking, apply_blur_masking]
    return None

def build(image_set, args):
    root = Path(args.coco_path)
    assert root.exists(), f'provided COCO path {root} does not exist'
    mode = 'keypoints'
    PATHS = {
        'train': (root / 'train',  root/'annotations'/f'{mode}_train_{args.num_keypoints}.json',root/'train_segm_npz'),
        'val': (root / 'val',  root/'annotations'/f'{mode}_val_{args.num_keypoints}.json',root/'val_segm_npz'),
        'test': (root / 'test', root/'annotations'/f'{mode}_test_{args.num_keypoints}.json',root/'test_segm_npz')
    }

    img_folder, ann_file, segmentation_folder = PATHS[image_set]
    dataset = CocoDetection(img_folder, ann_file, 
        transforms=make_coco_transforms(image_set), 
        al_transforms=albumentations_transform(image_set),
        apply_augm=args.apply_augmentation,
        occlusion_augmentation_transforms=occlusion_augmentation_transform(image_set),
        apply_occlusion_augmentation=args.apply_occlusion_augmentation,
        segmentation_folder=segmentation_folder, 
        split=image_set)
    return dataset