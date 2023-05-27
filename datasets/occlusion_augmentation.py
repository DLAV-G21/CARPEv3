import random
import numpy as np
import matplotlib.pyplot as plt
import cv2
from skimage.filters import gaussian
import time
from skimage.filters import gaussian


def apply_grid_masking(img, segm):
    img = img.copy()
    RECT_SIZE = 30
    MAX_RECT = 15
    nb_mask = random.randint(0,MAX_RECT)
    fill_value = np.array([128, 128, 128])

    indices_hor = random.choices(list(range(img.shape[1])),k=nb_mask)
    indices_ver = random.choices(list(range(img.shape[0])),k=nb_mask)

    RECT_SIZE = RECT_SIZE//2
    for i,j in zip(indices_hor, indices_ver):
        a = max(i - RECT_SIZE, 0)
        b = max(j - RECT_SIZE, 0)
        c = min(i + RECT_SIZE, img.shape[1])
        d = min(j + RECT_SIZE, img.shape[0])
        img[b:d,a:c] = fill_value

    return img
  
def get_source_in_mask(mask,nb_blur_source):
    coordinates = np.transpose(mask.nonzero())
    samples = random.choices(np.arange(len(coordinates)), k=nb_blur_source)
    return coordinates[samples,:]

def generate_image_segmentation(img, segm):
    NB_MAX_SOURCE = 10
    blur_radius = 15
    nb_blur_source = random.randint(0,NB_MAX_SOURCE) 
    blurred_img = (gaussian(img, sigma=30,channel_axis=2)*255).astype(np.uint8)
    sources = get_source_in_mask(segm, nb_blur_source)
    mask_2 = np.zeros((img.shape[0],img.shape[1],3), dtype=np.uint8)
    for s in sources:
        mask_2 = cv2.circle(mask_2, (s[1], s[0]), blur_radius, [255,255,255], -1)
        segm = cv2.circle(segm, (s[1], s[0]), blur_radius, 1, -1)
    
    out = np.where(mask_2==255, blurred_img, img)
    return out

def apply_blur_masking(img, segm):
    if random.random() < 0.5:
        # Blur the cars
        segm = segm.astype(np.uint8)
    else:
        # blur the background
        segm = (segm==0).astype(np.uint8)
    img = generate_image_segmentation(img, segm)
    return img
