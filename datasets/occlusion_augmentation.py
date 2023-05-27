import random
import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image
from skimage.filters import gaussian
import time
from albumentations.augmentations.blur.transforms import GaussianBlur


random.seed(1011)
def apply_grid_masking(img, segm):
  RECT_SIZE = 25*6
  MAX_RECT = 15
  nb_mask = random.randint(0,MAX_RECT)
  fill_value = np.array([128, 128, 128])

  indices_hor = random.choices(list(range(0, img.shape[1]//RECT_SIZE)),k=nb_mask)
  indices_ver = random.choices(list(range(0, img.shape[0]//RECT_SIZE)),k=nb_mask)

  for i,j in zip(indices_hor, indices_ver):
    img[j*RECT_SIZE:(j+1)*RECT_SIZE, i*RECT_SIZE:(i+1)*RECT_SIZE] = fill_value

  return img
  
def get_source_in_mask(mask,nb_blur_source):
    coordinates = np.transpose(mask.nonzero())
    samples = random.choices(np.arange(len(coordinates)), k=nb_blur_source)
    return coordinates[samples,:]

def generate_image_segmentation(img, segm):
    NB_MAX_SOURCE = 10
    nb_blur_source = random.randint(0,NB_MAX_SOURCE) 
    blur_radius = 15*6

    
    gaussian = GaussianBlur(blur_limit=(7, 7), sigma_limit=30, always_apply=True, p=1.0)
    img = img*255
    blurred_img = gaussian(image = img)['image'].astype(np.uint8)

    #blurred_img = (gaussian(img, sigma=30,channel_axis=2)*255).astype(np.uint8)

    sources = get_source_in_mask(segm,nb_blur_source)
    mask_2 = np.zeros((img.shape[0],img.shape[1],3), dtype=np.uint8)
    for s in sources:
        mask_2 = cv2.circle(mask_2, (s[1], s[0]), blur_radius, [255,255,255], -1)
        segm = cv2.circle(segm, (s[1], s[0]), blur_radius, 1, -1)

    out = np.where(mask_2==255, blurred_img, img)
    return out

def apply_blur_masking(img, segm):
    s = time.time()
    if random.random() < 0.5:
        # Blur the cars
        segm = segm.astype(np.uint8)
    else:
        # blur the background
        segm = (segm==0).astype(np.uint8)
    img = generate_image_segmentation(img, segm)

    d = time.time()-s
    print()
    print(f'apply_blur_masking took {d} seconds')
    return img
