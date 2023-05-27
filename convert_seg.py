import os
from PIL import Image, ImageOps

root = 'carp_data'
npz = os.path.join(root, 'train_segm_npz')
jpg = os.path.join(root, 'train_segm')
if not os.path.isdir(jpg):
    os.mkdir(jpg)
for file_name in os.listdir(npz):
    img = ImageOps.grayscale(Image.open(os.path.join(npz, file_name)))
    img.save(os.path.join(jpg, f'{os.path.splitext(file_name)[0]}.jpg'))
