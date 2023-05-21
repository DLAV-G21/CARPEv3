import os

import torchvision
import os
import torch
import numpy as np
from PIL import Image
from tqdm.notebook import tqdm



def generate_image_segmentation(img_path,save_path_img, save_path_segm):
  device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
  model = torchvision.models.detection.maskrcnn_resnet50_fpn(weights=torchvision.models.detection.MaskRCNN_ResNet50_FPN_Weights.DEFAULT)
  model = model.to(device)
  model.eval()
  for x in tqdm(os.listdir(img_path)):
    img2 = np.array(Image.open(os.path.join(img_path,x)).convert("RGB"))
    img = torch.Tensor(img2).to(device).permute(2,0,1).unsqueeze(0)/255
    with torch.no_grad():
      predictions = model(img)
    
    mask = np.zeros((img2.shape[0],img2.shape[1]))
    for idx, i in enumerate(predictions[0]["boxes"]):
      if int(predictions[0]["labels"][idx]) == 3 and predictions[0]["scores"][idx] > 0.95:
        
        i = [int(j) for j in list(i.detach().cpu().numpy())]
        mask[i[1]:i[3],i[0]:i[2]] = 1
    
    np.savez_compressed(os.path.join(save_path_segm,x[:-4]+".npz"), mask)
    del(img2)
    del(img)
    del(predictions)

def generate_train_val_test_split(config,root_path):
  """
  The given validation dataset will give our testing set. 
  Then, we will use part of the train_split to make our validation dataset.
  """
  train_prefix = ["180116","171206","180117"]
  test_prefix = ["180118","180310"]
  val_prefix = ["180114"]
  train_data_list = []
  val_data_list = []
  test_data_list = []

  from os import listdir
  from os.path import isfile, isdir, join
  for f in listdir(join(root_path, "images")):
    if isfile(join(root_path, "images",f)):
      prefix = f.split("_")[0]
      
      if prefix in train_prefix:
        train_data_list.append(f)
      
      if prefix in val_prefix:
        val_data_list.append(f)
      
      if prefix in test_prefix:
        test_data_list.append(f)

  return train_data_list, val_data_list, test_data_list
