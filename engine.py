# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Train and eval functions used in main.py
"""
import numpy as np
import os
from typing import Iterable

import torch
import cv2
from datasets.coco_eval import CocoEvaluator
from tqdm import tqdm
import itertools
from util.openpifpaf_helper import *
import json

def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, max_norm: float = 0, logger = None,
                    postprocessors = None, num_keypoints=24, visualize_folder=None, json_file=None):
    model.train()
    criterion.train()
    len_dl = len(data_loader)
    pbar = tqdm(data_loader)
    pbar.set_description(f"Epoch {epoch}, loss = init")
    for i, (samples, targets) in enumerate(pbar):
        samples = samples.to(device)
        targets =  [{k: v.to(device) if (v is not None) and (k not in ["image", "filename", "segmentation"]) else v for k, v in t.items()} for t in targets ]

        outputs = model(samples)
        loss_dict = criterion(outputs, targets)
        weight_dict = criterion.weight_dict

        losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

        if logger is not None: 
            logger.add_scalar("Loss/train",losses.item(),len_dl*epoch + i)

        optimizer.zero_grad()
        losses.backward()
        pbar.set_description(f"Epoch {epoch}, loss = {losses.item():.4f}")
        if max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        optimizer.step()

        results_outputs, results_targets = visualize(visualize_folder, targets, num_keypoints, device, postprocessors, outputs)
        results_outputs, results_targets = json_out(json_file, targets, device, postprocessors, outputs, results_outputs, results_targets)

@torch.no_grad()
def evaluate(model, criterion, postprocessors, data_loader, base_ds, device, epoch=0, logger=None, num_keypoints=24, visualize_folder=None, json_file=None):
    model.eval()
    if criterion is not None:
        criterion.eval()
    iou_types = ["keypoints"]
    coco_evaluator = None
    if base_ds is not None:
        coco_evaluator = CocoEvaluator(base_ds, iou_types)

    # From openpifpaf
    if coco_evaluator is not None:
        CAR_SIGMAS = [.5] * num_keypoints
        coco_evaluator.set_scale(np.array(CAR_SIGMAS))

    len_dl = len(data_loader)
    pbar = tqdm(data_loader)
    pbar.set_description(f"Epoch {epoch}, loss = init")
    for i, (samples, targets) in enumerate(pbar):
        samples = samples.to(device)

        targets = [{k: v.to(device) if (v is not None) and (k not in ["image", "filename", "segmentation"]) else v for k, v in t.items()} for t in targets ]

        outputs = model(samples)
        if criterion is not None:
            loss_dict = criterion(outputs, targets)
            weight_dict = criterion.weight_dict

            losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)
            pbar.set_description(f"Epoch {epoch}, loss = {losses.item():.4f}")

            if logger is not None: 
                logger.add_scalar("Loss/train", losses.item(), len_dl*epoch + i)

        results = postprocessors['keypoints'](outputs, targets)

        results_outputs, results_targets = visualize(visualize_folder, targets, num_keypoints, device, postprocessors, outputs, results)
        results_outputs, results_targets = json_out(json_file, targets, device, postprocessors, outputs, results_outputs, results_targets)

        if coco_evaluator is not None:
            coco_evaluator.update_keypoints(results_outputs)

    if (coco_evaluator is not None) and (len(coco_evaluator.keypoint_predictions) > 0):
        coco_evaluator.synchronize_between_processes()
        coco_evaluator.accumulate()
        coco_evaluator.summarize()
        if logger is not None:
            stats = coco_evaluator.coco_eval['keypoints'].stats.tolist()
            logger.add_scalar("AP", stats[0], epoch)  # for the checkpoint callback monitor.
            logger.add_scalar("val/AP", stats[0],epoch)
            logger.add_scalar("val/AP.5", stats[1],epoch)
            logger.add_scalar("val/AP.75", stats[2],epoch)
            logger.add_scalar("val/AP.med", stats[3],epoch)
            logger.add_scalar("val/AP.lar", stats[4],epoch)
            logger.add_scalar("val/AR", stats[5],epoch)
            logger.add_scalar("val/AR.5", stats[6],epoch)
            logger.add_scalar("val/AR.75", stats[7],epoch)
            logger.add_scalar("val/AR.med", stats[8],epoch)
            logger.add_scalar("val/AR.lar", stats[9],epoch)
        return coco_evaluator
    return None

def get_targets_results(targets, postprocessors, device):
    if(all('keypoints' in t for t in targets)):
        num_quary = max([t['keypoints'].shape[0] for t in targets]) + 1
        targets_ = {
            'keypoints' : torch.cat([
            torch.cat((t['keypoints'], -torch.ones((num_quary -t['keypoints'].shape[0],t['keypoints'].shape[1]), device=device)), dim=0).unsqueeze(0)
            for t in targets], dim=0),
            'labels' : torch.cat([
                torch.cat([
                    torch.cat([torch.ones ((           t['keypoints'].shape[0], 1), device=device), torch.zeros((           t['keypoints'].shape[0], 1), device=device)], dim=1),
                    torch.cat([torch.zeros((num_quary -t['keypoints'].shape[0], 1), device=device), torch.ones ((num_quary -t['keypoints'].shape[0], 1), device=device)], dim=1)
                ], dim=0).unsqueeze(0)
                for t in targets], dim=0),
        }

        return postprocessors['keypoints'](targets_, targets)
    return None

def visualize(visualize_folder, targets, num_keypoints, device, postprocessors, outputs, results_outputs=None, results_targets=None):
    if visualize_folder is not None:
        if not os.path.exists(visualize_folder):
            os.makedirs(visualize_folder)

        if results_outputs is None:
            results_outputs = postprocessors['keypoints'](outputs, targets)
        for target in targets:
            filt =[out for out in results_outputs if out["image_id"] == target["image_id"]]
            plot_and_save_keypoints_inference(target["image"], f"output_{target['filename']}", filt, visualize_folder, num_keypoints)
            
        if results_targets is None:
            results_targets = get_targets_results(targets, postprocessors, device)
        if results_targets is not  None:
            for target in targets:
                filt =[out for out in results_targets if out['image_id'] == target['image_id']]
                plot_and_save_keypoints_inference(target['image'], f"target_{target['filename']}", filt, visualize_folder, num_keypoints)

    return results_outputs, results_targets

def json_out(json_file, targets, device, postprocessors, outputs, results_outputs=None, results_targets=None):
    if json_file is not None:
        if results_outputs is None:
            results_outputs = postprocessors['keypoints'](outputs, targets)
            
        json_ = {}
        for target in targets:
            filt =[out for out in results_outputs if out["image_id"] == target["image_id"]]
            json_ = convert_keypoints_inference(json_, target['filename'], target["image_id"], filt, 'output')

        if results_targets is None:
            results_targets = get_targets_results(targets, postprocessors, device)
        if results_targets is not  None:
            for target in targets:
                filt =[out for out in results_outputs if out["image_id"] == target["image_id"]]
                json_ = convert_keypoints_inference(json_, target['filename'], target["image_id"], filt, 'target')

        save_json(json_file, json_)
    return results_outputs, results_targets
        
def plot_and_save_keypoints_inference(img, image_name, data, output_folder,num_keypoints):
    img = img.copy()
    skeleton = CAR_SKELETON_24 if num_keypoints ==24 else CAR_SKELETON_66
    colors =  np.array([
       [0.12156863, 0.46666667, 0.70588235],
       [0.68235294, 0.78039216, 0.90980392],
       [1.        , 0.49803922, 0.05490196],
       [1.        , 0.73333333, 0.47058824],
       [0.17254902, 0.62745098, 0.17254902],
       [0.59607843, 0.8745098 , 0.54117647],
       [0.83921569, 0.15294118, 0.15686275],
       [1.        , 0.59607843, 0.58823529],
       [0.58039216, 0.40392157, 0.74117647],
       [0.54901961, 0.3372549 , 0.29411765],
       [0.76862745, 0.61176471, 0.58039216],
       [0.89019608, 0.46666667, 0.76078431],
       [0.96862745, 0.71372549, 0.82352941],
       [0.49803922, 0.49803922, 0.49803922],
       [0.78039216, 0.78039216, 0.78039216],
       [0.7372549 , 0.74117647, 0.13333333],
       [0.85882353, 0.85882353, 0.55294118],
       [0.09019608, 0.74509804, 0.81176471]]
    )

    for lst in data: 
      kps = lst["keypoints"]
      all_found_kps = [ ]
      all_kps_coordinate=[]
      for i in range(num_keypoints):
        x,y,z = tuple(kps[i*3:(i+1)*3])
        if z > 0:
          x= int(x)
          y = int(y)
          all_found_kps.append(int(i+1))
          all_kps_coordinate.append((x,y))
        else:
          all_kps_coordinate.append((-1,-1))
        
      set_of_pairs = set(itertools.permutations(all_found_kps,2))

      for idx, (a,b) in enumerate(skeleton):
        if (a,b) in set_of_pairs:
          r,g,bc = colors[idx%len(colors)]                  
          cv2.line(img,all_kps_coordinate[a-1], all_kps_coordinate[b-1],color=[int(bc*255),int(g*255),int(r*255)],thickness=10)

      for a in all_found_kps:
        r,g,bc = colors[a%len(colors)]
        cv2.circle(img, all_kps_coordinate[a-1],5, color=[int(bc*255),int(g*255),int(r*255)],thickness=-1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    cv2.imwrite(os.path.join(output_folder, image_name),img)

def convert_keypoints_inference(json_, image_name, image_id, data, type_):
    if type_ not in json_:
      json_[type_] = {'images':[], 'annotations':[]}
    json_[type_]['images'].append({'file_name':str(image_name), 'id':int(image_id)})
    json_[type_]['annotations'].extend(data)
    return json_

def save_json(json_file, json_):
    with open(json_file, 'w') as f:
        json.dump(json_, f)
