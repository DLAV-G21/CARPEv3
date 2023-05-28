# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import argparse
import json
import logging
import os
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

import util.misc as utils
from datasets import get_coco_api_from_dataset
from datasets.coco import CocoDetection, make_coco_transforms
from datasets.inference_dataset import build_inference
from engine import evaluate
from models import build_model


def get_args_parser():
    parser = argparse.ArgumentParser('Set transformer detector', add_help=False)
    parser.add_argument('pretrained_weight_path',type=str)
    parser.add_argument('image',type=str)
    parser.add_argument('--coco_file_path', type=str)
    parser.add_argument('-j', '--json_file',type=str,help='Path to the json output file')
    parser.add_argument('-v','--visualize_folder',type=str, help='The folder in which we should store the output images')

    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--lr_backbone', default=1e-5, type=float)
    parser.add_argument('-b', '--batch_size', default=2, type=int)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--epochs', default=300, type=int)
    parser.add_argument('--lr_drop', default=200, type=int)
    parser.add_argument('--clip_max_norm', default=0.1, type=float,
                        help='gradient clipping max norm')


    # Model parameters
    parser.add_argument('--frozen_weights', type=str, default=None,
                        help='Path to the pretrained model. If set, only the mask head will be trained')
    # * Backbone
    parser.add_argument('--backbone', default='resnet50', type=str,
                        help='Name of the convolutional backbone to use')
    parser.add_argument('--dilation', action='store_true',
                        help='If true, we replace stride with dilation in the last convolutional block (DC5)')
    parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'),
                        help='Type of positional embedding to use on top of the image features')


    # * Transformer
    parser.add_argument('--enc_layers', default=6, type=int,
                        help='Number of encoding layers in the transformer')
    parser.add_argument('--dec_layers', default=6, type=int,
                        help='Number of decoding layers in the transformer')
    parser.add_argument('--dim_feedforward', default=2048, type=int,
                        help='Intermediate size of the feedforward layers in the transformer blocks')
    parser.add_argument('--hidden_dim', default=256, type=int,
                        help='Size of the embeddings (dimension of the transformer)')
    parser.add_argument('--dropout', default=0.1, type=float,
                        help='Dropout applied in the transformer')
    parser.add_argument('--nheads', default=8, type=int,
                        help='Number of attention heads inside the transformer\'s attentions')
    parser.add_argument('--num_queries', default=50, type=int,
                        help='Number of query slots')
    parser.add_argument('--pre_norm', action='store_true')

    # * Segmentation
    parser.add_argument('--masks', action='store_true',
                        help='Train segmentation head if the flag is provided')
    # Loss
    parser.add_argument('--no_aux_loss', dest='aux_loss', action='store_false',
                        help='Disables auxiliary decoding losses (loss at each layer)')
    # * Matcher
    parser.add_argument('--set_cost_class', default=10, type=float,
                        help='Class coefficient in the matching cost')
    parser.add_argument('--set_cost_keypoints', default=1, type=float,
                        help='L1 keypoints coefficient in the matching cost')
    # * Loss coefficients
    parser.add_argument('--mask_loss_coef', default=1, type=float)
    parser.add_argument('--dice_loss_coef', default=1, type=float)
    parser.add_argument('--bbox_loss_coef', default=5, type=float)
    parser.add_argument('--giou_loss_coef', default=2, type=float)
    parser.add_argument('--eos_coef', default=0.1, type=float,
                        help='Relative classification weight of the no-object class')

    # for keypoints
    parser.add_argument('-k', '--num_keypoints', default=24, type=int,
                        help='number of keypoints')
    
    # dataset parameters
    parser.add_argument('--dataset_file', default='coco')
    parser.add_argument('--remove_difficult', action='store_true')
    
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='device to use for training / testing')
    parser.add_argument('--pretrained_keypoints',  help='resume from pretrained keypoints detector', action='store_true')
    parser.add_argument('--num_workers', default=2, type=int)
    parser.add_argument('--input_image_resize',default=500,type=float)
    parser.add_argument('-t', '--threshold',default=0.5,type=float)
    parser.add_argument('--threshold_keypoints',default=0.5,type=float)
    parser.add_argument('--threshold_iou',default=0.5,type=float)

    return parser


def main(args):
    logging.basicConfig(level=10)
    log = logging.getLogger('g21')

    device = torch.device(args.device)

    model, criterion, postprocessors = build_model(args)

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info(f'Number of trainable parameters {n_parameters}' )

    dataset_val = CocoDetection(args.image, args.coco_file_path, make_coco_transforms('test'), None, None, None, None, None, 'test')\
         if args.coco_file_path is not None else \
         build_inference(args)

    sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    data_loader_val = DataLoader(dataset_val, args.batch_size, sampler=sampler_val,
                                 drop_last=False, collate_fn=utils.collate_fn, num_workers=args.num_workers)

    if os.path.exists(args.pretrained_weight_path):
        log.info('Loading pretrained weights from '+args.pretrained_weight_path)
        pretrained_state = torch.load(args.pretrained_weight_path)['model']
        model_state = model.state_dict()
        for k,v in model.state_dict().items():
            if k not in pretrained_state:
                log.info('The following key '+k+' has not been found in the pretrained dictionary.')
                pretrained_state[k] = v

        model_state.update(pretrained_state)
        model.load_state_dict(model_state)
    else:
        raise ValueError('The given weight path doesn\'t exist')

    model.to(device)

    coco_evaluator = evaluate(
        model, criterion if args.coco_file_path is not None else None, postprocessors,
        data_loader_val,
        get_coco_api_from_dataset(dataset_val) if args.coco_file_path is not None else None,
        device, num_keypoints=args.num_keypoints,
        visualize_folder=args.visualize_folder,
        json_file=args.json_file,
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser('CARPE training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)
