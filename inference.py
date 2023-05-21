# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import argparse
import datetime
import json
import random
import time
from pathlib import Path
import os
import numpy as np
import torch
from torch.utils.data import DataLoader, DistributedSampler
from datetime import datetime
import datasets
import util.misc as utils
from datasets import build_dataset, get_coco_api_from_dataset
from engine import evaluate, train_one_epoch
from models import build_model
from torch.utils.tensorboard import SummaryWriter
from datasets.inference_dataset import build_inference

import logging


def get_args_parser():
    parser = argparse.ArgumentParser('Set transformer detector', add_help=False)
    parser.add_argument("pretrained_weight_path")
    parser.add_argument("image_folder",type=str)
    parser.add_argument('--coco_file_path', type=str)
    parser.add_argument("-j",help="If we should write json output to disk",action="store_true")
    parser.add_argument("--viz", help="If we should display keypoints on images",action="store_true")
    parser.add_argument("--inference_out_folder",type=str, help="The folder in which we should store the output")

    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--lr_backbone', default=1e-5, type=float)
    parser.add_argument('--batch_size', default=2, type=int)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--epochs', default=300, type=int)
    parser.add_argument('--lr_drop', default=200, type=int)
    parser.add_argument('--clip_max_norm', default=0.1, type=float,
                        help='gradient clipping max norm')


    # * Backbone
    parser.add_argument('--backbone', default='resnet50', type=str,
                        help="Name of the convolutional backbone to use")
    parser.add_argument('--dilation', action='store_true',
                        help="If true, we replace stride with dilation in the last convolutional block (DC5)")
    parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'),
                        help="Type of positional embedding to use on top of the image features")

    # * Transformer
    parser.add_argument('--enc_layers', default=6, type=int,
                        help="Number of encoding layers in the transformer")
    parser.add_argument('--dec_layers', default=6, type=int,
                        help="Number of decoding layers in the transformer")
    parser.add_argument('--dim_feedforward', default=2048, type=int,
                        help="Intermediate size of the feedforward layers in the transformer blocks")
    parser.add_argument('--hidden_dim', default=256, type=int,
                        help="Size of the embeddings (dimension of the transformer)")
    parser.add_argument('--dropout', default=0.1, type=float,
                        help="Dropout applied in the transformer")
    parser.add_argument('--nheads', default=8, type=int,
                        help="Number of attention heads inside the transformer's attentions")
    parser.add_argument('--num_queries', default=25, type=int,
                        help="Number of query slots")
    parser.add_argument('--pre_norm', action='store_true')

    # * Matcher
    parser.add_argument('--set_cost_class', default=1, type=float,
                        help="Class coefficient in the matching cost")
    parser.add_argument('--set_cost_bbox', default=5, type=float,
                        help="L1 box/keypoints/links coefficient in the matching cost")
    parser.add_argument('--set_cost_giou', default=2, type=float,
                        help="giou box coefficient in the matching cost")
    # * Loss coefficients
    parser.add_argument('--mask_loss_coef', default=1, type=float)
    parser.add_argument('--dice_loss_coef', default=1, type=float)
    parser.add_argument('--bbox_loss_coef', default=5, type=float)
    parser.add_argument('--giou_loss_coef', default=2, type=float)
    parser.add_argument('--eos_coef', default=0.1, type=float,
                        help="Relative classification weight of the no-object class")

    # dataset parameters
    parser.add_argument('--dataset_file', default='coco')
    parser.add_argument('--remove_difficult', action='store_true')

    parser.add_argument('--output_dir', default=os.path.join("models","model_saves"),
                        help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--pretrained_detr',  help='resume from pretrained detr', action="store_true")
    parser.add_argument("--pretrained_keypoints",  help='resume from pretrained keypoints detector', action="store_true")
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--num_workers', default=2, type=int)
    parser.add_argument("--calibration_epochs", default=1,type=int)
    parser.add_argument("--apply_augmentation", action="store_true", help="If we apply the data augmentation")
    parser.add_argument("--apply_occlusion_augmentation", action="store_true", help="If we should apply the occlusion augmentation")
    parser.add_argument("--input_image_resize",default=(480,640),type=tuple)

    return parser


def main(args):
    logging.basicConfig(level=10)
    log = logging.getLogger("g21")

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    model, criterion, postprocessors = build_model(args)

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info(f'Number of trainable parameters {n_parameters}' )

    dataset_val = CocoDataset(args.image_folder, args.coco_file_path, make_coco_transforms("val",args.input_image_resize,False), None, None, None, None, "val")
         if args.coco_file_path is not None else 
         build_inference(args)

    sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    data_loader_val = DataLoader(dataset_val, args.batch_size, sampler=sampler_val,
                                 drop_last=False, collate_fn=utils.collate_fn, num_workers=args.num_workers)

    if not os.path.exists(args.inference_out_folder):
        os.makedirs(args.inference_out_folder)

    if os.path.exists(args.pretrained_weight_path):
        model_ckpt = torch.load(args.pretrained_weight_path)["model"]
        model_state = model.state_dict()
        pretrained_state = { k:v for k,v in model_ckpt.items() if k in model_state and v.size() == model_state[k].size() }
        for k,v in model.state_dict().items():
            if k not in pretrained_state:
                log.info("The following key "+k+" has not been found in the pretrained dictionary.")
                pretrained_state[k] = v

        model_state.update(pretrained_state)
        model.load_state_dict(model_state)
    else:
        raise ValueError("The given weight path doesn't exist")

    model.to(device)
    base_ds = None
    if args.coco_file_path is not None:
        base_ds = get_coco_api_from_dataset(dataset_val)

    coco_evaluator = evaluate(model, 
                            criterion, 
                            postprocessors,
                            data_loader_val,
                            base_ds,
                            device,
                            args.output_dir,
                            num_keypoints=args.num_keypoints,
                            visualize_keypoints=args.viz,
                            out_folder=args.inference_out_folder)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('CARPE training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
