# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
DETR model and criterion classes.
"""
import torch
import torch.nn.functional as F
from torch import nn

from util.misc import (NestedTensor, accuracy, get_world_size, interpolate,
                       is_dist_avail_and_initialized,
                       nested_tensor_from_tensor_list)

from .backbone import build_backbone
from .matcher import build_matcher
from .segmentation import (DETRsegm, PostProcessPanoptic, PostProcessSegm,
                           dice_loss, sigmoid_focal_loss)
from .transformer import build_transformer


class CARPE(nn.Module):
	""" This is the DETR module that performs object detection """
	def __init__(self, backbone, transformer, num_classes, num_queries, num_keypoints, aux_loss=False):
		""" Initializes the model.
		Parameters:
			backbone: torch module of the backbone to be used. See backbone.py
			transformer: torch module of the transformer architecture. See transformer.py
			num_classes: number of object classes
			num_queries: number of object queries, ie detection slot. This is the maximal number of objects
						 DETR can detect in a single image. For COCO, we recommend 100 queries.
			aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
		"""
		super().__init__()
		self.num_queries = num_queries
		self.transformer = transformer
		hidden_dim = transformer.d_model
		self.num_keypoints = num_keypoints
		num_body_parts = self.num_keypoints*3 + 2
		self.class_embed_out = nn.Linear(hidden_dim, num_classes + 1)
		self.pose_embed = MLP(hidden_dim, hidden_dim, num_body_parts, 3)
		self.query_embed = nn.Embedding(num_queries, hidden_dim)
		self.input_proj = nn.Conv2d(backbone.num_channels, hidden_dim, kernel_size=1)
		self.backbone = backbone
		self.aux_loss = aux_loss


	def forward(self, samples: NestedTensor):
		""" The forward expects a NestedTensor, which consists of:
			   - samples.tensor: batched images, of shape [batch_size x 3 x H x W]
			   - samples.mask: a binary mask of shape [batch_size x H x W], containing 1 on padded pixels

			It returns a dict with the following elements:
			   - "labels": the classification logits (including no-object) for all queries.
								Shape= [batch_size x num_queries x (num_classes + 1)]
			   - "keypoints": The normalized boxes coordinates for all queries, represented as
							   (center_x, center_y, height, width). These values are normalized in [0, 1],
							   relative to the size of each individual image (disregarding possible padding).
							   See PostProcess for information on how to retrieve the unnormalized bounding box.
			   - "aux_outputs": Optional, only returned when auxilary losses are activated. It is a list of
								dictionnaries containing the two above keys for each decoder layer.
		"""
		if isinstance(samples, (list, torch.Tensor)):
			samples = nested_tensor_from_tensor_list(samples)
		features, pos = self.backbone(samples)

		src, mask = features[-1].decompose()
		assert mask is not None
	
		hs = self.transformer(self.input_proj(src), mask, self.query_embed.weight, pos[-1])[0]

		outputs_class = self.class_embed_out(hs)
		outputs_keypoints = self.pose_embed(hs)	

		out = {'labels': outputs_class[-1], 'keypoints': outputs_keypoints[-1]}
		return out

class SetCriterion(nn.Module):
	""" This class computes the loss for DETR.
	The process happens in two steps:
		1) we compute hungarian assignment between ground truth boxes and the outputs of the model
		2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
	"""
	def __init__(self, num_classes, matcher, weight_dict, eos_coef, losses, num_keypoints):
		""" Create the criterion.
		Parameters:
			num_classes: number of object categories, omitting the special no-object category
			matcher: module able to compute a matching between targets and proposals
			weight_dict: dict containing as key the names of the losses and as values their relative weight.
			eos_coef: relative classification weight applied to the no-object category
			losses: list of all the losses to be applied. See get_loss for list of available losses.
		"""
		super().__init__()
		self.num_keypoints = num_keypoints
		self.num_classes = num_classes
		self.matcher = matcher
		self.weight_dict = weight_dict
		self.eos_coef = eos_coef
		self.losses = losses
		empty_weight = torch.ones(self.num_classes + 1)
		empty_weight[-1] = self.eos_coef
		self.register_buffer('empty_weight', empty_weight)
		self.l_deltas = 0.5
		self.l_vis = 0.2
		self.l_ctr = 0.5
		self.l_abs = 4

	def loss_labels(self, outputs, targets, indices, num_queries, log=True):
		"""Classification loss (NLL)
		targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
		"""
		assert 'labels' in outputs
		src_logits = outputs['labels']

		idx = self._get_src_permutation_idx(indices)
		target_classes_o = torch.cat([t['labels'][J] for t, (_, J) in zip(targets, indices)])
		target_classes = torch.full(src_logits.shape[:2], self.num_classes,
									dtype=torch.int64, device=src_logits.device)
		target_classes[idx] = target_classes_o

		loss_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes, self.empty_weight)
		losses = {'loss_labels': loss_ce}
		return losses

	def loss_keypoints(self, outputs, targets, indices, num_queries):
		"""Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
		   targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
		   The target boxes are expected in format (center_x, center_y, w, h), normalized by the image size.
		"""
		#assert 'keypoints' in outputs
		idx = self._get_src_permutation_idx(indices)
		src_boxes = outputs['keypoints'][idx]

		C_pred = src_boxes[:, :2]
		Z_pred = src_boxes[:, 2:2+self.num_keypoints*2]
		V_pred = src_boxes[:, 2+self.num_keypoints*2:]
	
		targets_keypoints = torch.cat([t['keypoints'][i] for t, (_, i) in zip(targets, indices)], dim=0)
		C_gt = targets_keypoints[:, :2]
		Z_gt = targets_keypoints[:, 2:2+self.num_keypoints*2]
		V_gt = targets_keypoints[:, 2+self.num_keypoints*2:]

		C_gt_expand = torch.repeat_interleave(C_gt.unsqueeze(1), self.num_keypoints, dim=1).view(-1,self.num_keypoints*2)
		A_gt = C_gt_expand + Z_gt
		C_pred_expand = torch.repeat_interleave(C_pred.unsqueeze(1), self.num_keypoints, dim=1).view(-1,self.num_keypoints*2)
		A_pred = C_pred_expand + Z_pred

		# Compute the L1 and L2 regression loss
		Vgt_ = torch.repeat_interleave(V_gt , 2, dim=1)
		offset_loss =  F.l1_loss(Vgt_  * Z_pred, Vgt_ * Z_gt, reduction = 'none')
		viz_loss  =  F.mse_loss(V_pred, V_gt, reduction = 'none')
		center_loss =  F.mse_loss(C_pred , C_gt, reduction='none')
		abs_loss = F.l1_loss(Vgt_ * A_pred, Vgt_ * A_gt, reduction='none')

		total_keypoints_loss = self.l_deltas * torch.sum(offset_loss) + self.l_vis *  torch.sum(viz_loss) + self.l_ctr *  torch.sum(center_loss) + self.l_abs * torch.sum(abs_loss)

		losses = {'loss_keypoints' : total_keypoints_loss / num_queries}

		return losses
	
	def _get_src_permutation_idx(self, indices):
		# permute predictions following indices
		batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
		src_idx = torch.cat([src for (src, _) in indices])
		return batch_idx, src_idx

	def _get_tgt_permutation_idx(self, indices):
		# permute targets following indices
		batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
		tgt_idx = torch.cat([tgt for (_, tgt) in indices])
		return batch_idx, tgt_idx

	def get_loss(self, loss, outputs, targets, indices, num_queries, **kwargs):
		loss_map = {
			'labels': self.loss_labels,
			'keypoints': self.loss_keypoints,
		}
		assert loss in loss_map, f'do you really want to compute {loss} loss?'
		return loss_map[loss](outputs, targets, indices, num_queries, **kwargs)

	def forward(self, outputs, targets):
		""" This performs the loss computation.
		Parameters:
			 outputs: dict of tensors, see the output specification of the model for the format
			 targets: list of dicts, such that len(targets) == batch_size.
					  The expected keys in each dict depends on the losses applied, see each loss' doc
		"""
		outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs'}
		
		# Compute the average number of target boxes accross all nodes, for normalization purposes
		num_queries = sum(len(t['labels']) for t in targets)
		num_queries = torch.as_tensor([num_queries], dtype=torch.float, device=next(iter(outputs.values())).device)
		if is_dist_avail_and_initialized():
			torch.distributed.all_reduce(num_queries)
		num_queries = torch.clamp(num_queries / get_world_size(), min=1).item()

		# Retrieve the matching between the outputs of the last layer and the targets
		indices = self.matcher(outputs_without_aux, targets)

		# Compute all the requested losses
		losses = {}
		for loss in self.losses:
			losses.update(self.get_loss(loss, outputs, targets, indices, num_queries))

		return losses

class PostProcess(nn.Module):
	""" This module converts the model's output into the format expected by the coco api"""
	def __init__(self, num_keypoints):
		super().__init__()
		self.num_keypoints = num_keypoints

	@torch.no_grad()
	def forward(self, outputs, targets):
		""" Perform the computation
		Parameters:
			outputs: raw outputs of the model
			target: tensor of dimension [batch_size x 2] containing the size of each images of the batch
						  For evaluation, this must be the original image size (before any data augmentation)
						  For visualization, this should be the image size after data augment, but before padding
		"""
		out_logits, out_keypoints = outputs['labels'], outputs['keypoints']
		threshold = 0.5

		target_sizes = torch.stack([torch.as_tensor([t["orig_size"][1], t["orig_size"][0]], device=t["orig_size"].device) for t in targets], dim=0)

		scores = out_logits.softmax(-1)

		C_pred = out_keypoints[:, :, :2] # shape (bs, N, 2)
		Z_pred = out_keypoints[:, :, 2:self.num_keypoints*2+2]  # shape (bs, N, 2*num_keypoints)
		V_pred = out_keypoints[:, :, self.num_keypoints*2+2:] 	# shape (bs, N, num_keypoints)

		target_sizes = target_sizes.expand(Z_pred.shape[1], Z_pred.shape[2]//target_sizes.shape[1], *target_sizes.shape)\
			.permute(2,0,1,3).reshape(Z_pred.shape)
		
		V_pred = torch.repeat_interleave(V_pred, 2, dim=2)
		C_pred_expand = C_pred.expand(self.num_keypoints, *C_pred.shape).permute(1,2,0,3).reshape(Z_pred.shape) * target_sizes
		A_pred = Z_pred * target_sizes

		positions = C_pred_expand + A_pred
		positions[V_pred < threshold] = -1

		image_ids = torch.stack([t["image_id"] for t in targets], dim=0).squeeze(1).cpu().numpy()
		
		results = []
		for image_id, scores_, positions_ in zip(image_ids, scores, positions):
			for s, position in zip(scores_, positions_):
				score, category = s.topk(1)
				score = score.item()
				category = category.item() + 1
				if category == out_logits.shape[-1] or score < threshold:
					continue
				nbr_keypoints = 0
				keypoints = [0.] * self.num_keypoints * 3

				for i in range(self.num_keypoints):
					x, y = position[i*2:(i+1)*2]
					if x >= 0 and y >= 0:
						nbr_keypoints += 1
						keypoints[i*3:(i+1)*3] = [x.item(), y.item(), 2]
				results.append(
					{'image_id': image_id, 'category_id': category, 'score': score, "nbr_keypoints": nbr_keypoints, "keypoints": list(keypoints)}
				)
		return  results

class MLP(nn.Module):
	""" Very simple multi-layer perceptron (also called FFN)"""

	def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
		super().__init__()
		self.num_layers = num_layers
		h = [hidden_dim] * (num_layers - 1)
		self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

	def forward(self, x):
		for i, layer in enumerate(self.layers):
			x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
		return x


def build(args):
	# the `num_classes` naming here is somewhat misleading.
	# it indeed corresponds to `max_obj_id + 1`, where max_obj_id
	# is the maximum id for a class in your dataset. For example,
	# COCO has a max_obj_id of 90, so we pass `num_classes` to be 91.
	# As another example, for a dataset that has a single class with id 1,
	# you should pass `num_classes` to be 2 (max_obj_id + 1).
	# For more details on this, check the following discussion
	# https://github.com/facebookresearch/detr/issues/108#issuecomment-650269223
    
	num_classes = 1
	
	device = torch.device(args.device)

	backbone = build_backbone(args)

	transformer = build_transformer(args)

	model = CARPE(
		backbone,
		transformer,
		num_classes=num_classes,
		num_queries=args.num_queries,
		num_keypoints=args.num_keypoints,
		aux_loss=args.aux_loss,
	)
	matcher = build_matcher(args)
	weight_dict = {'loss_labels': args.set_cost_class, 'loss_keypoints': args.set_cost_keypoints}

	losses = ['labels', 'keypoints']
	criterion = SetCriterion(num_classes, matcher=matcher, weight_dict=weight_dict, num_keypoints=args.num_keypoints,
							 eos_coef=args.eos_coef, losses=losses)
	criterion.to(device)
	postprocessors = {'keypoints': PostProcess(args.num_keypoints)}

	return model, criterion, postprocessors
