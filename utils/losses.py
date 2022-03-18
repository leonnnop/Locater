# -*- coding: utf-8 -*-

"""
Custom loss function definitions.
"""
import os
import torch.nn as nn
import torch.nn.functional as F
import torch
from general_util import save_p_img
import numpy as np
try:
    from itertools import ifilterfalse
except ImportError:  # py3k
    from itertools import filterfalse as ifilterfalse



class BootstrappedBCEWithLogitsLoss(nn.Module):
    def __init__(self, top_k_percent_pixels=None,
                 hard_example_mining_step=100000, **kwargs):
        super(BootstrappedBCEWithLogitsLoss, self).__init__()
        self.top_k_percent_pixels = top_k_percent_pixels
        if top_k_percent_pixels is not None:
            assert(top_k_percent_pixels > 0 and top_k_percent_pixels < 1)
        self.hard_example_mining_step = hard_example_mining_step
        self.bceloss = nn.BCEWithLogitsLoss(reduction='none')
        self.step = 0

    def _set_step(self):
        self.step += 1

    def forward(self, pred_logits, gts):
        # * B H,W

        # pred_logits = torch.sigmoid(output)

        num_pixels = float(pred_logits.size(1) * pred_logits.size(2))
        pred_logits = pred_logits.view(pred_logits.size(0), pred_logits.size(1) * pred_logits.size(2))
        gts = gts.view(gts.size(0), gts.size(1) * gts.size(2))

        pixel_losses = self.bceloss(pred_logits, gts) # * B,H*W
        # print(pixel_losses.shape)

        if self.hard_example_mining_step == 0:
            top_k_pixels = int(self.top_k_percent_pixels * num_pixels)
        else:
            ratio = min(
                1.0, self.step / float(self.hard_example_mining_step))
            top_k_pixels = int(
                (ratio * self.top_k_percent_pixels + (1.0 - ratio)) * num_pixels)

        top_k_loss, top_k_indices = torch.topk(
                        pixel_losses, k=top_k_pixels, dim=1)

        final_loss = torch.mean(top_k_loss)

        self._set_step()

        return final_loss


class BCEWithLogitsLoss(nn.Module):

    def __init__(self, **kwargs):
        super(BCEWithLogitsLoss, self).__init__()
        self.loss = nn.BCEWithLogitsLoss()

    def forward(self, output, target):
        loss = self.loss(output, target)
        return loss

class BinaryFocalLoss(nn.Module):

    def __init__(self, alpha=3, gamma=2, ignore_index=None, reduction='mean',**kwargs):
        super(BinaryFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.smooth = 1e-4 # set '1e-4' when train with FP16
        self.ignore_index = ignore_index
        self.reduction = reduction

        assert self.reduction in ['none', 'mean', 'sum']

    def forward(self, output, target):

        prob = torch.sigmoid(output)
        prob = torch.clamp(prob, self.smooth, 1.0 - self.smooth)

        valid_mask = None
        if self.ignore_index is not None:
            valid_mask = (target != self.ignore_index).float()

        pos_mask = (target == 1).float()
        neg_mask = (target == 0).float()
        if valid_mask is not None:
            pos_mask = pos_mask * valid_mask
            neg_mask = neg_mask * valid_mask

        pos_weight = (pos_mask * torch.pow(1 - prob, self.gamma)).detach()
        pos_loss = -torch.sum(pos_weight * torch.log(prob)) / (torch.sum(pos_weight) + 1e-4)
        
        
        neg_weight = (neg_mask * torch.pow(prob, self.gamma)).detach()
        neg_loss = -self.alpha * torch.sum(neg_weight * F.logsigmoid(-output)) / (torch.sum(neg_weight) + 1e-4)
        loss = pos_loss + neg_loss

        return loss


class BalancedBCEWithLogitsLoss(nn.Module):

    def __init__(self, fore=1.5, back=0.5, ignore_index=None, reduction='mean', **kwargs):
        super(BalancedBCEWithLogitsLoss, self).__init__()
        self.fore = fore
        self.back = back
        self.smooth = 1e-4 # set '1e-4' when train with FP16
        self.ignore_index = ignore_index
        self.reduction = reduction

        assert self.reduction in ['none', 'mean', 'sum']

    def forward(self, output, target):

        prob = torch.sigmoid(output)
        prob = torch.clamp(prob, self.smooth, 1.0 - self.smooth)

        valid_mask = None
        if self.ignore_index is not None:
            valid_mask = (target != self.ignore_index).float()

        pos_mask = (target == 1).float()
        neg_mask = (target == 0).float()
        if valid_mask is not None:
            pos_mask = pos_mask * valid_mask
            neg_mask = neg_mask * valid_mask

        pos_weight = (pos_mask * self.fore).detach()
        pos_loss = -torch.sum(pos_weight * torch.log(prob)) / (torch.sum(pos_weight) + 1e-4)
        
        
        neg_weight = (neg_mask * self.back).detach()
        neg_loss = -torch.sum(neg_weight * F.logsigmoid(-output)) / (torch.sum(neg_weight) + 1e-4)
        loss = pos_loss + neg_loss

        return loss