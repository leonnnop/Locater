# -*- coding: utf-8 -*-
import os
import time
import argparse
import os.path as osp
import sys
sys.path.append('.')

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.transforms import Compose

from network.mainnetwork import VLFTrans

from utils import AverageMeter

from dataloader.vid_anchor_test import ReferDataset_VID as ReferDataset_test
from dataloader.vid_anchor_test import test_collate_fn

from utils.transforms import Resize, ToTensor, Normalize

import numpy as np
import random

from dist_utils import *
from general_util import *

parser = argparse.ArgumentParser(
    description='Locater evaluation routine')

def load_args(parser):
    parser.add_argument('--data-root', type=str, default='./datasets/')
    parser.add_argument('--snapshot', default=None)
    # parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('-j', '--workers', default=16, type=int, metavar='N',
                        help='number of data loading workers (default: 16)')

    # Training procedure settings
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='Do not use cuda to train model')
    parser.add_argument('--log-interval', type=int, default=200, metavar='N',
                        help='report interval')
    parser.add_argument('--no-pin-memory', default=False, action='store_true',
                        help='enable CUDA memory pin on DataLoader')

    # Model settings
    parser.add_argument('--size', default=320, type=int,
                        help='image size')
    parser.add_argument("--in-chans", default=3, type=int)

    parser.add_argument('--N1', default=3, type=int)
    parser.add_argument('--N1_test', default=-1, type=int)

    # * for testing (temp, spat, mul) 
    parser.add_argument('--dataset', default='A2D', type=str)
    parser.add_argument('--testing-type', default='NORM', type=str)

    return parser

parser = load_args(parser)
args = parser.parse_args()

args.local_rank = int(os.environ["LOCAL_RANK"])

if args.N1_test == -1:
    args.N1_test = args.N1

args.distributed = False
if 'WORLD_SIZE' in os.environ:
    args.distributed = int(os.environ['WORLD_SIZE']) > 1
    sync_print('use distribute method', args)

args.world_size = 1
if args.distributed:
    torch.cuda.set_device(args.local_rank)
    torch.distributed.init_process_group(backend='nccl',
                                         init_method='env://')
    args.world_size = torch.distributed.get_world_size()

args_dict = vars(args)

args.cuda = not args.no_cuda and torch.cuda.is_available()

image_size = (args.size, args.size)

input_transform_val = Compose([
    ToTensor(),
    Resize(image_size, test=True),
    Normalize(
        mean=[0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5])
])

refer_val = ReferDataset_test(dataset_root=args.data_root,
                            transform=input_transform_val,
                            N1=args.N1_test,
                            dataset=args.dataset,
                            testing_type=args.testing_type)
val_sampler = None
if args.distributed:
    val_sampler = torch.utils.data.distributed.DistributedSampler(refer_val)
val_loader = DataLoader(refer_val, batch_size=1,
                        pin_memory=(not args.no_pin_memory),
                        shuffle=False,
                        sampler=val_sampler,
                        num_workers=args.workers,
                        collate_fn=test_collate_fn
                        )

sync_print('dataset loaded', args)

net = VLFTrans(img_dim=args.size, in_chans=args.in_chans)

assert osp.exists(args.snapshot)
sync_print('Loading state dict from: {0}'.format(args.snapshot), args)
snapshot_dict = torch.load(args.snapshot, map_location='cpu')
net.load_state_dict(snapshot_dict)

if args.distributed:
    net = torch.nn.SyncBatchNorm.convert_sync_batchnorm(net).cuda()
    # net = net.to(device)
    net = torch.nn.parallel.DistributedDataParallel(
        net,
        find_unused_parameters=True,
        device_ids=[args.local_rank],
        output_device=args.local_rank
    )
else: net = net.cuda()

sync_print('Argument list to program', args)
sync_print('\n'.join(['--{0} {1}'.format(arg, args_dict[arg])
                    for arg in args_dict]), args)
sync_print('\n\n', args)

def compute_mask_IU(masks, target, only_label=False):
    assert(target.shape[-2:] == masks.shape[-2:])
    temp = (masks * target)
    intersection = temp.sum()
    if only_label:
        union = target.sum()
    else:
        union = ((masks + target) - temp).sum()

    return intersection, union

def evaluate():
    net.eval()
    save_count = 0

    with torch.no_grad():
        eval_seg_iou_list = [.5, .6, .7, .8, .9]
        cum_I = 0
        cum_U = 0
        meaniou = 0
        seg_correct = torch.zeros(len(eval_seg_iou_list),1).cuda().squeeze()

        seg_total = torch.tensor([0.]).cuda()
        start_time = time.time()

        for seq_idx, (seq_dataset, global_images, words) in enumerate(val_loader):
            if seq_idx % (args.log_interval//args.world_size) == 0 or batch_idx == (len(val_loader) - 1):
                sync_print('Evaluating [{}+{}] {}/{} sequence....'.format(seq_dataset.seq_name, str(seq_dataset.obj_n), int(seq_idx),len(refer_val)//args.world_size), args)
            seq_dataloader=DataLoader(seq_dataset, batch_size=1, shuffle=False, num_workers=args.workers//args.world_size, pin_memory=True)

            if args.distributed:
                net.module._reset_memory()
            else:
                net._reset_memory()

            # * process global feature
            if args.cuda:
                global_images = global_images.cuda()
                for key in words:
                    words[key] = words[key].cuda()
            if args.distributed:
                net.module._prep_global_mem(global_images, words)
            else:
                net._prep_global_mem(global_images, words)

            # * 
            valid_labels = seq_dataset.labels

            for batch_idx, (imgs, mask) in enumerate(seq_dataloader):
                if (mask.min() != -1.):
                    if args.cuda:
                        imgs = imgs.cuda()
                        mask = mask.float().cuda()

                    out_masks, _attns = net(vis=imgs, lang=words)


                # * there exists no empty mask in A2D
                if mask.min() != -1.:

                    out_mask = out_masks[-1]

                    out = out_mask.squeeze()
                    out = torch.sigmoid(out)
                    
                    out = out.unsqueeze(0).unsqueeze(0)
                    out = F.interpolate(
                            out, size=(mask.shape[-2], mask.shape[-1]),
                            mode='bilinear', align_corners=True)
                    mask = mask.squeeze()

                    seg_total += 1

                    max_prob = 0.5

                    thresholded_out = (out > max_prob).float().data
                    iter, union = compute_mask_IU(thresholded_out, mask)
                    cum_I += iter
                    cum_U += union

                    # for temp setting empty mask
                    if union == 0:
                        iou = 1.
                    else:
                        iou = iter / union

                    meaniou += iou

                    for idx, seg_iou in enumerate(eval_seg_iou_list):
                        seg_correct[idx] += (iou >= seg_iou)


        # Print final accumulated IoUs
        if args.distributed:
            seg_total = reduce_tensor(seg_total, args)
            seg_correct = reduce_tensor(seg_correct, args)
            meaniou = reduce_tensor(meaniou, args)
            cum_I = reduce_tensor(cum_I, args)
            cum_U = reduce_tensor(cum_U, args)

        overall = cum_I / cum_U
        mean = meaniou / seg_total

        if args.local_rank == 0:
            print('precision@X for custom Threshold')
            for idx, seg_iou in enumerate(eval_seg_iou_list):
                rep_idx = eval_seg_iou_list.index(eval_seg_iou_list[idx])
                print('precision@{:s} = {:.5f}'.format(
                    str(seg_iou), float(seg_correct[rep_idx] / seg_total)))
            print('-' * 32)

            print('custom mAP.5:.95 = {:.5f}'.format(float(torch.mean(seg_correct)) / float(seg_total)))
            print('-' * 32)

        # Print maximum IoU
        if args.local_rank == 0:
            print('Evaluation done. Elapsed time: {:.3f} (s) '.format(
                time.time() - start_time))
            print('custom o-iou: {:<15.13f} | custom m-iou: {:<15.13f}'.format(float(overall), float(mean)))

    return float(overall), float(mean)

if __name__ == '__main__':
    evaluate()
