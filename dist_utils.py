import os
import shutil
import time

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import torch.distributed as torch_dist

import numpy as np

def reduce_tensor(tensor, args):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= args.world_size
    return rt

def sync_print(str_p, args):
    if args.local_rank==0:
        print(str_p)

def is_distributed():
    return torch_dist.is_initialized()

def get_world_size():
    if not torch_dist.is_initialized():
        return 1
    return torch_dist.get_world_size()

def get_rank():
    if not torch_dist.is_initialized():
        return 0
    return torch_dist.get_rank()