# -*- coding: utf-8 -*-

"""
Generic Image Transform utillities.
"""

import cv2
import numpy as np
from collections import Iterable

import torch.nn.functional as F
from torch.autograd import Variable
from torchvision.transforms import functional as tF

import torch
from general_util import save_p_img, save_rgb_img

import random


class ResizePad:
    """
    Resize and pad an image to given size.
    """

    def __init__(self, size):
        if not isinstance(size, (int, Iterable)):
            raise TypeError('Got inappropriate size arg: {}'.format(size))

        self.h, self.w = size

    def __call__(self, img):
        h, w = img.shape[:2]
        scale = min(self.h / h, self.w / w)
        resized_h = int(np.round(h * scale))
        resized_w = int(np.round(w * scale))
        pad_h = int(np.floor(self.h - resized_h) / 2)
        pad_w = int(np.floor(self.w - resized_w) / 2)

        resized_img = cv2.resize(img, (resized_w, resized_h))

        # if img.ndim > 2:
        if img.ndim > 2:
            new_img = np.zeros(
                (self.h, self.w, img.shape[-1]), dtype=resized_img.dtype)
        else:
            resized_img = np.expand_dims(resized_img, -1)
            new_img = np.zeros((self.h, self.w, 1), dtype=resized_img.dtype)
        new_img[pad_h: pad_h + resized_h,
                pad_w: pad_w + resized_w, ...] = resized_img
        return new_img


class CropResize:
    """Remove padding and resize image to its original size."""

    def __call__(self, img, size):
        if not isinstance(size, (int, Iterable)):
            raise TypeError('Got inappropriate size arg: {}'.format(size))
        im_h, im_w = img.data.shape[:2]
        input_h, input_w = size
        scale = max(input_h / im_h, input_w / im_w)
        resized_h = int(np.round(im_h * scale))
        resized_w = int(np.round(im_w * scale))
        crop_h = int(np.floor(resized_h - input_h) / 2)
        crop_w = int(np.floor(resized_w - input_w) / 2)
        resized_img = F.interpolate(
            img.unsqueeze(0).unsqueeze(0), size=(resized_h, resized_w),
            mode='bilinear', align_corners=True)

        resized_img = resized_img.squeeze().unsqueeze(0)

        return resized_img[0, crop_h: crop_h + input_h,
                           crop_w: crop_w + input_w]

class Resize(object):
    """Resize the largest of the sides of the image to a given size"""
    def __init__(self, size, test=False):
        if not isinstance(size, (int, Iterable)):
            raise TypeError('Got inappropriate size arg: {}'.format(size))
        
        self.test = test
        self.size_h, self.size_w = size

    def __call__(self, input):
        phrase, img, mask = input

        img = F.interpolate(
            img, size=(self.size_h, self.size_w),
            mode='bilinear', align_corners=True)
            
        if not self.test:
            # * only resize mask while easy training statistic
            mask = mask.unsqueeze(0)
            mask = F.interpolate(
                mask,
                size=(self.size_h, self.size_w),
                mode='nearest')
            mask = mask.squeeze()

        return (phrase, img, mask)

class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""
    def __init__(self):
        pass
    def __call__(self, input):
        phrase, img, mask = input

        if isinstance(img, np.ndarray):
            img = img.transpose((0, 3, 1, 2))

            img = (torch.from_numpy(img)).contiguous().float()
            # if isinstance(img, torch.ByteTensor):
        if img.max() > 1.:
            img = img.div(255)

        if isinstance(mask, np.ndarray):
            mask = torch.from_numpy(mask).float()

        return (phrase, img, mask)

class ResizeAnnotation:
    """Resize the largest of the sides of the annotation to a given size"""
    def __init__(self, size, padding_width):
        if not isinstance(size, (int, Iterable)):
            raise TypeError('Got inappropriate size arg: {}'.format(size))

        self.final_size = size
        # self.padding = 20
        # self.size = size - self.padding
        self.padding_width = padding_width
        self.padding = (padding_width,padding_width,padding_width,padding_width)
        self.size = size - 2*self.padding[0]
        self.value = 0.

    def __call__(self, img):
        # print(img.shape)
        im_h, im_w = img.shape[-2:]
        if img.ndim == 3:
            squeeze_flag = False
        else:
            squeeze_flag = True

        while len(img.shape) < 4:
            img = img.unsqueeze(0)
        out = F.interpolate(
            img,
            size=(self.size, self.size),
            mode='nearest').squeeze(0)
        if squeeze_flag:
            out = out.squeeze(0)
        if squeeze_flag:
            out = out.squeeze()
        if self.padding_width > 0:
            out = F.pad(out, self.padding, 'constant', self.value)
        # print(out.shape)

        return out

class Normalize(object):

    def __init__(self, mean, std, inplace=False):
        super().__init__()
        self.mean = mean
        self.std = std
        self.inplace = inplace

    def __call__(self, input):
        phrase, tensor, mask = input

        dtype = tensor.dtype
        mean = torch.as_tensor(self.mean, dtype=dtype, device=tensor.device)
        std = torch.as_tensor(self.std, dtype=dtype, device=tensor.device)

        if tensor.dim() == 3:
            if mean.ndim == 1:
                mean = mean.view(-1, 1, 1)
            if std.ndim == 1:
                std = std.view(-1, 1, 1)

        elif tensor.dim() == 4:
            if mean.ndim == 1:
                mean = mean.view(1, -1, 1, 1)
            if std.ndim == 1:
                std = std.view(1, -1, 1, 1)

        tensor.sub_(mean).div_(std)

        return (phrase, tensor, mask)

class ToNumpy:
    """Transform an torch.*Tensor to an numpy ndarray."""

    def __call__(self, x):
        return x.numpy()

class BalancedRandomCropT(object):

    def __init__(self, output_size, max_step=5, max_obj_num=5, min_obj_pixel_num=100):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size
        self.max_step = max_step
        self.max_obj_num = max_obj_num
        self.min_obj_pixel_num = min_obj_pixel_num

    def __call__(self, input):
        img, mask, instance = input
        h, w = img.shape[-2:]
        new_h, new_w = self.output_size
        new_h = h if new_h >= h else new_h
        new_w = w if new_w >= w else new_w

        is_contain_obj = False
        step = 0
        while (not is_contain_obj) and (step < self.max_step):
            step += 1
            top = np.random.randint(0, h - new_h + 1)
            left = np.random.randint(0, w - new_w + 1)
            after_crop = []
            contains = []

            tmp = mask[top: top + new_h, left:left + new_w]
            contains.append(np.unique(tmp))
            after_crop.append(tmp)

            all_obj = list(np.sort(contains[0]))

            if all_obj[-1] == 0:
                continue

            # remove background
            if all_obj[0] == 0:
                all_obj = all_obj[1:]
            # remove small obj
            new_all_obj = []
            for obj_id in all_obj:
                after_crop_pixels = torch.sum(after_crop[0] == obj_id)
                if after_crop_pixels > self.min_obj_pixel_num:
                    new_all_obj.append(obj_id)

            if len(new_all_obj) == 0:
                is_contain_obj = False
            else:
                is_contain_obj = True

            all_obj = [0] + new_all_obj

        img = img[:, top: top + new_h, left:left + new_w]
        mask = mask[top: top + new_h, left:left + new_w]
        instance = instance[top: top + new_h, left:left + new_w]

        img = F.interpolate(
            img.unsqueeze(0),
            size=(h, w),
            mode='nearest').squeeze()
        mask = F.interpolate(
            mask.unsqueeze(0).unsqueeze(0),
            size=(h, w),
            mode='nearest').squeeze()
        instance = F.interpolate(
            instance.unsqueeze(0).unsqueeze(0),
            size=(h, w),
            mode='nearest').squeeze()

        return (phrase, img, mask)

class CopyPaste(object):
    def __init__(self, p=0.3):
        super().__init__()
        self.p = p

        self.cfg = InstaBoostConfig(action_candidate=('normal', 'horizontal', 'skip'),
                                    action_prob=(0.5, 0.5, 0),
                                    color_prob=0.,
                                    heatmap_flag=True)

    def __call__(self, input):
        phrase, img, mask = input

        if 'left' in phrase or 'right' in phrase:
            return (phrase, img, mask)

        if torch.rand(1) < self.p:
            mask = mask.int().numpy().astype(np.uint8)
            img = img[0]

            mask, img = get_new_data(mask, img, self.cfg, background=None)

            width = img.shape[1]
            height = img.shape[0]

            mask, labels = get_coco_masks(mask, height, width)

            img = [img[None,:,:,:]]
            img = np.concatenate(img, axis=0)

            mask = torch.from_numpy(mask).float()

        return (phrase, img, mask)

class RandomHorizontalFlip(object):

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, input):
        phrase, img, mask = input

        if torch.rand(1) < self.p:
            flip_flag = True
        else:
            flip_flag = False

        if flip_flag:
            img = tF.hflip(img)
            mask = tF.hflip(mask)

        if flip_flag:
            ori = phrase
            if 'left' in phrase:
                ori = ori.replace('left', 'right')
            elif 'right' in phrase:
                ori = ori.replace('right', 'left')

            phrase = ori

        return (phrase, img, mask)
