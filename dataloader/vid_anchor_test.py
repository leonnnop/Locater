# -*- coding: utf-8 -*-

import os
import sys
sys.path.append('.')
import cv2
import json
import uuid
import tqdm
import torch
import numpy as np
import os.path as osp
import scipy.io as sio

import torch.utils.data as data
import h5py

import random
from transformers import AutoTokenizer, AutoModel
from torch.utils.data.dataloader import default_collate

from general_util import save_rgb_img, touch_dir

class Seq_Test(data.Dataset):
    def __init__(self, image_root, label_root, seq_name, images, labels, obj_n, transform=None):
        self.image_root = image_root
        self.label_root = label_root
        self.seq_name = seq_name
        self.images = images
        self.labels = labels
        self.obj_n = obj_n
        self.num_frame = len(self.images)

        self.transform = transform

    def __len__(self):
        return len(self.images)

    def read_image(self, idx):
        img_name = self.images[idx]
        img_path = os.path.join(self.image_root, self.seq_name, img_name)

        img = cv2.imread(img_path)
        if img.shape[-1] > 1:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        else:
            img = np.stack([img] * 3)

        return img

    def read_label(self, label_name):
        label_path = os.path.join(self.label_root, (self.seq_name.replace('/','_')+'_'+str(self.obj_n)+'_'+label_name))
        label = torch.load(label_path).float().unsqueeze(0)
        return label

    def __getitem__(self, idx):
        img_name = self.images[idx]
        img = self.read_image(idx)
        img = img[None,:,:,:] # * 1,H,W,3

        current_label_name = img_name.split('.')[0] + '.pth'
        if current_label_name in self.labels:
            current_label = self.read_label(current_label_name)
        else:
            # * prepare a small mask for testing
            current_label = torch.from_numpy(np.zeros_like(img)[:,:,:,0]) - 1.

        if self.transform is not None:
            _, img, mask  = self.transform(('', img, current_label))
            
        return img, mask



class ReferDataset_VID(object):

    def __init__(self, dataset_root, dataset='A2D',
                 transform=None, max_query_len=20, 
                 N1=3, testing_type=None):

        self.images = []
        self.dataset_root = dataset_root
        self.dataset = dataset
        self.query_len = max_query_len
        self.transform = transform

        self.testing_type = testing_type

        self.N1 = N1 # * global

        if self.dataset == 'JHMDB':
            _prefix_dataset = self.dataset
            self.mask_dir = osp.join(self.dataset_root, self.dataset, 'mask')
            self.all_im_dir = osp.join(
                self.dataset_root, self.dataset, 'images')
            self.vid_frames_dict = torch.load(osp.join(
                self.dataset_root, self.dataset, 'vid_frames_dict.pth'))
            json_path = osp.join(self.dataset_root, dataset, _prefix_dataset+'_test_expressions.json')
        elif self.dataset == 'A2D':
            _prefix_dataset = 'A2D_'+testing_type
            self.all_im_dir = osp.join(
                self.dataset_root, dataset+'_SUBSET', testing_type, 'allPngs320H')
            self.mask_dir = osp.join(self.dataset_root, dataset+'_SUBSET', testing_type, 'mask')
            self.vid_frames_dict = torch.load(osp.join(
                self.dataset_root, dataset+'_SUBSET', testing_type, 'vid_frame_dict.pth'))
            json_path = osp.join(self.dataset_root, dataset+'_SUBSET', testing_type, _prefix_dataset+'_test_expressions.json')
        
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

        self.ann_f = json.load(open(json_path, 'r'))
        self.seqs = self.ann_f.keys()

        self.samples = []
        self._construct_sample()

    def _construct_sample(self):
        for seq in self.seqs:
            data = self.ann_f[seq]
            images = list(map(lambda x: x + '.png', list(data["frames"])))
            images = np.sort(np.unique(images))

            for obj in data['expressions']:
                labels = list(map(lambda x: x + '.pth', list(data["expressions"][obj]['labels'])))
                labels = np.sort(np.unique(labels))
                exp = data["expressions"][obj]['exp']
                
                self.samples.append((seq, images, labels, obj, exp))

    def __len__(self):
        return len(self.samples)

    def tokenize_phrase(self, phrase):
        phrase = self.tokenizer(phrase.lower(), padding='max_length', max_length=self.query_len+2, truncation=True, return_tensors="pt")
        for key in phrase:
            phrase[key] = phrase[key].squeeze()
        return phrase

    def check_query(self, query):
        wr_q = {
            'the peson in red is standing': 'the person in red is standing',
            'rd black formula car riding on the right side of the track': 'red black formula car riding on the right side of the track',
            'white standing near the building': 'person standing near the building',
            'red black is riding on the left': 'red black car is riding on the left',
            'solder in camouflage uniforms crawling ': 'soldier in camouflage uniforms crawling',
            'white in shirt is walking in the airport terminal': 'person in shirt is walking in the airport terminal',
            'the left with yellow t shirt on the left running': 'the left man with yellow t shirt on the left running',
            'black is climbing the stairs': 'man is climbing the stairs',
            'white black is walking on the beach': 'white black dog is walking on the beach',
            'on in blue running on the left': 'man in blue running on the left',
            'a blue bar is sitting on the floor in the middle': 'a blue ball is sitting on the floor in the middle',
            'solder crawling on the ground': 'soldier crawling on the ground',
            'white brown is eating  the corn': 'white brown dog is eating the corn',
            'white in white tracksuit is running during competition': 'man in white tracksuit is running during competition',
            'white licking her claws on the left': 'white cat licking her claws on the left'
        }

        if query in wr_q:
            return wr_q[query]
        else:
            return query

    def _extract_one_frame_with_n(self, vid_name, frame_n):
        img_name = '{:0>5d}.png'.format(frame_n)
        img_path = osp.join(self.all_im_dir, vid_name, img_name)

        img = cv2.imread(img_path)
        if img is None:
            print(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        return img

    def _extract_global_imgs(self, seq):

        if self.dataset == 'JHMDB':
            category, vid_name = seq.split('/')[0], seq.split('/')[1]
            frame_num = self.vid_frames_dict[category][vid_name]

        elif self.dataset == 'A2D':
            frame_num = self.vid_frames_dict[seq]

        n_step = self.N1

        all_imgs = []
        if n_step != 0: # vid
            step = int(frame_num / n_step)

            for idx in range(n_step):
                frame_n = (1+step)//2 + idx*step

                img = self._extract_one_frame_with_n(seq, frame_n)
                all_imgs.append(img[None,:,:,:])

            imgs = np.concatenate(all_imgs, axis=0)
        else:
            imgs = -1.

        return imgs

    def __getitem__(self, idx):
        seq, images, labels, obj_n, phrase = self.samples[idx]
        global_images = self._extract_global_imgs(seq)
        _tmp_m = torch.from_numpy(np.ones_like(global_images)[:,:,:,0])

        if self.transform is not None:
            phrase, global_images, _  = self.transform((phrase, global_images, _tmp_m))
        phrase = self.check_query(phrase.lower())
        phrase = self.tokenize_phrase(phrase)

        seq_dataset = Seq_Test(self.all_im_dir, self.mask_dir, seq, images, labels, obj_n, transform=self.transform)
        return seq_dataset, global_images, phrase


def test_collate_fn(batch):
    assert len(batch) == 1

    seq_dataset = batch[0][0]
    global_images = default_collate([e[1] for e in batch])
    phrase = default_collate([e[2] for e in batch])

    return seq_dataset, global_images, phrase


if __name__ == "__main__":
    pass