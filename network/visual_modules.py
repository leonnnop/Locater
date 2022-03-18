import math
import logging
from functools import partial
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

import timm

from network.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from network.layers import DropPath, to_2tuple, trunc_normal_

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x

class ProjectDotDecode(nn.Module):
    def __init__(self, dim, num_heads=8, bias=True, scale=None, cat_enc_mode=False, norm=False):
        super().__init__()
        # print(num_heads)
        self.num_heads = num_heads
        head_dim = dim // num_heads

        self.scale = scale or head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=bias)
        self.k = nn.Linear(dim, dim, bias=bias)

        dec_dim = num_heads*num_heads
        if cat_enc_mode == 'concat':
            dec_dim += dim
        elif cat_enc_mode == 'residule':
            self.res_proj = nn.Linear(dim, dec_dim, bias=bias)

        self.mask_head = self._pup_predict_block(in_c=dec_dim)
        self.cat_enc_mode = cat_enc_mode

        self.norm = norm


    def _pup_predict_block(self, in_c):
        extra_in_channels = int(in_c/4)
        in_channels = [
                in_c,
                extra_in_channels,
                extra_in_channels,
                extra_in_channels,
                extra_in_channels,
                ]
        out_channels = [
                extra_in_channels,
                extra_in_channels,
                extra_in_channels,
                extra_in_channels,
                1,
                ]

        modules = []
        for i, (in_channel, out_channel) in enumerate(zip(in_channels, out_channels)):
            modules.append(
                    nn.Conv2d(
                        in_channels=in_channel,
                        out_channels=out_channel,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                        )
                    )
            if i != 4:
                modules.append(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False))

        return nn.Sequential(*modules)

    def forward(self, params):
        x, y = params
        B, N, C = x.shape
        _, M, _ = y.shape

        q = self.q(x).reshape(B, N, 1, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k = self.k(y).reshape(B, M, 1, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k = q[0], k[0] # * B,H,N,D

        if self.norm: # * in cosine distance
            q = torch.nn.functional.normalize(q, p=2, dim=-1)
            k = torch.nn.functional.normalize(k, p=2, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.scale # * B,H,N,M
        
        if self.num_heads != 1:
            P = int(math.sqrt(N))
            # print(self.num_heads)
            attn = attn.transpose(-1,-2).reshape(B,-1,N) # * B,H*M,N
            if self.cat_enc_mode == 'concat':
                # * B,N,D -> B,D,N -> B,D+H*M,N
                _x = x.transpose(-1,-2)
                attn = torch.cat([_x, attn], dim=1)
            elif self.cat_enc_mode == 'residule':
                _x = self.res_proj(x).transpose(-1,-2)
                attn += _x

            attn = attn.view(B,-1,P,P)
            attn = self.mask_head(attn)
        else:
            attn = attn.squeeze(1)

        return attn 

class ProjectDotDistance(nn.Module):
    def __init__(self, dim, num_heads=8, bias=True, scale=None):
        super().__init__()
        # print(num_heads)
        self.num_heads = num_heads
        head_dim = dim // num_heads

        self.scale = scale or head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=bias)
        self.k = nn.Linear(dim, dim, bias=bias)

    def forward(self, params):
        x, y = params
        B, N, C = x.shape
        _, M, _ = y.shape

        q = self.q(x).reshape(B, N, 1, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k = self.k(y).reshape(B, M, 1, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k = q[0], k[0] 

        attn = (q @ k.transpose(-2, -1)) * self.scale
        
        if self.num_heads == 1:
            attn = attn.squeeze(1)

        return attn 

class CrossAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.k = nn.Linear(dim, dim, bias=qkv_bias)
        self.v = nn.Linear(dim, dim, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, params):
        query, key, value = params
        B, N_q, C = query.shape
        N_k = key.shape[1]
        
        q = self.q(query).reshape(B, N_q, 1, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k = self.k(key).reshape(B, N_k, 1, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        v = self.v(value).reshape(B, N_k, 1, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = q[0], k[0], v[0]   # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N_q, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

class PatchEmbed(nn.Module):
    """ 2D Image to Patch Embedding
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, norm_layer=None, flatten=True):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.flatten = flatten

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC
        x = self.norm(x)
        return x

