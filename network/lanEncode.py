import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np
from transformers import AutoTokenizer, AutoModel
import os
import sys
sys.path.append('.')
from network.layers import DropPath, to_2tuple, trunc_normal_
from network.visual_modules import Mlp, Attention, Block
from network.helpers import _load_block_weights

class LanEncoder_LSTM(nn.Module):
    def __init__(self, dict_size=30000, emb_size=300, hid_size=384, lang_layers=3):
        super(LanEncoder_LSTM, self).__init__()
        self.emb = nn.Embedding(dict_size, emb_size)
        self.lang_model = nn.LSTM(emb_size, hid_size, num_layers=lang_layers, bidirectional=True)
        self.norm = nn.LayerNorm(2*hid_size)

    def forward(self, lang):
        lang = lang['input_ids']

        lang = self.emb(lang)
        lang = torch.transpose(lang,0,1)
        lang, _ = self.lang_model(lang)
        lang = torch.transpose(lang,0,1)

        hidden = None

        lang = self.norm(lang)

        return lang, hidden
        

class EncodeNet(nn.Module):
    def __init__(self, embed_dim=768, depth=4, num_heads=12, mlp_ratio=4.,
                 qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.,
                 norm_layer=nn.LayerNorm, init_depth=6,
                 cascade_lang_fuse=False, post_depth=1):
        super(EncodeNet, self).__init__()

        self.modal_embed = nn.Parameter(torch.zeros(1, 2, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)
        self.cascade_lang_fuse = cascade_lang_fuse

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(depth)])
        self.pos_blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(post_depth)])

        self.norm = norm_layer(embed_dim)

        trunc_normal_(self.modal_embed, std=.02)
        self.apply(self._init_weights)


    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore()
    def load_pretrained(self, checkpoint_path, prefix='', start_i=0):
        _load_block_weights(self, checkpoint_path, prefix, start_i)

    def forward(self, vis, lang):
        N_v = vis.shape[1]
        B,N_l,C = lang.shape
        # B,1+N_v,C
        vis_pos = vis + self.modal_embed[:,:1]
        lang_pos = lang + self.modal_embed[:,1:]

        if vis.shape[0]!=B:
            T =  int(vis.shape[0]/B)
            lang_pos = lang_pos.view(B,1,N_l,C).repeat(1,T,1,1).contiguous().view(-1,N_l,C)
        
        modal_f = torch.cat([vis_pos, lang_pos], dim=1) #* B,N_v+N_l,C

        for blk in self.blocks:
            modal_f = blk(modal_f)
        modal_f = self.norm(modal_f)
            
        modal_f = modal_f[:,:N_v]
        if self.cascade_lang_fuse:
            _lang = modal_f[:,N_v:]
        else:
            _lang = lang

        for pos_blk in self.pos_blocks:
            modal_f = pos_blk(modal_f)

        modal_f = self.norm(modal_f)

        return (modal_f, _lang)


