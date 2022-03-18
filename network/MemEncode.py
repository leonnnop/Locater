import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np
from transformers import AutoTokenizer, AutoModel
# import math
import sys
sys.path.append('.')
from network.layers import DropPath, to_2tuple, trunc_normal_
from network.visual_modules import Mlp, Attention, Block, ProjectDotDistance

def update_function(c_func, o_func, p_func, m_func, _fea, _mem, num_heads=8):
    B = _fea.shape[0]
    M = _mem.shape[1]
    N = _fea.shape[1]
    H = num_heads

    _ap_mem = torch.mean(_mem, dim=1, keepdim=True) # * B,1,C
    c_global = c_func(torch.cat([_fea, _ap_mem.expand_as(_fea)], dim=2)) # * B,N,C

    _cur_mem = m_func(_mem).reshape(B,M,H,-1).transpose(1,2) # * B,M,H,D -> B,H,M,D
    _cur_mem = _cur_mem.unsqueeze(2) # * B,H,1,M,D
    _cur_mem = _cur_mem.expand(-1,-1,N,-1,-1) # * B,H,N,M,D

    o_global = o_func((c_global, _mem)) # * B,H,N,M
    _c_global = c_global.reshape(B,N,H,-1).transpose(1,2)  # * B,H,N,D
    # print
    _c = torch.matmul(o_global.transpose(-1,-2), _c_global).transpose(1,2).contiguous().view(B,M,-1) # * B,H,M,D -> B,M,H,D -> B,M,C
    _c = p_func(_c)

    _o = torch.mean((1-o_global).unsqueeze(-1)*_cur_mem, dim=2).transpose(1,2).reshape(B,M,-1)

    _mem = _c + _o

    return _mem


@torch.no_grad()
def _check_memory(mem_size, mem_state):
    if mem_state.shape[1] < mem_size:
        _r_size = mem_size - mem_state.shape[1]
        # print('in check memory and pad [{}]'.format(_r_size))
        _mem = torch.mean(mem_state, dim=1, keepdim=True).repeat(1,_r_size,1) # * B,M,C -> B,1,C -> B,R,C
        mem_state = torch.cat([mem_state, _mem], dim=1)

    return mem_state


class GlobalMemoryNet(nn.Module):

    def __init__(self, mem_size, embed_dim=768, mlp_ratio=4, num_heads=8):
        super(GlobalMemoryNet, self).__init__()

        self.mem_size = int(mem_size)
        self.c_func = nn.Sequential(*[
                                    Mlp(in_features=2*embed_dim, 
                                        hidden_features=embed_dim*mlp_ratio, 
                                        out_features=embed_dim),
                                    ])
        self.o_func = nn.Sequential(*[
                                    ProjectDotDistance(dim=embed_dim, num_heads=num_heads),
                                    nn.Sigmoid()
                                    ])
        self.m_func = Mlp(in_features=embed_dim, 
                          hidden_features=embed_dim*mlp_ratio, 
                          out_features=embed_dim)
                                    
        self.post_func = Mlp(in_features=embed_dim, 
                             hidden_features=embed_dim*mlp_ratio, 
                             out_features=embed_dim)

        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, cur_fea):
        B,T,N,C = cur_fea.shape
        cur_fea = self.norm1(cur_fea)

        # * init mem
        if cur_fea.shape[1]*cur_fea.shape[2] <= self.mem_size:
            return cur_fea.contiguous().view(B,-1,C)
        else:
            with torch.no_grad():
                _init_t = int(self.mem_size // float(N))
                if _init_t > 0:
                    init_state = cur_fea[:,:_init_t].contiguous().view(B,-1,C)
                else:
                    init_state = cur_fea[:,0].mean(1).view(B,1,C)

        init_state = _check_memory(self.mem_size, init_state)
        cur_mem = init_state
        for _t in range(_init_t, T):
            _cur_v = cur_fea[:,_t]
            cur_mem = update_function(self.c_func, self.o_func, self.post_func, self.m_func, _cur_v, cur_mem)
            cur_mem = self.norm2(cur_mem)

        return cur_mem

class LocalMemoryNet(nn.Module):

    def __init__(self, mem_size, embed_dim=768, mlp_ratio=4, num_heads=8):
        super(LocalMemoryNet, self).__init__()

        self.mem_size = int(mem_size)
        self.c_func = nn.Sequential(*[
                                    Mlp(in_features=2*embed_dim, 
                                        hidden_features=embed_dim*mlp_ratio, 
                                        out_features=embed_dim),
                                    nn.LayerNorm(embed_dim)
                                    ])
        self.o_func = nn.Sequential(*[
                                    ProjectDotDistance(dim=embed_dim, num_heads=num_heads),
                                    nn.Sigmoid()
                                    ])
        self.m_func = Mlp(in_features=embed_dim, 
                          hidden_features=embed_dim*mlp_ratio, 
                          out_features=embed_dim)

        self.post_func = Mlp(in_features=embed_dim, 
                             hidden_features=embed_dim*mlp_ratio, 
                             out_features=embed_dim)

        self.mask_enc = nn.Conv2d(in_channels=1, 
                                  out_channels=embed_dim, 
                                  kernel_size=3,
                                  padding=1
                                  )

        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, cur_fea, prev_m=None, cur_mask=None):
        # * cur_fea B,T,N,C
        B,T,N,C = cur_fea.shape
        
        cur_mask = cur_mask.contiguous().view(-1,1,*cur_mask.shape[-2:])
        _mask_enc = self.mask_enc(cur_mask) # * B*T,C,H,W
        _mask_enc = torch.nn.functional.interpolate(_mask_enc, int(np.sqrt(N)), mode='bilinear', align_corners=False)
        _mask_enc = _mask_enc.flatten(2).transpose(1,2) # * B*T,N,C
        _mask_enc = _mask_enc.view(B,T,N,C)

        cur_fea = cur_fea + _mask_enc

        cur_fea = self.norm1(cur_fea)

        # * init mem
        _n_newfeat = cur_fea.shape[1]*cur_fea.shape[2]
        if prev_m is None:
            if _n_newfeat <= self.mem_size:
                return cur_fea.contiguous().view(B,-1,C)
            else:
                with torch.no_grad():
                    _init_t = int(self.mem_size // float(N))
                    if _init_t > 0:
                        prev_m = cur_fea[:,:_init_t].contiguous().view(B,-1,C)
                    else:
                        prev_m = cur_fea[:,0].mean(1).view(B,1,C)
        else:
            if prev_m.shape[1] == self.mem_size:
                _init_t = 0
            elif (_n_newfeat + prev_m.shape[1]) <= self.mem_size:
                return torch.cat([prev_m, cur_fea.contiguous().view(B,-1,C)], dim=1)
            else:
                with torch.no_grad():
                    _init_t = int((self.mem_size - prev_m.shape[1]) // float(N))
                    if _init_t > 0:
                        prev_m = torch.cat([prev_m, cur_fea[:,:_init_t].contiguous().view(B,-1,C)], dim=1)
                    # * else: cat mem size < 1 (typically 0.5), just use average pooled tensor to pad remaining state -> in _check_memory(func)

        prev_m = _check_memory(self.mem_size, prev_m)

        cur_mem = prev_m
        for _t in range(_init_t, T):
            _cur_v = cur_fea[:,_t]
            cur_mem = update_function(self.c_func, self.o_func, self.post_func, self.m_func, _cur_v, cur_mem)
            cur_mem = self.norm2(cur_mem)

        return cur_mem

class QueryNet(nn.Module):

    def __init__(self, embed_dim=768, mlp_ratio=4, num_heads=8):
        super(QueryNet, self).__init__()

        self.vis_proj = Mlp(in_features=3*embed_dim, 
                          hidden_features=embed_dim*mlp_ratio, 
                          out_features=embed_dim)
        self.lin_proj = Mlp(in_features=embed_dim, 
                          hidden_features=embed_dim*mlp_ratio, 
                          out_features=embed_dim)
        self.value_proj = Mlp(in_features=embed_dim, 
                          hidden_features=embed_dim*mlp_ratio, 
                          out_features=embed_dim)

        self.norm = nn.LayerNorm(embed_dim)
        self.num_heads = num_heads

        head_dim = embed_dim // num_heads
        self.scale = head_dim ** -0.5
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, lang_f, cur_v, _loc_mem, _glo_mem):
        _v = torch.mean(cur_v, dim=1)
        _m_l = torch.mean(_loc_mem, dim=1)
        _m_g = torch.mean(_glo_mem, dim=1)
        
        B,L,C = lang_f.shape
        _v_p = self.vis_proj(torch.cat([_v, _m_l, _m_g], dim=1)).view(B,self.num_heads,C // self.num_heads,1)# * B,H,D,1
        _l_p = self.lin_proj(lang_f) # * B,L,C
        _l_v = self.value_proj(lang_f) # * B,L,C

        k = _l_p.reshape(B, L, self.num_heads, C // self.num_heads).transpose(1,2) # * B,H,L,D

        _attn = torch.matmul(k, _v_p).squeeze(-1) * self.scale # * B,H,L
        _attn = torch.softmax(_attn, dim=-1) # * B,H,L

        query = torch.bmm(_attn, _l_v)

        query = self.norm(query) # * B,H,C

        return query, _attn


