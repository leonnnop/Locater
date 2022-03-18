import sys
sys.path.append('.')

from collections import OrderedDict

import torch
import torch.nn as nn
from network.vision_transformer import vision_transformer
from network.lanEncode import EncodeNet, LanEncoder_LSTM
from network.MemEncode import GlobalMemoryNet, LocalMemoryNet, QueryNet
from network.visual_modules import CrossAttention, ProjectDotDecode

from network.helpers import IntermediateSequential

class VLFTrans(nn.Module):
    def __init__(
        self,
        img_dim=320,
        patch_dim=16.,
        in_chans=3,
        embedding_dim=768,
        K=3,
        local_mem_size=1.5,
        global_mem_size=2.,
        fuse_depth=2,
        post_depth=2,
        vision_depth=6,
        cat_enc_mode='concat'
    ):
        super(VLFTrans, self).__init__()

        self.img_dim = img_dim
        self.embedding_dim = embedding_dim
        self.patch_dim = patch_dim
        self.patch_num = int(img_dim / self.patch_dim) ** 2

        self.in_chans = in_chans
        self.K = K
        
        self.local_mem_size = local_mem_size * self.patch_num
        self.global_mem_size = global_mem_size * self.patch_num

        self._local_mem = None
        self._global_mem = None

        _linguistic_enc_func = LanEncoder_LSTM
        _vision_enc_func = vision_transformer

        self.vision_encoder = _vision_enc_func(
                                    img_size=img_dim,
                                    in_chans=in_chans,
                                    depth=vision_depth
                                    )
                                    
        self.lang_encoder = _linguistic_enc_func()
        self.trans_fusion = IntermediateSequential(OrderedDict(
                                [(str(i), EncodeNet(depth=fuse_depth,post_depth=post_depth, cascade_lang_fuse=False))
                                    for i in range(K)]), return_intermediate=True)
        
        _loc_mem_func = LocalMemoryNet
        _glo_mem_func = GlobalMemoryNet

        if self.local_mem_size > 0:
            self.localmem_writer = _loc_mem_func(mem_size=self.local_mem_size, embed_dim=self.embedding_dim)
        if self.global_mem_size > 0:
            self.globalmem_writer = _glo_mem_func(mem_size=self.global_mem_size, embed_dim=self.embedding_dim)
        self.mem_reader = nn.Sequential(*[
                                    CrossAttention(dim=self.embedding_dim, qkv_bias=True),
                                    nn.LayerNorm(self.embedding_dim)
                                    ])

        self.aux_decoders = self._decoder_gen()
        self.query_generator = QueryNet(embed_dim=self.embedding_dim, num_heads=8)

        self.main_dec_net = ProjectDotDecode(dim=self.embedding_dim, num_heads=8, cat_enc_mode=cat_enc_mode, norm=False)


    def _decoder_gen(self):
        predict = []
        for idx in range(self.K):
            # predict.append(self._predict_block(self.embedding_dim))
            predict.append(self._predict_block(self.embedding_dim))

        predicts = nn.ModuleList(predict)

        return predicts

    def _predict_block(self, in_c):
        return nn.Sequential(
                    nn.Conv2d(
                        in_channels=in_c,
                        out_channels=self.embedding_dim,
                        kernel_size=1,
                        stride=1,
                        padding=self._get_padding('VALID', (1, 1),),
                    ),
                    nn.BatchNorm2d(self.embedding_dim),
                    nn.ReLU(),
                    nn.Conv2d(
                        in_channels=self.embedding_dim,
                        out_channels=1,
                        kernel_size=1,
                        stride=1,
                        padding=self._get_padding('VALID', (1, 1),),
                    ),
                    nn.Upsample(scale_factor=self.patch_dim, mode='bilinear', align_corners=False)
                )

    def encode(self, vis, lang):
        B,T,C,H,W = vis.shape
        vis = vis.view(-1,C,H,W)

        lang_f, hidden_states  = self.lang_encoder(lang)

        vis_f = self.vision_encoder.forward_features(vis)

        return vis_f, lang_f

    def decode(self, x, q, x_int=None):
        # B,T*(1+N_v)+N_l,C
        all_decodes = []

        if x_int is not None:
            for idx, _k in enumerate(x_int.keys()):
                fused_f = x_int[_k]
                fused_f = fused_f[:,1:self.patch_num+1]
                fused_f = self._reshape_output(fused_f, fused_f.shape[2])
                # B,C,H,W
                fused_f = self.aux_decoders[idx](fused_f)
                all_decodes.append(fused_f)

        all_decodes.append(self._main_decode(x, q))

        return all_decodes

    def _main_decode(self, c_v, q):
        return self.main_dec_net((c_v, q))

    def _prep_global_mem(self, vis, lang):
        B,T,C,H,W = vis.shape
        vis_f, lang_f = self.encode(vis, lang)
        (fused_v, _), _ = self.trans_fusion(vis_f, lang_f)
        fused_v = fused_v.view(B,T,*fused_v.shape[1:])
        global_v = fused_v[:,:,1:]

        self._global_mem = self.globalmem_writer(cur_fea=global_v)

        return True

    def _reset_memory(self):
        self._local_mem = None
        self._global_mem = None

        return True

    def _forward_testing(self, vis, lang):
        B,T,C,H,W = vis.shape

        # B,T,C,H,W
        vis_f, lang_f = self.encode(vis, lang)
        # fusion
        (fused_v, _post_lang), _interm_fused_v = self.trans_fusion(vis_f, lang_f)
        fused_v = fused_v.view(B,T,*fused_v.shape[1:])
        _post_lang = _post_lang.view(B,T,*_post_lang.shape[1:])[:,-1]

        # * exclude cls token
        fused_v = fused_v[:,:,1:]

        # * seperate vis fea
        cur_v = fused_v[:,-1]

        if self._local_mem is None: _loc_mem = cur_v
        else: _loc_mem = self._local_mem

        # gerneate query
        query, _query_attn = self.query_generator(lang_f, cur_v, _loc_mem, self._global_mem) # * B,C
        
        _cat_mem = torch.cat([_loc_mem, self._global_mem], dim=1)
        context_v = self.mem_reader((cur_v, _cat_mem, _cat_mem))

        all_deocodes = self.decode(context_v, query)

        # memory update
        _pred_mask = (torch.sigmoid(all_deocodes[-1]) > 0.5).float()
        if self.local_mem_size > 0:
            self._local_mem = self.localmem_writer(prev_m=self._local_mem, cur_fea=cur_v.unsqueeze(1), cur_mask=_pred_mask)
        
        return all_deocodes, (_query_attn)

    def forward(self, **kwargs):
        return self._forward_testing(**kwargs)

    def _get_padding(self, padding_type, kernel_size):
        assert padding_type in ['SAME', 'VALID']
        if padding_type == 'SAME':
            _list = [(k - 1) // 2 for k in kernel_size]
            return tuple(_list)
        return tuple(0 for _ in kernel_size)

    def _reshape_output(self, x, embedding_dim):
        x = x.view(
            x.size(0),
            int(self.img_dim / self.patch_dim),
            int(self.img_dim / self.patch_dim),
            embedding_dim,
        )
        x = x.permute(0, 3, 1, 2).contiguous()
        return x

    def load_state_dict(self, new_state):
        state = self.state_dict()
        for layer in state:
            if layer in new_state:
                if state[layer].size() == new_state[layer].size():
                    state[layer] = new_state[layer]
            elif 'module.'+layer in new_state:
                new_layer = 'module.'+layer
                if state[layer].size() == new_state[new_layer].size():
                    state[layer] = new_state[new_layer]
            else: print(layer)

        super().load_state_dict(state)

