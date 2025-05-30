# Copyright (c) OpenMMLab. All rights reserved.
import warnings
from collections import OrderedDict
from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp
from mmcv.cnn import build_norm_layer
from mmcv.cnn.bricks.transformer import FFN, build_dropout
from mmengine.logging import MMLogger
from mmengine.model import BaseModule, ModuleList
from mmengine.model.weight_init import (constant_init, trunc_normal_,
                                        trunc_normal_init)
from typing import Optional, Sequence, Tuple, Union
from mmdet.registry import MODELS
from mmdet.utils import OptConfigType, OptMultiConfig
from torch import Tensor, nn
from specdetr.specdetrutils import PatchEmbed, PatchMerging,AdaptivePadding
from mmengine.model import xavier_init


def expand_tensor_along_second_dim(x, num):
    assert x.size(1)<=num
    # 计算需要重复的次数
    repeat_times = num // x.size(1)
    # 使用 repeat 函数对 x 张量进行复制
    x = x.repeat(1, repeat_times, 1, 1)
    # 如果 num 不是 x.size(1) 的整数倍，则进行切片操作
    if num % x.size(1) != 0:
        x = torch.cat([x, x[:, :num % x.size(1)]], dim=1)
    return x

def extract_tensor_along_second_dim(x, m):
    # 计算等间隔的索引
    idx = torch.linspace(0, x.size(1) - 1, m).long().to(x.device)
    # 使用 index_select 函数在第二个维度上进行抽取
    x = torch.index_select(x, 1, idx)

    return x


class No_backbone_ST(nn.Module):
    def __init__(self,
                 in_channels=3,
                 embed_dims=96,
                 strides=(1, 2, 2, 4),
                 patch_size=(1, 2, 2, 4),
                 patch_norm=True,
                 act_cfg=dict(type='GELU'),
                 norm_cfg=dict(type='LN'),
                 pretrained=None,
                 num_levels =2,
                 token_masking=False,
                 init_cfg=None):
        assert not (init_cfg and pretrained), \
            'init_cfg and pretrained cannot be specified at the same time'
        if isinstance(pretrained, str):
            warnings.warn('DeprecationWarning: pretrained is deprecated, '
                          'please use "init_cfg" instead')
            self.init_cfg = dict(type='Pretrained', checkpoint=pretrained)
        elif pretrained is None:
            self.init_cfg = init_cfg
        else:
            raise TypeError('pretrained must be a str or None')

        #super(No_backbone_ST, self).__init__(init_cfg=init_cfg)
        super().__init__()
        assert strides[0] == patch_size[0], 'Use non-overlapping patch embed.'
        self.embed_dims =embed_dims
        self.in_channels = in_channels
        self.token_masking = token_masking
        ######THIS GAVE THE Parameter indices which did not receive grad for rank 0: 2 3 4 5 8 9 10 11 PROBLEM!!!
        """ self.patch_embed = PatchEmbed( 
            in_channels=in_channels,
            embed_dims=embed_dims,
            conv_type='Conv2d',
            kernel_size=patch_size[0],
            stride=strides[0],
            norm_cfg=norm_cfg if patch_norm else None,
            init_cfg=None) """
        self.num_levels = num_levels
        self.conv = nn.Conv2d(in_channels, embed_dims, kernel_size=1)
        """ self.mlp = nn.Sequential(
            nn.Linear(in_channels, embed_dims),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(embed_dims, embed_dims),
            nn.LeakyReLU(negative_slope=0.2)
        ) """
        if norm_cfg is not None:
            self.norm = build_norm_layer(norm_cfg, embed_dims)[1]
        if token_masking:
            self.mask_token = nn.Parameter(torch.randn(1, self.embed_dims))

            # self.norm = build_norm_layer(norm_cfg, 128*128)[1]

    def init_weights(self):
        # Conv layer
        xavier_init(self.conv, distribution='uniform', bias=0.)


        # Norm layer
        if hasattr(self, 'norm'):
            if isinstance(self.norm, nn.LayerNorm):
                nn.init.constant_(self.norm.bias, 0)
                nn.init.constant_(self.norm.weight, 1.0)

        # Optional mask token
        if self.token_masking:
            nn.init.normal_(self.mask_token, std=0.02)

    def train(self, mode=True):
        """Convert the model into training mode while keep layers freezed."""
        super(No_backbone_ST, self).train(mode)

    def forward(self, x):

        # x, hw_shape = self.patch_embed(x)
        # outs = []
        # out = x.view(-1, *hw_shape, self.embed_dims).permute(0, 3, 1, 2).contiguous()

        if self.in_channels < x.size(1):
            x = extract_tensor_along_second_dim(x, self.in_channels)
        outs = []
        # out = expand_tensor_along_second_dim(x, self.embed_dims)
        out = self.conv(x)
        out = self.norm(out.flatten(2).transpose(1, 2))

        ##IMPLEMENT TOKEN MASKING? Use flag in config?
        if self.token_masking:
            B, N, D = out.shape
            mask_ratio = 0.1  # for example, mask 40% of tokens
            num_mask = int(mask_ratio * N)  # number of tokens to mask

            # Generate random mask indices per sample
            mask_indices = torch.rand(B, N, device=out.device).argsort(dim=1)[:, :num_mask]  # [B, num_mask]

            # Create full token mask (0 = keep, 1 = mask)
            mask = torch.zeros(B, N, dtype=torch.bool, device=out.device)
            mask.scatter_(1, mask_indices, 1)  # mark masked positions as 1

            # Expand mask to match embedding shape
            mask = mask.unsqueeze(-1).expand(-1, -1, D)

            # Replace masked tokens with mask_token
            out = torch.where(mask, self.mask_token.expand(B, N, D), out)  # masked output


        # BN
        # out = self.norm(out.flatten(2)).transpose(1, 2)
        # y = x.reshape(x.size(0),x.size(1),-1).permute(0, 2, 1)
        # out = self.mlp(y)
        out = out.permute(0, 2, 1).reshape(x.size(0), self.embed_dims,x.size(2),x.size(3)).contiguous()
        outs.append(out)
        if self.num_levels > 1:
            mean = outs[0].mean(dim=(2, 3), keepdim=True).detach()
            outs.append(mean)
        return outs