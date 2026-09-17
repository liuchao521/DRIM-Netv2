"""
DEIMv2: Real-Time Object Detection Meets DINOv3
Copyright (c) 2025 The DEIMv2 Authors. All Rights Reserved.
---------------------------------------------------------------------------------
Modified from DINOv3 (https://github.com/facebookresearch/dinov3)

Copyright (c) Meta Platforms, Inc. and affiliates.

This software may be used and distributed in accordance with
the terms of the DINOv3 License Agreement.
"""

import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp

from functools import partial
# from ..core import register
from network.DINOV3.vit_tiny import VisionTransformer
from network.DINOV3.dinov3 import DinoVisionTransformer
from network.DINOV3.mfcn import MFCN


class SpatialPriorModulev2(nn.Module):
    def __init__(self, inplanes=16):
        super().__init__()

        # 1/4
        self.stem = nn.Sequential(
            *[
                nn.Conv2d(3, inplanes, kernel_size=3, stride=2, padding=1, bias=False),
                nn.SyncBatchNorm(inplanes),
                nn.GELU(),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            ]
        )
        # 1/8
        self.conv2 = nn.Sequential(
            *[
                nn.Conv2d(inplanes, 2 * inplanes, kernel_size=3, stride=2, padding=1, bias=False),
                nn.SyncBatchNorm(2 * inplanes),
            ]
        )
        # 1/16
        self.conv3 = nn.Sequential(
            *[
                nn.GELU(),
                nn.Conv2d(2 * inplanes, 4 * inplanes, kernel_size=3, stride=2, padding=1, bias=False),
                nn.SyncBatchNorm(4 * inplanes),
            ]
        )
        # 1/32
        self.conv4 = nn.Sequential(
            *[
                nn.GELU(),
                nn.Conv2d(4 * inplanes, 4 * inplanes, kernel_size=3, stride=2, padding=1, bias=False),
                nn.SyncBatchNorm(4 * inplanes),
            ]
        )

    def forward(self, x):
        c1 = self.stem(x)
        c2 = self.conv2(c1)     # 1/8
        c3 = self.conv3(c2)     # 1/16
        c4 = self.conv4(c3)     # 1/32

        return c2, c3, c4


# @register()
class DINOv3STAs(nn.Module):
    def __init__(
        self,
        name="dinov3_vits16plus",
        weights_path=None,

        interaction_indexes=[5,8,11],
        finetune=True,
        embed_dim=192,
        num_heads=3,
        patch_size=16,
        use_sta=True,
        conv_inplane=32,
        hidden_dim=256,
    ):
        super(DINOv3STAs, self).__init__()
        if weights_path is None:
            weights_path = os.environ.get("DINOV3_WEIGHTS")
        # if 'dinov3' in name:
        #     self.dinov3 = DinoVisionTransformer(name=name)
        #     if weights_path is not None and os.path.exists(weights_path):
        #         print(f'Loading ckpt from {weights_path}...')
        #         self.dinov3.load_state_dict(torch.load(weights_path))
        #     else:
        #         print('Training DINOv3 from scratch...')
        if 'dinov3' in name:
            self.dinov3 = DinoVisionTransformer(name=name)
            if weights_path is not None and os.path.exists(weights_path):
                print(f'Loading ckpt from {weights_path}...')
                ckpt = torch.load(weights_path, map_location='cpu')  # 关键
                # 如果权重文件是 {"state_dict": ...} 这种格式，解包一下
                if isinstance(ckpt, dict) and 'state_dict' in ckpt:
                    ckpt = ckpt['state_dict']
                self.dinov3.load_state_dict(ckpt)
            else:
                raise FileNotFoundError(
                    "DINOv3 weights were not found. Set DINOV3_WEIGHTS to the "
                    "official dinov3_vits16plus checkpoint path."
                )
        else:
            self.dinov3 =  VisionTransformer(embed_dim=embed_dim, num_heads=num_heads, return_layers=interaction_indexes)
            if weights_path is not None and os.path.exists(weights_path):
                print(f'Loading ckpt from {weights_path}...')
                self.dinov3._model.load_state_dict(torch.load(weights_path))
            else:
                print('Training ViT-Tiny from scratch...')
        # Preserve the pretrained representation and adapt only normalization.
        for p in self.dinov3.parameters():
            p.requires_grad = False

        trainable_norm_params = 0
        for module_name, module in self.dinov3.named_modules():
            if isinstance(module, nn.LayerNorm) or "norm" in module_name.lower():
                for p in module.parameters(recurse=False):
                    p.requires_grad = True
                    trainable_norm_params += p.numel()

        print(f"DINOv3 LN-tuning enabled: {trainable_norm_params} trainable normalization parameters")


        embed_dim = self.dinov3.embed_dim
        self.interaction_indexes = interaction_indexes
        self.patch_size = patch_size

        if not finetune:
            self.dinov3.eval()
            self.dinov3.requires_grad_(False)
        # self.mfcn = MFCN(
        #     inplanes=[224, 224, 224],  # 每个输入层的通道数
        #     outplanes=[224 + 224 + 224],  # 拼接后的通道
        #     instrides=[8, 16, 32],  # 每层输入的下采样倍率
        #     outstrides=[8]           # 输出分辨率为 1/8
        #         )
        # init the feature pyramid
        self.use_sta = use_sta
        if use_sta:
            print(f"Using Lite Spatial Prior Module with inplanes={conv_inplane}")
            self.sta = SpatialPriorModulev2(inplanes=conv_inplane)
        else:
            conv_inplane = 0

        # linear projection
        hidden_dim = hidden_dim if hidden_dim is not None else embed_dim
        self.convs = nn.ModuleList([
            nn.Conv2d(embed_dim + conv_inplane*2, hidden_dim, kernel_size=1, stride=1, padding=0, bias=False),
            nn.Conv2d(embed_dim + conv_inplane*4, hidden_dim, kernel_size=1, stride=1, padding=0, bias=False),
            nn.Conv2d(embed_dim + conv_inplane*4, hidden_dim, kernel_size=1, stride=1, padding=0, bias=False)
        ])
        # norm
        self.norms = nn.ModuleList([
            nn.SyncBatchNorm(hidden_dim),
            nn.SyncBatchNorm(hidden_dim),
            nn.SyncBatchNorm(hidden_dim)
        ])

    def forward(self, x):
        # Code for matching with oss
        H_c, W_c = x.shape[2] // 16, x.shape[3] // 16
        H_toks, W_toks = x.shape[2] // self.patch_size, x.shape[3] // self.patch_size
        bs, C, h, w = x.shape

        if len(self.interaction_indexes) > 0 and not isinstance(self.dinov3, VisionTransformer):
            all_layers = self.dinov3.get_intermediate_layers(
                x, n=self.interaction_indexes, return_class_token=True
            )
        else:
            all_layers = self.dinov3(x)

        if len(all_layers) == 1:    # repeat the same layer for all the three scales
            all_layers = [all_layers[0], all_layers[0], all_layers[0]]
        
        sem_feats = []
        num_scales = len(all_layers) - 2
        for i, sem_feat in enumerate(all_layers):
            feat, _ = sem_feat
            sem_feat = feat.transpose(1, 2).view(bs, -1, H_c, W_c).contiguous()  # [B, D, H, W]
            resize_H, resize_W = int(H_c * 2**(num_scales-i)), int(W_c * 2**(num_scales-i))
            sem_feat = F.interpolate(sem_feat, size=[resize_H, resize_W], mode="bilinear", align_corners=False)
            sem_feats.append(sem_feat)

        # fusion
        fused_feats = []
        if self.use_sta:
            detail_feats = self.sta(x)
            for sem_feat, detail_feat in zip(sem_feats, detail_feats):
                fused_feats.append(torch.cat([sem_feat, detail_feat], dim=1))
        else:
            fused_feats = sem_feats

        c2 = self.norms[0](self.convs[0](fused_feats[0]))
        c3 = self.norms[1](self.convs[1](fused_feats[1]))
        c4 = self.norms[2](self.convs[2](fused_feats[2]))
        # out = self.mfcn({"features": [c2, c3, c4]})
        # feature_fused = out["feature_align"]
        # print(feature_fused.shape)

        # print("11")
        # print(c2.shape)
        # print(c3.shape)
        # print(c4.shape)

        return c2, c3, c4
        # return feature_fused 
