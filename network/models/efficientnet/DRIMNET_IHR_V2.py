import torch
from torch import nn
from torch.nn import functional as F
import kornia

import math
import os
import math

from timm.data import OPENAI_CLIP_MEAN, OPENAI_CLIP_STD

from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD, OPENAI_CLIP_MEAN, OPENAI_CLIP_STD
from timm.layers import trunc_normal_, AvgPool2dSame, DropPath, Mlp, GlobalResponseNormMlp, \
    LayerNorm2d, LayerNorm, create_conv2d, get_act_layer, make_divisible, to_ntuple

from typing import Callable, List, Optional, Tuple, Union  
from functools import partial


# from network.models.efficientnet.CRFB import  CRFB as TRFL
# from network.clip_encoder import CLIPEncoderStages as CLIPEncoder
from network.DINOV3.dinov3_adapter import DINOv3STAs as DINO
from network.DINOV3.mfcn import MFCN
# from network.DINOV3.mfcn1 import ModalityContrastFusion
# from network.DINOV3.hp1 import  HFP
from network.DINOV3.IFP import IFP as HFP
from network.DINOV3.DRCF import ContrastDrivenFeatureAggregation

class Downsample(nn.Module):

    def __init__(self, in_chs, out_chs, stride=1, dilation=1):
        super().__init__()
        avg_stride = stride if dilation == 1 else 1
        if stride > 1 or dilation > 1:
            avg_pool_fn = AvgPool2dSame if avg_stride == 1 and dilation > 1 else nn.AvgPool2d
            self.pool = avg_pool_fn(2, avg_stride, ceil_mode=True, count_include_pad=False)
        else:
            self.pool = nn.Identity()

        if in_chs != out_chs:
            self.conv = create_conv2d(in_chs, out_chs, 1, stride=1)
        else:
            self.conv = nn.Identity()

    def forward(self, x):
        x = self.pool(x)
        x = self.conv(x)
        return x
from .utils import (
    round_filters,
    round_repeats,
    drop_connect,
    get_same_padding_conv2d,
    get_model_params,
    efficientnet_params,
    load_pretrained_weights,
    Swish,
    MemoryEfficientSwish,
)

def conv1x1(inplanes, outplanes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(inplanes, outplanes, kernel_size=1, stride=stride, bias=False)


class MBConvBlock(nn.Module):

    def __init__(self, block_args, global_params):
        super().__init__()
        self._block_args = block_args
        self._bn_mom = 1 - global_params.batch_norm_momentum
        self._bn_eps = global_params.batch_norm_epsilon
        self.has_se = (self._block_args.se_ratio is not None) and (0 < self._block_args.se_ratio <= 1)
        self.id_skip = block_args.id_skip  # skip connection and drop connect

        # Get static or dynamic convolution depending on image size
        Conv2d = get_same_padding_conv2d(image_size=global_params.image_size)

        # Expansion phase
        inp = self._block_args.input_filters  # number of input channels
        oup = self._block_args.input_filters * self._block_args.expand_ratio  # number of output channels
        if self._block_args.expand_ratio != 1:
            
            self._expand_conv = Conv2d(in_channels=inp, out_channels=oup, kernel_size=1, bias=False)
            # self._expand_conv1 =CRU(op_channel=oup)
            self._bn0 = nn.BatchNorm2d(num_features=oup, momentum=self._bn_mom, eps=self._bn_eps)
            
            # self._bn0 = AdaMixBN(oup)

        # Depthwise convolution phase
        k = self._block_args.kernel_size
        s = self._block_args.stride
        self._depthwise_conv = Conv2d(
            in_channels=oup, out_channels=oup, groups=oup,  # groups makes it depthwise
            kernel_size=k, stride=s, bias=False)
        # self._depthwise_conv = CRU(op_channel=oup)
        self._bn1 = nn.BatchNorm2d(num_features=oup, momentum=self._bn_mom, eps=self._bn_eps)
        # self._bn1 = AdaMixBN(oup)
        

        # Squeeze and Excitation layer, if desired
        if self.has_se:
            num_squeezed_channels = max(1, int(self._block_args.input_filters * self._block_args.se_ratio))
            self._se_reduce = Conv2d(in_channels=oup, out_channels=num_squeezed_channels, kernel_size=1)
            self._se_expand = Conv2d(in_channels=num_squeezed_channels, out_channels=oup, kernel_size=1)
            # self.scc = SCC(in_channels =oup, op_channels=oup)
            # self.scc = SCC(op_channel=oup)

        # Output phase
        final_oup = self._block_args.output_filters
        # self._project_conv = SCC(final_oup)
        self._project_conv = Conv2d(in_channels=oup, out_channels=final_oup, kernel_size=1, bias=False)
        self._bn2 = nn.BatchNorm2d(num_features=final_oup, momentum=self._bn_mom, eps=self._bn_eps)
        # self._bn2 = AdaMixBN(final_oup)
        self._swish = MemoryEfficientSwish()

    def forward(self, inputs, drop_connect_rate=None):
        """
        :param inputs: input tensor
        :param drop_connect_rate: drop connect rate (float, between 0 and 1)
        :return: output of block
        """
        # print(self._block_args.input_filters * self._block_args.expand_ratio)
        # Expansion and Depthwise Convolution
        x = inputs
        
        if self._block_args.expand_ratio != 1:
            # print("12",x.shape)
            x = self._expand_conv(inputs)
            # x = self._expand_conv1(x)
            # print("_expand_conv",x.shape)
            
            x = self._swish(self._bn0(x))
            
        x = self._swish(self._bn1(self._depthwise_conv(x)))

        # Squeeze and Excitation
        if self.has_se:
            x_squeezed = F.adaptive_avg_pool2d(x, 1)
            # print("x_squeezed",x_squeezed.shape)
            x_squeezed = self._se_expand(self._swish(self._se_reduce(x_squeezed)))
            # x_squeezed = self.scc(x_squeezed)
            x = torch.sigmoid(x_squeezed) * x

        x = self._bn2(self._project_conv(x))

        # Skip connection and drop connect
        input_filters, output_filters = self._block_args.input_filters, self._block_args.output_filters
        if self.id_skip and self._block_args.stride == 1 and input_filters == output_filters:
            if drop_connect_rate:
                x = drop_connect(x, p=drop_connect_rate, training=self.training)
            x = x + inputs  # skip connection
        return x

    def set_swish(self, memory_efficient=True):
        """Sets swish function as memory efficient (for training) or standard (for export)"""
        self._swish = MemoryEfficientSwish() if memory_efficient else Swish()



class EfficientNet(nn.Module):
    """
    An EfficientNet model. Most easily loaded with the .from_name or .from_pretrained methods

    Args:
        blocks_args (list): A list of BlockArgs to construct blocks
        global_params (namedtuple): A set of GlobalParams shared between blocks

    Example:
        model = EfficientNet.from_pretrained('efficientnet-b0')

    """

    def __init__(self, blocks_args=None, global_params=None,escape=''):
        super().__init__()
        assert isinstance(blocks_args, list), 'blocks_args should be a list'
        assert len(blocks_args) > 0, 'block args must be greater than 0'
        self.escape=escape
        self._global_params = global_params
        self._blocks_args = blocks_args
        # Get static or dynamic convolution depending on image size
        Conv2d = get_same_padding_conv2d(image_size=global_params.image_size)

        # Batch norm parameters
        bn_mom = 1 - self._global_params.batch_norm_momentum
        bn_eps = self._global_params.batch_norm_epsilon

        # Stem
        in_channels = 3  # rgb
        out_channels = round_filters(32, self._global_params)  # number of output channels
        # self._conv_stem =SCC(out_channels)
        self._conv_stem = Conv2d(in_channels, out_channels, kernel_size=3, stride=2, bias=False)
        self._bn0 = nn.BatchNorm2d(num_features=out_channels, momentum=bn_mom, eps=bn_eps)  #这里进行了修改
        # crossnorm= CrossNorm(out_channels)
        # selfnorm = SelfNorm(out_channels)
        # self.cs = CrossNorm(out_channels)
        # self.cnsn = CNSN(crossnorm=crossnorm, selfnorm=selfnorm)

        # self._bn0 = AdaMixBN(out_channels)
        # Build blocks
        self._blocks = nn.ModuleList([])
        self.stage_map=[]
        stage_count=0
        for block_args in self._blocks_args:

            # Update block input and output filters based on depth multiplier.
            block_args = block_args._replace(
                input_filters=round_filters(block_args.input_filters, self._global_params),
                output_filters=round_filters(block_args.output_filters, self._global_params),
                num_repeat=round_repeats(block_args.num_repeat, self._global_params)
            )
            stage_count+=1
            self.stage_map+=['']*(block_args.num_repeat - 1)
            self.stage_map.append('b%s'%stage_count)
            # The first block needs to take care of stride and filter size increase.
            self._blocks.append(MBConvBlock(block_args, self._global_params))
            
            if block_args.num_repeat > 1:
                block_args = block_args._replace(input_filters=block_args.output_filters, stride=1)
            for _ in range(block_args.num_repeat - 1):
                self._blocks.append(MBConvBlock(block_args, self._global_params))

        # Head
        in_channels = block_args.output_filters  # output of final block
        out_channels = round_filters(1280, self._global_params)
        self._conv_head = Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self._bn1 = nn.BatchNorm2d(num_features=out_channels, momentum=bn_mom, eps=bn_eps)#
        # self._bn1 = AdaMixBN(out_channels)

        # Final linear layer
        self._avg_pooling = nn.AdaptiveAvgPool2d(1)
        self._dropout = nn.Dropout(self._global_params.dropout_rate)
        self._fc = nn.Linear(out_channels, self._global_params.num_classes)
        self._swish = MemoryEfficientSwish()
        # self.last = last(1792,7,7)
        # self.last = EMA(1792)
        # self.last = CRU_Modified(1792)
        # self.first = last(48,112,112)
        # self.first = EMA(48)
    def set_swish(self, memory_efficient=True):
        """Sets swish function as memory efficient (for training) or standard (for export)"""
        self._swish = MemoryEfficientSwish() if memory_efficient else Swish()
        for block in self._blocks:
            block.set_swish(memory_efficient)


    def extract_features(self, inputs,layers):
        """ Returns output of the final convolution layer """
        # Stem
        x = self._swish(self._bn0(self._conv_stem(inputs)))
        # x = self.first(x)+x
        # print(x.shape)
        layers['b0']=x
        # Blocks
        for idx, block in enumerate(self._blocks):
            drop_connect_rate = self._global_params.drop_connect_rate
            if drop_connect_rate:
                drop_connect_rate *= float(idx) / len(self._blocks)
            x = block(x, drop_connect_rate=drop_connect_rate)
            stage=self.stage_map[idx]
            if stage:     
                layers[stage]=x
                if stage==self.escape:
                    return None
        # Head
        x = self._bn1(self._conv_head(x))
        # x2 = self.cnsn(x)
        x=self._swish(x)
        return x

    def forward(self, x):
        """ Calls extract_features to extract features, applies final linear layer, and returns logits. """
        bs = x.size(0)
        # print("x",x.shape)
        layers={}
        x = self.extract_features(x,layers) #torch.Size([32, 1792, 7, 7])
        
        # print(x.shape)
        if x is None:
            return layers
        layers['final']=x
        x = self._avg_pooling(x)
        x = x.view(bs, -1)
        x = self._dropout(x)
        x = self._fc(x)

        return layers

    @classmethod
    def from_name(cls, model_name, override_params=None,escape=''):
        cls._check_model_name_is_valid(model_name)
        blocks_args, global_params = get_model_params(model_name, override_params)
        return cls(blocks_args, global_params,escape)

    @classmethod
    def from_pretrained(cls, model_name, advprop=False, num_classes=1000, in_channels=3,escape=''):
        model = cls.from_name(model_name, override_params={'num_classes': num_classes},escape=escape)
        load_pretrained_weights(model, model_name, load_fc=(num_classes == 1000), advprop=advprop)
        if in_channels != 3:
            Conv2d = get_same_padding_conv2d(image_size = model._global_params.image_size)
            out_channels = round_filters(32, model._global_params)
            model._conv_stem = Conv2d(in_channels, out_channels, kernel_size=3, stride=2, bias=False)
        return model
    
    @classmethod
    def get_image_size(cls, model_name):
        cls._check_model_name_is_valid(model_name)
        _, _, res, _ = efficientnet_params(model_name)
        return res

    @classmethod
    def _check_model_name_is_valid(cls, model_name):
        """ Validates model name. """ 
        valid_models = ['efficientnet-b'+str(i) for i in range(9)]
        if model_name not in valid_models:
            raise ValueError('model_name should be one of: ' + ', '.join(valid_models))
        
class ConvBN(torch.nn.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size=1, stride=1, padding=0, dilation=1, groups=1, with_bn=True):
        super().__init__()
        self.add_module('conv', torch.nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, dilation, groups))
        if with_bn:
            self.add_module('bn', torch.nn.BatchNorm2d(out_planes))
            torch.nn.init.constant_(self.bn.weight, 1)
            torch.nn.init.constant_(self.bn.bias, 0)
class Block(nn.Module):
    def __init__(self, dim, mlp_ratio=3, drop_path=0.):
        super().__init__()
        self.dwconv = ConvBN(dim, dim, 7, 1, (7 - 1) // 2, groups=dim, with_bn=True)
        self.f1 = ConvBN(dim, mlp_ratio * dim, 1, with_bn=False)
        self.f2 = ConvBN(dim, mlp_ratio * dim, 1, with_bn=False)
        self.g = ConvBN(mlp_ratio * dim, dim, 1, with_bn=True)
        self.dwconv2 = ConvBN(dim, dim, 7, 1, (7 - 1) // 2, groups=dim, with_bn=False)
        self.act = nn.ReLU6()
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x1, x2 = self.f1(x), self.f2(x)
        x = self.act(x1) * x2
        x = self.dwconv2(self.g(x))
        x = input + self.drop_path(x)
        return x          
class NormProj(nn.Module):
    """
    轻量归一化 + 1x1 投影：
    - 默认用 GroupNorm(1, C)（等效 LayerNorm2d），比 BN 稳
    """
    def __init__(self, cin, cout, norm='gn'):
        super().__init__()
        if norm == 'gn':
            self.norm = nn.GroupNorm(1, cin)
        elif norm == 'bn':
            self.norm = nn.BatchNorm2d(cin)
        else:
            self.norm = nn.Identity()
        self.proj = nn.Conv2d(cin, cout, 1)

    def forward(self, x):
        return self.proj(self.norm(x))
class cranet(nn.Module):
    """
    EfficientNet(b1/b3/b5) ↔ CLIP(所有层patch特征) 融合网络
    - 可学习层权重：每个桥一组 L 维 logits（softmax 融合各层）
    - 每个桥带门控 gate（SE 风格）
    - Norm+投影：query/key 在 1x1 前做轻量归一化（默认 GroupNorm(1,C)）
    - 2D 位置编码：对齐到 28×28 后，给每个分支和 final_feat 注入可学习 pos2d
    - 默认带分类头（BCEWithLogitsLoss：out_classes=1）
    """
    def __init__(
        self,
        embed_dim=256,
        model_name1='efficientnet-b4',
        advprop1=True,
        num_classes1=1000,
        device=None,
        out_classes=2,
        use_head=True,
        norm_type='gn'   # 'gn' | 'bn' | 'id'
    ):
        super().__init__()
        self.device   = device
        self.use_head = use_head
        self.EFF = EfficientNet.from_pretrained(model_name=model_name1, advprop=advprop1, num_classes=num_classes1)
        self.dino = DINO()
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(768, num_classes1)
        )
        # Preserve coarse spatial evidence and distribution statistics at each scale.
        self.inc_proj = nn.Sequential(
            nn.Linear(57, 128),
            nn.LayerNorm(128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 768),
        )
        self.inc_classifier = nn.Linear(768, num_classes1)
        self.reduce_28 = nn.Conv2d(312, 256, 1)
        self.reduce_14 = nn.Conv2d(368, 256, 1)
        self.reduce_7  = nn.Conv2d(528, 256, 1)
        # from model.CDFA import ContrastDrivenFeatureAggregation

        # self.cdfa_28 = ModalityContrastFusion(in_c=256, dim=256)
        # self.cdfa_14 = ModalityContrastFusion(in_c=256, dim=256)
        # self.cdfa_7  = ModalityContrastFusion(in_c=256, dim=256)
        self.cdfa_28 = ContrastDrivenFeatureAggregation(in_c=256, dim=256, num_heads=4)
        self.cdfa_14 = ContrastDrivenFeatureAggregation(in_c=256, dim=256, num_heads=4)
        self.cdfa_7  = ContrastDrivenFeatureAggregation(in_c=256, dim=256, num_heads=4)

        self.align_b3 = nn.Conv2d(56, 256, 1)
        self.align_b4 = nn.Conv2d(112, 256, 1)
        self.align_b6 = nn.Conv2d(272, 256, 1)

        # self.HFP3 = HFP(56, (0.25, 0.25))
        # self.HFP4 = HFP(112, ratio=(0.1, 0.1))
        self.HFP1= HFP(256, (0.25, 0.25), isdct=False)#28 
        self.HFP2 = HFP(256, ratio=(0.1, 0.1), isdct=False)#14
        self.HFP5 = HFP(256, ratio=(0.1, 0.1), isdct=True)#7

        # for p in self.dino.parameters():
        #     p.requires_grad = False

        self.mfcn = MFCN(
                    inplanes=[256, 256, 256],
                    outplanes=[256*3],
                    instrides=[8, 16, 32],
                    outstrides=[8]
                )

    # ------------------------------ 前向 ------------------------------
    def forward(
        self,
        x_im: torch.Tensor,
        pixel_values: torch.Tensor = None,
        return_gate=False,
        return_embedding=False,
        return_inc_embedding=False,
    ):

        # EfficientNet
        layers = self.EFF(x_im)
        final  = layers["final"]              # [B, 1792, 7, 7]
        b4, b3, b5 = layers["b4"], layers["b3"], layers["b5"]
        b6 = layers["b6"]

        c2,c3,c4 = self.dino(x_im)
        # b3 = self.HFP3(b3)
        # b4 = self.HFP4(b4)

        # print("b4",b4.shape)
        # print("b3",b3.shape)
        # print("b5",b5.shape)
        # print("b6",b6.shape)
        # print("c2",c2.shape)
        # print("c3",c3.shape)
        # print("c4",c4.shape)

        # b4 torch.Size([48, 112, 14, 14])
        # b3 torch.Size([48, 56, 28, 28])
        # b5 torch.Size([48, 160, 14, 14])
        # b6 torch.Size([48, 272, 7, 7])
        # c2 torch.Size([48, 256, 28, 28])
        # c3 torch.Size([48, 256, 14, 14])
        # c4 torch.Size([48, 256, 7, 7])
        # fused_28 = torch.cat([b3, c2], dim=1)#312
        # fused_14 = torch.cat([b4, c3], dim=1)#368
        # fused_7  = torch.cat([b6, c4], dim=1)#528
        # fused_28 = self.reduce_28(fused_28)
        # fused_14 = self.reduce_14(fused_14)
        # fused_7  = self.reduce_7(fused_7)

        # fused_28 = self.cdfa_28(fused_28, c2, b3)
        # fused_14 = self.cdfa_14(fused_14, c3, b4)
        # # fused_7  = self.cdfa_7(fused_7,  c4, b6)
        # fused_28 = self.cdfa_28(fused_28, c2, self.align_b3(b3))
        # fused_28 = self.HFP1(fused_28 )
        # # print(fused_28.shape)

        # fused_14 = self.cdfa_14(fused_14, c3, self.align_b4(b4))
        # fused_14 = self.HFP2(fused_14)
        # # print(fused_14.shape)

        # fused_7  = self.cdfa_7(fused_7,  c4, self.align_b6(b6))
        # fused_7 = self.HFP5(fused_7 )
        # print(fused_7.shape)
        fused_28 = torch.cat([b3, c2], dim=1)   # [B,312,28,28]
        fused_14 = torch.cat([b4, c3], dim=1)   # [B,368,14,14]
        fused_7  = torch.cat([b6, c4], dim=1)   # [B,528,7,7]

        fused_28 = self.reduce_28(fused_28)     # [B,256,28,28]
        fused_14 = self.reduce_14(fused_14)     # [B,256,14,14]
        fused_7  = self.reduce_7(fused_7)       # [B,256,7,7]

        # ------ DRCF + IG-AHFP (28×28) ------
        fused_28, inc_28 = self.cdfa_28(
            fused_28,          # x
            c2,                # fg: DINO 结构分支
            self.align_b3(b3)  # bg: EfficientNet 纹理分支对齐到 256
        )
        fused_28 = self.HFP1(fused_28, inc_map=inc_28)

        # ------ DRCF + IG-AHFP (14×14) ------
        fused_14, inc_14 = self.cdfa_14(
            fused_14,
            c3,
            self.align_b4(b4)
        )
        fused_14 = self.HFP2(fused_14, inc_map=inc_14)

        # ------ DRCF + IG-AHFP (7×7) ------
        fused_7, inc_7 = self.cdfa_7(
            fused_7,
            c4,
            self.align_b6(b6)
        )
        fused_7 = self.HFP5(fused_7, inc_map=inc_7)

        def summarize_inc_map(inc_map):
            coarse = F.adaptive_avg_pool2d(inc_map, (4, 4)).flatten(1)
            flat = inc_map.flatten(1)
            stats = torch.stack(
                [
                    flat.mean(dim=1),
                    flat.std(dim=1, unbiased=False),
                    flat.amax(dim=1),
                ],
                dim=1,
            )
            return torch.cat([coarse, stats], dim=1)

        inc_descriptor = torch.cat(
            [summarize_inc_map(x) for x in (inc_28, inc_14, inc_7)],
            dim=1,
        )
        inc_embedding = self.inc_proj(inc_descriptor)
        inc_embedding = F.normalize(inc_embedding, p=2, dim=1)
        inc_logits = self.inc_classifier(inc_embedding)



        # mfcn = MFCN(
        #     inplanes=[256, 256, 256],
        #     outplanes=[256*3],
        #     instrides=[8, 16, 32],
        #     outstrides=[8]
        # )
        out = self.mfcn({"features": [fused_28, fused_14, fused_7]})
        # out = self.mfcn({"features": [c2, c3, c4]})
        fused_feature = out["feature_align"]
        # print(fused_feature.shape)  # t
        # b3 torch.Size([64, 56, 28, 28])
        # b4 torch.Size([64, 112, 14, 14])
        # b6 torch.Size([64, 272, 7, 7])
        # c2 torch.Size([64, 256, 28, 28])
        # c3 torch.Size([64, 256, 14, 14])
        # c4 torch.Size([64, 256, 7, 7])
        x = self.classifier[0](fused_feature)
        embedding = self.classifier[1](x)
        embedding = F.normalize(embedding, p=2, dim=1)
        logits = self.classifier[2](embedding)
        if return_embedding and return_inc_embedding:
            return logits, embedding, inc_embedding, inc_logits
        if return_embedding:
            return logits, embedding
        return logits  # [B, num_classes]

        # exit()

        # b4 torch.Size([64, 112, 14, 14])
        # b3 torch.Size([64, 56, 28, 28])
        # b5 torch.Size([64, 160, 14, 14])
        # c2 torch.Size([64, 256, 28, 28])
        # c3 torch.Size([64, 256, 14, 14])
        # c4 torch.Size([64, 256, 7, 7])



        # # CLIP
        # if pixel_values is None:
        #     pixel_values = self._to_clip_pixels(x_im)
        # patch_tokens, clip_cls, clip_map = self.clip1.forward_all(pixel_values)
        # B, L, HW, D = patch_tokens.shape
        # side = int(math.isqrt(HW))
        # assert side * side == HW, f"HW={HW} 不是完全平方"

        # # -> [B, L, D, h, w]
        # patch_maps = patch_tokens.permute(0, 1, 3, 2).contiguous().view(B, L, D, side, side)

        # # 层权
        # self._ensure_layer_logits(L, device=patch_maps.device)
        # k_b5 = self._fuse_layers_weighted(patch_maps, self.layer_logits[0])  # [B,1024,h,w]
        # k_b3 = self._fuse_layers_weighted(patch_maps, self.layer_logits[1])
        # k_b1 = self._fuse_layers_weighted(patch_maps, self.layer_logits[2])

        # # —— Norm+投影（query/key）—— #
        # q_b5 = self.query_proj[0](b5)  # [B,embed_dim,*,*]
        # q_b3 = self.query_proj[1](b3)
        # q_b1 = self.query_proj[2](b1)
        # k_b5 = self.key_proj[0](k_b5)  # [B,embed_dim,*,*]
        # k_b3 = self.key_proj[1](k_b3)
        # k_b1 = self.key_proj[2](k_b1)

        # # 融合（CABlock）
        # bridge_feats, gate_vals = [], []
        # prev_fused = None  # 用来存储前一个 stage 的输出

        # for i, (q, k) in enumerate([(q_b5, k_b5), (q_b3, k_b3), (q_b1, k_b1)]):

        #     # --- 构造 query 列表 ---
        #     if prev_fused is not None:
        #         query_in = [prev_fused, q]   # 把上一个 stage 的 fused 加进来
        #     else:
        #         query_in = [q]               # 第一个 stage 只用自己的 q

        #     fused = self.cablocks[i](query_in, k)  # 支持 list query
        #     fused = F.interpolate(fused, size=self.target_size, mode='bilinear', align_corners=False)

        #     # —— 加 2D 位置编码 —— #
        #     fused = fused + self.pos2d

        #     # —— 门控 —— #
        #     g = self.gates[i](fused)          # [B,1,1,1], 0~1
        #     fused = fused * g

        #     # 保存结果
        #     bridge_feats.append(fused)
        #     gate_vals.append(g.squeeze(-1).squeeze(-1).squeeze(-1))  
        #     prev_fused = fused

        # fu1, fu2, fu3 = bridge_feats
        # clip_feat = self.clip_proj(clip_map)  # Linear(D, embed_dim)
        # clip_feat = F.interpolate(clip_feat, size=self.target_size, mode="bilinear", align_corners=False)
        # g_clip = self.clip_gate(clip_feat)   # [B,1,1,1]
        # clip_feat = clip_feat * g_clip
        # final_feat = self.final_proj(final)                          # [B,embed_dim,7,7]
        # final_feat = F.interpolate(final_feat, size=self.target_size, mode='bilinear', align_corners=False)
        # final_feat = final_feat + self.pos2d
        # g_final = self.final_gate(final_feat)   # [B,1,1,1]
        # final_feat = final_feat * g_final
        # final_last = self.trfl([fu3, final_feat, clip_feat]) 
        # x = self._avg_pooling(final_last)                            # [B,4*embed_dim,1,1]
        # x = torch.flatten(x, 1)                                      # [B,4*embed_dim]
        # x = self._dropout(x)

        # if self.use_head:
        #     logits = self.head(x)                                    # [B,out_classes]
        #     if return_gate:
        #         return logits, gate_vals, [torch.softmax(p, dim=0) for p in self.layer_logits]
        #     return logits
        # else:
        #     if return_gate:
        #         return x, gate_vals, [torch.softmax(p, dim=0) for p in self.layer_logits]
            # return x
       
