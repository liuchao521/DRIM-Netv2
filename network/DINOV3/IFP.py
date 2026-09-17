import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_dct as DCT
from mmcv.cnn import ConvModule
from mmengine.model import BaseModule


class DctSpatialInteractionDF(BaseModule):
    """Inconsistency-Guided DCT Spatial High-Frequency Interaction for Deepfake
    - 先在频域屏蔽低频，保留高频
    - 逆 DCT 得到高频响应图
    - 可选地用不一致性图 inc_map 引导空间注意力
    - 再把高频响应变成 [0,1] 的空间注意力，做残差调制
    """

    def __init__(self,
                 in_channels,
                 ratio=(0.25, 0.25),
                 isdct=True,
                 init_cfg=dict(type='Xavier', layer='Conv2d', distribution='uniform')):
        super().__init__(init_cfg)
        self.ratio = ratio
        self.isdct = isdct

        # 用高频响应生成空间 attention mask
        # 输入 1 通道（平均后的高频响应），输出 1 通道 mask
        self.spatial_att = nn.Sequential(
            ConvModule(1, 1, kernel_size=3, padding=1, bias=False),
        )

        # 残差缩放系数，可学习，初始很小避免一开始扰动太大
        self.gamma = nn.Parameter(torch.zeros(1))

        if not self.isdct:
            # 不用 DCT 时，退化为轻量空间注意力
            self.spatial1x1 = nn.Sequential(
                ConvModule(in_channels, 1, kernel_size=1, bias=False)
            )

    def forward(self, x, inc_map=None):
        """
        Args:
            x: (B, C, H, W) 特征
            inc_map: (B, 1, H', W') or None
                     由 DRCF / inconsistency 模块输出的不一致性图
        """
        b, c, h, w = x.size()

        # 轻量版：不做 DCT，直接空间注意力 + 残差
        if not self.isdct:
            mask = torch.sigmoid(self.spatial1x1(x))  # (B,1,H,W)

            if inc_map is not None:
                # 将不一致性图对齐到当前分辨率
                inc_resized = F.interpolate(
                    inc_map, size=(h, w),
                    mode='bilinear', align_corners=False
                )
                # 高频+不一致性双重约束
                mask = mask * inc_resized

            return x + self.gamma * (x * mask)

        orig_dtype = x.dtype
        x_float = x.float()

        # 1. 频域变换
        freq = DCT.dct_2d(x_float, norm='ortho')  # (B,C,H,W)

        # 2. 高通权重
        weight = self._compute_weight(h, w, self.ratio).to(x_float.device)
        weight = weight.view(1, 1, h, w).expand_as(freq)  # (B,C,H,W)

        freq_hp = freq * weight

        # 3. 逆 DCT 得到高频响应图
        high = DCT.idct_2d(freq_hp, norm='ortho')  # (B,C,H,W)

        # 4. 聚合成 1 通道空间图（关注“哪里”高频多）
        att_map = high.abs().mean(dim=1, keepdim=True)  # (B,1,H,W)

        # 4.1 如果有不一致性图，用它做空间引导
        if inc_map is not None:
            inc_resized = F.interpolate(
                inc_map, size=(h, w),
                mode='bilinear', align_corners=False
            )  # (B,1,H,W)
            # 只在“高频且不一致”的区域提升权重
            att_map = att_map * inc_resized

        # 5. 卷积 + sigmoid 生成 [0,1] 的空间 attention
        att_map = torch.sigmoid(self.spatial_att(att_map))  # (B,1,H,W)

        # 6. 残差方式调制原始特征
        out = x_float + self.gamma * (x_float * att_map)  # x + γ·(x⊙A)
        return out.to(orig_dtype)

    def _compute_weight(self, h, w, ratio):
        h0 = int(h * ratio[0])
        w0 = int(w * ratio[1])
        weight = torch.ones((h, w), requires_grad=False)
        weight[:h0, :w0] = 0  # 左上角低频区域屏蔽
        return weight
class DctChannelInteractionDF(BaseModule):
    """Inconsistency-Guided DCT Channel High-Frequency Interaction for Deepfake
    - 在高频响应上做全局池化
    - 可选地用不一致性图 inc_map 先对空间位置加权
    - 用 SE 风格的 MLP 得到通道权重
    - 残差调制通道响应
    """

    def __init__(self,
                 in_channels,
                 ratio=(0.25, 0.25),
                 reduction=16,
                 isdct=True,
                 init_cfg=dict(type='Xavier', layer='Conv2d', distribution='uniform')):
        super().__init__(init_cfg)
        self.in_channels = in_channels
        self.ratio = ratio
        self.isdct = isdct

        hidden = max(in_channels // reduction, 4)

        # SE 风格 MLP：C -> C/r -> C
        self.mlp = nn.Sequential(
            nn.Conv2d(in_channels, hidden, kernel_size=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, in_channels, kernel_size=1, bias=True)
        )

        self.gamma = nn.Parameter(torch.zeros(1))  # 残差缩放

    def forward(self, x, inc_map=None):
        """
        Args:
            x: (B, C, H, W)
            inc_map: (B, 1, H', W') or None
        """
        n, c, h, w = x.size()

        # 不用 DCT 时，就是普通 SE 通道注意力
        if not self.isdct:
            amaxp = F.adaptive_max_pool2d(x, 1)
            aavgp = F.adaptive_avg_pool2d(x, 1)
            channel = self.mlp(amaxp) + self.mlp(aavgp)
            weight_c = torch.sigmoid(channel)
            return x + self.gamma * (x * weight_c)

        orig_dtype = x.dtype
        x_float = x.float()

        # 1. 频域
        freq = DCT.dct_2d(x_float, norm='ortho')

        # 2. 高频
        weight = self._compute_weight(h, w, self.ratio).to(x_float.device)
        weight = weight.view(1, 1, h, w).expand_as(freq)
        freq_hp = freq * weight

        # 3. 回到空间域的高频响应
        high = DCT.idct_2d(freq_hp, norm='ortho')  # (B,C,H,W)

        # 3.1 如果有不一致性图，先用它做空间加权
        if inc_map is not None:
            inc_resized = F.interpolate(
                inc_map, size=(h, w),
                mode='bilinear', align_corners=False
            )  # (B,1,H,W)
            high = high * inc_resized  # 只关心“不一致”的高频响应

        # 4. 在高频图上做通道统计（关注“哪些通道”对高频敏感）
        amaxp = F.adaptive_max_pool2d(high, 1)  # (B,C,1,1)
        aavgp = F.adaptive_avg_pool2d(high, 1)  # (B,C,1,1)

        channel = self.mlp(amaxp) + self.mlp(aavgp)
        weight_c = torch.sigmoid(channel)  # (B,C,1,1)

        out = x_float + self.gamma * (x_float * weight_c)
        return out.to(orig_dtype)

    def _compute_weight(self, h, w, ratio):
        h0 = int(h * ratio[0])
        w0 = int(w * ratio[1])
        weight = torch.ones((h, w), requires_grad=False)
        weight[:h0, :w0] = 0
        return weight
class IFP(BaseModule):
    """Inconsistency-Guided High Frequency Perception for Deepfake
    - 空间高频注意力 + 通道高频注意力
    - 都用残差形式，减少对 backbone 的破坏
    - 可选地接受由 DRCF 产生的不一致性图 inc_map 进行引导
    """

    def __init__(self,
                 in_channels,
                 ratio=(0.25, 0.25),
                 isdct=True,
                 init_cfg=dict(type='Xavier', layer='Conv2d', distribution='uniform')):
        super().__init__(init_cfg)

        self.spatial = DctSpatialInteractionDF(in_channels, ratio=ratio, isdct=isdct)
        self.channel = DctChannelInteractionDF(in_channels, ratio=ratio, isdct=isdct)

        # 融合后再做一次 3×3 卷积 + GN 稍微清洗特征
        self.out = nn.Sequential(
            ConvModule(in_channels, in_channels, kernel_size=3, padding=1),
            nn.GroupNorm(8, in_channels)
        )

    def forward(self, x, inc_map=None):
        """
        Args:
            x: (B, C, H, W)
            inc_map: (B, 1, H', W') or None
        """
        # print("111111")
        s = self.spatial(x, inc_map=inc_map)
        c = self.channel(x, inc_map=inc_map)
        fused = s + c
        return self.out(fused)
