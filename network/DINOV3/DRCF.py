
import torch
import torch.nn as nn
import math
import torch.nn.functional as F

class CBR(nn.Module):
    def __init__(self, in_c, out_c, kernel_size=3, padding=1, dilation=1, stride=1, act=True):
        super().__init__()
        self.act = act
        self.conv = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size, padding=padding, dilation=dilation, bias=False, stride=stride),
            nn.BatchNorm2d(out_c)
        )
        self.relu = nn.ReLU(inplace=True)
    def forward(self, x):
        x = self.conv(x)
        if self.act == True:
            x = self.relu(x)
        return x
    
class ContrastDrivenFeatureAggregation(nn.Module):
    def __init__(self, in_c, dim, num_heads, kernel_size=3, padding=1, stride=1,
                 attn_drop=0., proj_drop=0.):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.v = nn.Linear(dim, dim)
        self.attn_fg = nn.Linear(dim, kernel_size ** 4 * num_heads)
        self.attn_bg = nn.Linear(dim, kernel_size ** 4 * num_heads)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.unfold = nn.Unfold(kernel_size=kernel_size, padding=padding, stride=stride)
        self.pool = nn.AvgPool2d(kernel_size=stride, stride=stride, ceil_mode=True)
        self.input_cbr = nn.Sequential(
            CBR(in_c, dim, kernel_size=3, padding=1),
            CBR(dim, dim, kernel_size=3, padding=1),
        )
        self.output_cbr = nn.Sequential(
            CBR(dim, dim, kernel_size=3, padding=1),
            CBR(dim, dim, kernel_size=3, padding=1),
        )
        self.fg_align = nn.Conv2d(in_c, dim, 1)
        self.bg_align = nn.Conv2d(in_c, dim, 1)
    def forward(self, x, fg, bg):
        x = self.input_cbr(x)
        x = x.permute(0, 2, 3, 1)  # (B,H,W,C)
        fg = fg.permute(0, 2, 3, 1)
        bg = bg.permute(0, 2, 3, 1)
        # 如果 fg/bg 还是 in_c 维，这里可以打开 align：
        # fg = self.fg_align(fg.permute(0,3,1,2)).permute(0,2,3,1)
        # bg = self.bg_align(bg.permute(0,3,1,2)).permute(0,2,3,1)

        B, H, W, C = x.shape

        # value 特征
        v = self.v(x).permute(0, 3, 1, 2)  # (B,C,H,W)
        v_unfolded = self.unfold(v).reshape(
            B, self.num_heads, self.head_dim,
            self.kernel_size * self.kernel_size,
            -1
        ).permute(0, 1, 4, 3, 2)  # (B,heads,L,ks^2,head_dim)

        # 1) foreground attention
        attn_fg = self.compute_attention(fg, B, H, W, C, 'fg')  # (B,heads,L,ks^2,ks^2)
        x_weighted_fg = self.apply_attention(attn_fg, v_unfolded, B, H, W, C)

        # 2) background attention，作用在前一步结果上
        v_unfolded_bg = self.unfold(
            x_weighted_fg.permute(0, 3, 1, 2)
        ).reshape(
            B, self.num_heads, self.head_dim,
            self.kernel_size * self.kernel_size,
            -1
        ).permute(0, 1, 4, 3, 2)

        attn_bg = self.compute_attention(bg, B, H, W, C, 'bg')
        x_weighted_bg = self.apply_attention(attn_bg, v_unfolded_bg, B, H, W, C)
        x_weighted_bg = x_weighted_bg.permute(0, 3, 1, 2)  # (B,C,H,W)

        out = self.output_cbr(x_weighted_bg)  # (B,C,H,W)

        # 3) ★ 由前景/背景注意力差异构造 inconsistency map
        # attn_fg / attn_bg: (B,heads,L,ks^2,ks^2), L = h*w
        h = math.ceil(H / self.stride)
        w = math.ceil(W / self.stride)
        # 对 head、patch 中心/邻域做平均，得到每个位置一个标量不一致性
        inc_token = (attn_fg - attn_bg).pow(2).mean(dim=(1, 3, 4))  # (B,L)
        inc_map_coarse = inc_token.reshape(B, 1, h, w)              # (B,1,h,w)
        # 上采样到与 x 同分辨率
        inc_map = F.interpolate(
            inc_map_coarse, size=(H, W),
            mode='bilinear', align_corners=False
        )  # (B,1,H,W)

        return out, inc_map

    
    def compute_attention(self, feature_map, B, H, W, C, feature_type):
        attn_layer = self.attn_fg if feature_type == 'fg' else self.attn_bg
        h, w = math.ceil(H / self.stride), math.ceil(W / self.stride)
        feature_map_pooled = self.pool(feature_map.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        attn = attn_layer(feature_map_pooled).reshape(B, h * w, self.num_heads,
                                                      self.kernel_size * self.kernel_size,
                                                      self.kernel_size * self.kernel_size).permute(0, 2, 1, 3, 4)
        attn = attn * self.scale
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)
        return attn
    def apply_attention(self, attn, v, B, H, W, C):
        x_weighted = (attn @ v).permute(0, 1, 4, 3, 2).reshape(
            B, self.dim * self.kernel_size * self.kernel_size, -1)
        x_weighted = F.fold(x_weighted, output_size=(H, W), kernel_size=self.kernel_size,
                            padding=self.padding, stride=self.stride)
        x_weighted = self.proj(x_weighted.permute(0, 2, 3, 1))
        x_weighted = self.proj_drop(x_weighted)
        return x_weighted
