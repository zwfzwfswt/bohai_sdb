# -*- coding: utf-8 -*-
"""
Swin-BathyUNet++（自研改进模型）。

改进点：
  1. 多尺度光谱嵌入（Multi-Scale Spectral Embedding）
  2. SE 通道注意力（Squeeze-and-Excitation）
  3. 密集跳连（Dense Skip Connections, 参考 UNet++）
  4. 深度分段感知损失（外部实现）
  5. 可学习相对位置偏置
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from .swin_bathyunet import (
    SwinTransformerBlock,
    VisionTransformerStage,
    window_partition,
    window_reverse,
)


# ===================== SE 通道注意力 =====================

class SEBlock(nn.Module):
    """Squeeze-and-Excitation：自适应通道加权。"""

    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels, max(channels // reduction, 4)),
            nn.ReLU(inplace=True),
            nn.Linear(max(channels // reduction, 4), channels),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.fc(x).unsqueeze(-1).unsqueeze(-1)
        return x * w


# ===================== 多尺度光谱嵌入 =====================

class MultiScaleSpectralEmbedding(nn.Module):
    """
    并行多尺度卷积 (1×1, 3×3, 5×5)，融合光谱与空间特征。
    输出通道 = out_channels。
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        mid = out_channels // 3

        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels, mid, 1, bias=False),
            nn.BatchNorm2d(mid),
            nn.ReLU(inplace=True),
        )
        self.branch3 = nn.Sequential(
            nn.Conv2d(in_channels, mid, 3, padding=1, bias=False),
            nn.BatchNorm2d(mid),
            nn.ReLU(inplace=True),
        )
        self.branch5 = nn.Sequential(
            nn.Conv2d(in_channels, mid, 5, padding=2, bias=False),
            nn.BatchNorm2d(mid),
            nn.ReLU(inplace=True),
        )
        rest = out_channels - mid * 3
        self.fuse = nn.Sequential(
            nn.Conv2d(mid * 3, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        self.se = SEBlock(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b1 = self.branch1(x)
        b3 = self.branch3(x)
        b5 = self.branch5(x)
        out = self.fuse(torch.cat([b1, b3, b5], dim=1))
        return self.se(out)


# ===================== Conv Block + SE =====================

class ConvBlockSE(nn.Module):
    """Conv-BN-ReLU ×2 + SE。"""

    def __init__(self, in_ch: int, out_ch: int, dropout: float = 0.0):
        super().__init__()
        layers = [
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        ]
        if dropout > 0:
            layers.append(nn.Dropout2d(dropout))
        self.conv = nn.Sequential(*layers)
        self.se = SEBlock(out_ch)

    def forward(self, x):
        return self.se(self.conv(x))


# ===================== Dense Skip Adapter =====================

class DenseSkipAdapter(nn.Module):
    """
    密集跳连适配器：融合当前层 skip + 上一层 skip（上采样后）。
    实现 UNet++ 风格的跨层连接。
    """

    def __init__(self, ch_current: int, ch_lower: int):
        super().__init__()
        self.up_lower = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(ch_lower, ch_current, 1, bias=False),
            nn.BatchNorm2d(ch_current),
        )
        self.fuse = nn.Sequential(
            nn.Conv2d(ch_current * 2, ch_current, 1, bias=False),
            nn.BatchNorm2d(ch_current),
            nn.ReLU(inplace=True),
        )

    def forward(self, skip_current: torch.Tensor,
                skip_lower: torch.Tensor) -> torch.Tensor:
        lower_up = self.up_lower(skip_lower)
        lower_up = F.interpolate(lower_up, size=skip_current.shape[2:],
                                 mode="bilinear", align_corners=False)
        return self.fuse(torch.cat([skip_current, lower_up], dim=1))


# ===================== Swin-BathyUNet++ =====================

class SwinBathyUNetPP(nn.Module):
    """
    自研 Swin-BathyUNet++ 模型。

    参数：
        in_channels:      输入波段数（默认 8）
        base_channels:    基础通道数（默认 64）
        swin_depth:       Swin block 层数
        swin_heads:       多头注意力头数
        swin_window_size: 窗口大小
        dropout:          Dropout 比例
    """

    def __init__(self, in_channels: int = 8, base_channels: int = 64,
                 swin_depth: int = 2, swin_heads: int = 8,
                 swin_window_size: int = 8, dropout: float = 0.1):
        super().__init__()
        bc = base_channels

        # ── 多尺度光谱嵌入 ──
        self.stem = MultiScaleSpectralEmbedding(in_channels, bc)

        # ── Encoder (with SE) ──
        self.enc1 = ConvBlockSE(bc, bc)
        self.enc2 = ConvBlockSE(bc, bc * 2)
        self.enc3 = ConvBlockSE(bc * 2, bc * 4)
        self.enc4 = ConvBlockSE(bc * 4, bc * 8, dropout=dropout)

        # ── Bottleneck + Swin ──
        self.bottleneck = ConvBlockSE(bc * 8, bc * 16, dropout=dropout)
        self.bottleneck_swin = VisionTransformerStage(
            bc * 16, depth=swin_depth, num_heads=swin_heads,
            window_size=swin_window_size, dropout=dropout,
        )

        # ── Dense Skip Adapters ──
        self.dense_skip3 = DenseSkipAdapter(bc * 4, bc * 8)   # e3 + e4↑
        self.dense_skip2 = DenseSkipAdapter(bc * 2, bc * 4)   # e2 + e3↑

        # ── Swin on skip features ──
        self.vit4 = VisionTransformerStage(bc * 8, swin_depth, swin_heads,
                                           swin_window_size, dropout)
        self.vit3 = VisionTransformerStage(bc * 4, swin_depth, swin_heads,
                                           swin_window_size, dropout)

        # ── Decoder ──
        self.up4 = nn.ConvTranspose2d(bc * 16, bc * 8, 2, stride=2)
        self.dec4 = ConvBlockSE(bc * 16, bc * 8)

        self.up3 = nn.ConvTranspose2d(bc * 8, bc * 4, 2, stride=2)
        self.dec3 = ConvBlockSE(bc * 8, bc * 4)

        self.up2 = nn.ConvTranspose2d(bc * 4, bc * 2, 2, stride=2)
        self.dec2 = ConvBlockSE(bc * 4, bc * 2)

        self.up1 = nn.ConvTranspose2d(bc * 2, bc, 2, stride=2)
        self.dec1 = ConvBlockSE(bc * 2, bc)

        # ── Regression Head ──
        self.head = nn.Conv2d(bc, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, C, H, W) → (B, 1, H, W)"""

        # Stem
        x0 = self.stem(x)                              # (B, bc, H, W)

        # Encoder
        e1 = self.enc1(x0)                             # (B, bc,   H,   W)
        e2 = self.enc2(F.max_pool2d(e1, 2))            # (B, bc*2, H/2, W/2)
        e3 = self.enc3(F.max_pool2d(e2, 2))            # (B, bc*4, H/4, W/4)
        e4 = self.enc4(F.max_pool2d(e3, 2))            # (B, bc*8, H/8, W/8)

        # Bottleneck
        b = self.bottleneck(F.max_pool2d(e4, 2))       # (B, bc*16, H/16, W/16)
        b = self.bottleneck_swin(b)

        # Dense skip: 融合跨层信息
        e3_dense = self.dense_skip3(e3, e4)             # e3 + e4↑
        e2_dense = self.dense_skip2(e2, e3)             # e2 + e3↑

        # Swin 增强 skip features
        e4_swin = self.vit4(e4)
        e3_swin = self.vit3(e3_dense)

        # Decoder
        d4 = self.up4(b)
        d4 = self._align_cat(d4, e4_swin)
        d4 = self.dec4(d4)

        d3 = self.up3(d4)
        d3 = self._align_cat(d3, e3_swin)
        d3 = self.dec3(d3)

        d2 = self.up2(d3)
        d2 = self._align_cat(d2, e2_dense)
        d2 = self.dec2(d2)

        d1 = self.up1(d2)
        d1 = self._align_cat(d1, e1)
        d1 = self.dec1(d1)

        return self.head(d1)

    @staticmethod
    def _align_cat(up: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        """对齐尺寸后拼接。"""
        if up.shape[2:] != skip.shape[2:]:
            up = F.interpolate(up, size=skip.shape[2:],
                               mode="bilinear", align_corners=False)
        return torch.cat([up, skip], dim=1)


# ===================== 损失函数 =====================

def depth_weighted_mse(pred: torch.Tensor, target: torch.Tensor,
                       shallow_thresh: float = 5.0,
                       shallow_weight: float = 2.0) -> torch.Tensor:
    """
    深度分段加权 MSE 损失。
    浅水区（< shallow_thresh）加大权重，提升浅水精度。
    """
    weight = torch.where(target < shallow_thresh, shallow_weight, 1.0)
    return (weight * (pred - target) ** 2).mean()
