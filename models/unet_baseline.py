# -*- coding: utf-8 -*-
"""
PyTorch U-Net 基线模型（回归版）。
参考现有 UNet/unet.py (Keras)，改为 PyTorch 实现。
输入 (B, C, H, W)  → 输出 (B, 1, H, W)。
训练时取中心像素作为点回归目标。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """Conv3x3 - BN - ReLU × 2"""

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
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class UNetBaseline(nn.Module):
    """
    标准 4 层 U-Net（回归版本）。
    编码器：[64, 128, 256, 512]
    瓶颈：1024
    解码器对称。
    """

    def __init__(self, in_channels: int = 8, base_channels: int = 64,
                 dropout: float = 0.1):
        super().__init__()
        bc = base_channels   # 64

        # Encoder
        self.enc1 = ConvBlock(in_channels, bc)
        self.enc2 = ConvBlock(bc, bc * 2)
        self.enc3 = ConvBlock(bc * 2, bc * 4)
        self.enc4 = ConvBlock(bc * 4, bc * 8, dropout=dropout)

        # Bottleneck
        self.bottleneck = ConvBlock(bc * 8, bc * 16, dropout=dropout)

        # Decoder
        self.up4 = nn.ConvTranspose2d(bc * 16, bc * 8, 2, stride=2)
        self.dec4 = ConvBlock(bc * 16, bc * 8)
        self.up3 = nn.ConvTranspose2d(bc * 8, bc * 4, 2, stride=2)
        self.dec3 = ConvBlock(bc * 8, bc * 4)
        self.up2 = nn.ConvTranspose2d(bc * 4, bc * 2, 2, stride=2)
        self.dec2 = ConvBlock(bc * 4, bc * 2)
        self.up1 = nn.ConvTranspose2d(bc * 2, bc, 2, stride=2)
        self.dec1 = ConvBlock(bc * 2, bc)

        # Head
        self.head = nn.Conv2d(bc, 1, kernel_size=1)

    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(F.max_pool2d(e1, 2))
        e3 = self.enc3(F.max_pool2d(e2, 2))
        e4 = self.enc4(F.max_pool2d(e3, 2))

        # Bottleneck
        b = self.bottleneck(F.max_pool2d(e4, 2))

        # Decoder + skip connections
        d4 = self.up4(b)
        d4 = self._pad_cat(d4, e4)
        d4 = self.dec4(d4)

        d3 = self.up3(d4)
        d3 = self._pad_cat(d3, e3)
        d3 = self.dec3(d3)

        d2 = self.up2(d3)
        d2 = self._pad_cat(d2, e2)
        d2 = self.dec2(d2)

        d1 = self.up1(d2)
        d1 = self._pad_cat(d1, e1)
        d1 = self.dec1(d1)

        return self.head(d1)                   # (B, 1, H, W)

    @staticmethod
    def _pad_cat(up: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        """处理上采样与 skip 之间的尺寸差异。"""
        dh = skip.size(2) - up.size(2)
        dw = skip.size(3) - up.size(3)
        up = F.pad(up, [dw // 2, dw - dw // 2, dh // 2, dh - dh // 2])
        return torch.cat([up, skip], dim=1)


def get_center_pred(output: torch.Tensor) -> torch.Tensor:
    """从 (B,1,H,W) 提取中心像素 → (B,1)。"""
    h, w = output.shape[2], output.shape[3]
    return output[:, :, h // 2, w // 2]
