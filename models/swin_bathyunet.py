# -*- coding: utf-8 -*-
"""
Swin-BathyUNet（参考复现版）。
改编自 Swin-BathyUNet-main/bathymetry/swin-bathyunet.py，
适配 8 波段输入和可配置窗口大小。

原作者：Panagiotis Agrafiotis
论文：https://doi.org/10.1016/j.isprsjprs.2025.04.020
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


# ===================== 辅助函数 =====================

def window_partition(x: torch.Tensor, window_size: int) -> torch.Tensor:
    """(B, H, W, C) → (num_windows*B, ws, ws, C)"""
    B, H, W, C = x.shape
    pad_h = (window_size - H % window_size) % window_size
    pad_w = (window_size - W % window_size) % window_size
    if pad_h > 0 or pad_w > 0:
        x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h))
    _, Hp, Wp, _ = x.shape
    x = x.view(B, Hp // window_size, window_size,
               Wp // window_size, window_size, C)
    return x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)


def window_reverse(windows: torch.Tensor, window_size: int,
                   H: int, W: int) -> torch.Tensor:
    """(num_windows*B, ws, ws, C) → (B, H, W, C)"""
    pad_h = (window_size - H % window_size) % window_size
    pad_w = (window_size - W % window_size) % window_size
    Hp, Wp = H + pad_h, W + pad_w
    B = int(windows.shape[0] / (Hp // window_size * Wp // window_size))
    x = windows.view(B, Hp // window_size, Wp // window_size,
                     window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, Hp, Wp, -1)
    return x[:, :H, :W, :].contiguous()


# ===================== 注意力模块 =====================

class WindowAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.num_heads = num_heads
        self.scale = nn.Parameter(torch.ones(1) * (dim // num_heads) ** -0.5)
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.proj = nn.Linear(dim, dim)
        self.attn_drop = nn.Dropout(dropout)
        self.proj_drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        return self.proj_drop(self.proj(x))


class CrossAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.num_heads = num_heads
        self.scale = nn.Parameter(torch.ones(1) * (dim // num_heads) ** -0.5)
        self.q_proj = nn.Linear(dim, dim)
        self.kv_proj = nn.Linear(dim, dim * 2)
        self.out_proj = nn.Linear(dim, dim)
        self.attn_drop = nn.Dropout(dropout)

    def forward(self, query, context):
        B, N, C = query.shape
        q = self.q_proj(query).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        kv = self.kv_proj(context).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv.unbind(0)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = self.attn_drop(attn.softmax(dim=-1))
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        return self.out_proj(x)


# ===================== Swin Transformer Block =====================

class SwinTransformerBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, window_size: int = 8,
                 shift_size: int = 0, mlp_ratio: float = 4.0,
                 dropout: float = 0.1):
        super().__init__()
        self.window_size = window_size
        self.shift_size = shift_size
        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(dim, num_heads, dropout)
        self.cross_attn = CrossAttention(dim, num_heads, dropout)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(int(dim * mlp_ratio), dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor, H: int, W: int,
                cross_input: torch.Tensor | None = None) -> torch.Tensor:
        B, L, C = x.shape
        shortcut = x
        x = self.norm1(x).view(B, H, W, C)

        # shift
        if self.shift_size > 0:
            x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))

        # window attention
        x_win = window_partition(x, self.window_size)
        x_win = x_win.view(-1, self.window_size * self.window_size, C)
        attn_win = self.attn(x_win)
        attn_win = attn_win.view(-1, self.window_size, self.window_size, C)
        x = window_reverse(attn_win, self.window_size, H, W)

        if self.shift_size > 0:
            x = torch.roll(x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))

        x = x.view(B, H * W, C)
        x = shortcut + x

        # cross attention
        if cross_input is not None:
            x = x + self.cross_attn(x, cross_input)

        # MLP
        x = x + self.mlp(self.norm2(x))
        return x


# ===================== Vision Transformer =====================

class VisionTransformerStage(nn.Module):
    """对特征图施加 Swin blocks 的阶段。"""

    def __init__(self, in_channels: int, depth: int = 1, num_heads: int = 8,
                 window_size: int = 8, dropout: float = 0.1):
        super().__init__()
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(
                dim=in_channels, num_heads=num_heads,
                window_size=window_size,
                shift_size=0 if (i % 2 == 0) else window_size // 2,
                dropout=dropout,
            )
            for i in range(depth)
        ])
        self.norm = nn.LayerNorm(in_channels)

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        """feat: (B, C, H, W) → (B, C, H, W)"""
        B, C, H, W = feat.shape
        x = rearrange(feat, "b c h w -> b (h w) c")
        for blk in self.blocks:
            x = blk(x, H, W)
        x = self.norm(x)
        return rearrange(x, "b (h w) c -> b c h w", h=H, w=W)


# ===================== Conv Block =====================

class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


# ===================== Swin-BathyUNet =====================

class SwinBathyUNet(nn.Module):
    """
    Swin-BathyUNet：U-Net + Swin Transformer 跳连注意力。
    适配 Planet 8 波段；窗口大小可配置。
    """

    def __init__(self, in_channels: int = 8, base_channels: int = 64,
                 swin_depth: int = 1, swin_heads: int = 8,
                 swin_window_size: int = 8, dropout: float = 0.1):
        super().__init__()
        bc = base_channels

        # Encoder
        self.enc1 = ConvBlock(in_channels, bc)
        self.enc2 = ConvBlock(bc, bc * 2)
        self.enc3 = ConvBlock(bc * 2, bc * 4)
        self.enc4 = ConvBlock(bc * 4, bc * 8)

        # Bottleneck
        self.bottleneck = ConvBlock(bc * 8, bc * 16)

        # Swin stages on encoder skip features
        self.vit4 = VisionTransformerStage(bc * 8, swin_depth, swin_heads,
                                           swin_window_size, dropout)
        self.vit3 = VisionTransformerStage(bc * 4, swin_depth, swin_heads,
                                           swin_window_size, dropout)
        self.vit2 = VisionTransformerStage(bc * 2, swin_depth, swin_heads,
                                           swin_window_size, dropout)

        # Decoder
        self.up4 = nn.ConvTranspose2d(bc * 16, bc * 8, 2, stride=2)
        self.dec4 = ConvBlock(bc * 16, bc * 8)
        self.up3 = nn.ConvTranspose2d(bc * 8, bc * 4, 2, stride=2)
        self.dec3 = ConvBlock(bc * 8, bc * 4)
        self.up2 = nn.ConvTranspose2d(bc * 4, bc * 2, 2, stride=2)
        self.dec2 = ConvBlock(bc * 4, bc * 2)
        self.up1 = nn.ConvTranspose2d(bc * 2, bc, 2, stride=2)
        self.dec1 = ConvBlock(bc * 2, bc)

        self.head = nn.Conv2d(bc, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(F.max_pool2d(e1, 2))
        e3 = self.enc3(F.max_pool2d(e2, 2))
        e4 = self.enc4(F.max_pool2d(e3, 2))

        b = self.bottleneck(F.max_pool2d(e4, 2))

        # Decoder with Swin-enhanced skip connections
        d4 = self.up4(b)
        e4s = self.vit4(e4)
        e4s = F.interpolate(e4s, size=d4.shape[2:], mode="bilinear", align_corners=False)
        d4 = self.dec4(torch.cat([d4, e4s], dim=1))

        d3 = self.up3(d4)
        e3s = self.vit3(e3)
        e3s = F.interpolate(e3s, size=d3.shape[2:], mode="bilinear", align_corners=False)
        d3 = self.dec3(torch.cat([d3, e3s], dim=1))

        d2 = self.up2(d3)
        e2s = self.vit2(e2)
        e2s = F.interpolate(e2s, size=d2.shape[2:], mode="bilinear", align_corners=False)
        d2 = self.dec2(torch.cat([d2, e2s], dim=1))

        d1 = self.up1(d2)
        d1 = F.interpolate(d1, size=e1.shape[2:], mode="bilinear", align_corners=False)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))

        return self.head(d1)
