# -*- coding: utf-8 -*-
"""
PyTorch Dataset：从 .npz 加载 patch 样本，用于深度学习模型训练。
支持数据增强（翻转、旋转、高斯噪声、亮度微扰）。
"""

import numpy as np
import torch
from torch.utils.data import Dataset


class PatchDataset(Dataset):
    """
    从 npz 文件加载 patch 数据。
    npz 需包含：X_{split} (N, H, W, C), y_{split} (N,)
    返回 (patch: Tensor[C,H,W], depth: Tensor[1])
    """

    def __init__(self, npz_path: str, split: str = "train",
                 mean: np.ndarray | None = None,
                 std: np.ndarray | None = None,
                 augmentation: bool = False):
        data = np.load(npz_path)
        self.X = data[f"X_{split}"].astype(np.float32)   # (N, H, W, C)
        self.y = data[f"y_{split}"].astype(np.float32)    # (N,)
        self.patch_size = int(data["patch_size"])
        self.in_ch = int(data["in_ch"])
        self.augmentation = augmentation and (split == "train")

        # 归一化（per-band）
        if mean is not None and std is not None:
            self.mean = mean.astype(np.float32)
            self.std = std.astype(np.float32)
        else:
            self.mean = None
            self.std = None

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        patch = self.X[idx].copy()            # (H, W, C)
        depth = self.y[idx]

        # 归一化
        if self.mean is not None:
            patch = (patch - self.mean) / self.std

        # 数据增强
        if self.augmentation:
            patch = self._augment(patch)

        # (H, W, C) → (C, H, W)
        patch = np.transpose(patch, (2, 0, 1))
        return (
            torch.from_numpy(patch),
            torch.tensor([depth], dtype=torch.float32),
        )

    def _augment(self, patch: np.ndarray) -> np.ndarray:
        """随机增强：翻转、旋转、噪声、亮度微扰。"""
        # 随机水平翻转
        if np.random.rand() > 0.5:
            patch = np.flip(patch, axis=1).copy()
        # 随机垂直翻转
        if np.random.rand() > 0.5:
            patch = np.flip(patch, axis=0).copy()
        # 随机 90° 旋转
        k = np.random.randint(0, 4)
        if k > 0:
            patch = np.rot90(patch, k, axes=(0, 1)).copy()
        # 高斯噪声 (σ ∈ [0.01, 0.03])
        if np.random.rand() > 0.5:
            sigma = np.random.uniform(0.01, 0.03)
            noise = np.random.normal(0, sigma, patch.shape).astype(np.float32)
            patch = patch + noise
        # 亮度微扰 (±5%)
        if np.random.rand() > 0.5:
            factor = np.random.uniform(0.95, 1.05)
            patch = patch * factor
        return patch


def compute_band_stats(npz_path: str) -> tuple[np.ndarray, np.ndarray]:
    """从训练集计算 per-band mean / std，返回形状 (1,1,C)。"""
    data = np.load(npz_path)
    X_train = data["X_train"]                        # (N, H, W, C)
    mean = X_train.mean(axis=(0, 1, 2), keepdims=True).astype(np.float32)
    std = X_train.std(axis=(0, 1, 2), keepdims=True).astype(np.float32) + 1e-6
    return mean[0], std[0]                            # (1, 1, C) keepdims 去 batch 维
