# -*- coding: utf-8 -*-
"""
像素级 Dataset：为 Stumpf / 机器学习模型提供特征向量。
从 patch npz 中提取中心像素光谱 + 衍生特征。
"""

import numpy as np


# Planet 8 波段名称（SR 产品典型顺序）
BAND_NAMES = [
    "coastal_blue", "blue", "green_i", "green",
    "yellow", "red", "red_edge", "nir",
]


def extract_pixel_features(npz_path: str, split: str = "train",
                           feature_list: list[str] | None = None):
    """
    从 npz 提取中心像素特征。

    参数：
        npz_path:     采样 npz 文件路径
        split:        train / val / test
        feature_list: 特征组合列表，可选 raw_bands / log_bands / ndwi / band_ratios

    返回：
        X: (N, F) float32
        y: (N,)   float32
        names: 特征名列表
    """
    if feature_list is None:
        feature_list = ["raw_bands", "log_bands", "ndwi", "band_ratios"]

    data = np.load(npz_path)
    patches = data[f"X_{split}"].astype(np.float32)   # (N, H, W, C)
    y = data[f"y_{split}"].astype(np.float32)

    # 取中心像素
    H, W = patches.shape[1], patches.shape[2]
    ch, cw = H // 2, W // 2
    pixels = patches[:, ch, cw, :]                     # (N, C)

    features = []
    names = []

    if "raw_bands" in feature_list:
        features.append(pixels)
        names.extend([f"b{i}_{BAND_NAMES[i]}" for i in range(pixels.shape[1])])

    if "log_bands" in feature_list:
        log_p = np.log1p(np.clip(pixels, 0, None))
        features.append(log_p)
        names.extend([f"log_{BAND_NAMES[i]}" for i in range(log_p.shape[1])])

    if "ndwi" in feature_list:
        green = pixels[:, 3]    # green 波段
        nir = pixels[:, 7]      # NIR 波段
        denom = green + nir + 1e-8
        ndwi = ((green - nir) / denom).reshape(-1, 1)
        features.append(ndwi)
        names.append("ndwi")

    if "band_ratios" in feature_list:
        eps = 1e-8
        # Green / Blue
        ratio_gb = (pixels[:, 3] / (pixels[:, 1] + eps)).reshape(-1, 1)
        # Green / Red
        ratio_gr = (pixels[:, 3] / (pixels[:, 5] + eps)).reshape(-1, 1)
        # Blue / Red
        ratio_br = (pixels[:, 1] / (pixels[:, 5] + eps)).reshape(-1, 1)
        # ln(Green) / ln(Blue)  (Stumpf 核心特征)
        ln_green = np.log1p(np.clip(pixels[:, 3], 0, None))
        ln_blue = np.log1p(np.clip(pixels[:, 1], 0, None))
        ratio_stumpf = (ln_green / (ln_blue + eps)).reshape(-1, 1)
        features.append(ratio_gb)
        features.append(ratio_gr)
        features.append(ratio_br)
        features.append(ratio_stumpf)
        names.extend(["ratio_green_blue", "ratio_green_red",
                       "ratio_blue_red", "stumpf_ratio"])

    X = np.concatenate(features, axis=1)
    return X, y, names
