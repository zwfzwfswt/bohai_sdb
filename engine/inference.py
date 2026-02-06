# -*- coding: utf-8 -*-
"""
全图推理：滑动窗口遍历 Planet 影像 → 生成水深 GeoTIFF。
"""

import numpy as np
import rasterio
from rasterio.transform import from_bounds
import torch
from tqdm import tqdm


@torch.no_grad()
def sliding_window_inference(
    model: torch.nn.Module,
    raster_path: str,
    output_path: str,
    patch_size: int = 64,
    stride: int | None = None,
    band_mean: np.ndarray | None = None,
    band_std: np.ndarray | None = None,
    batch_size: int = 64,
    device: str = "cuda",
) -> str:
    """
    对完整 Planet 影像做滑动窗口推理。

    参数：
        model:       训练好的模型（输出 (B,1,H,W)）
        raster_path: 输入影像路径
        output_path: 输出水深 GeoTIFF 路径
        patch_size:  窗口大小
        stride:      步长（默认 = patch_size // 2，即 50% 重叠）
        band_mean:   (1,1,C) 归一化均值
        band_std:    (1,1,C) 归一化标准差
        batch_size:  推理批量
        device:      cuda / cpu

    返回：
        output_path
    """
    if stride is None:
        stride = patch_size // 2

    dev = torch.device(device if torch.cuda.is_available() else "cpu")
    model.to(dev).eval()

    # 读取影像元数据
    with rasterio.open(raster_path) as src:
        img = src.read().astype(np.float32)       # (C, H, W)
        profile = src.profile.copy()
        C, H, W = img.shape

    # 归一化 (C,H,W) → 先转 (H,W,C) 归一化再转回
    if band_mean is not None and band_std is not None:
        img_hwc = np.transpose(img, (1, 2, 0))    # (H,W,C)
        img_hwc = (img_hwc - band_mean) / band_std
        img = np.transpose(img_hwc, (2, 0, 1))    # (C,H,W)

    # NaN → 0
    img = np.nan_to_num(img, nan=0.0, posinf=0.0, neginf=0.0)

    # 输出累加矩阵
    depth_sum = np.zeros((H, W), dtype=np.float64)
    count_map = np.zeros((H, W), dtype=np.float64)

    # 收集窗口坐标
    coords = []
    for r in range(0, H - patch_size + 1, stride):
        for c in range(0, W - patch_size + 1, stride):
            coords.append((r, c))
    # 边缘补充
    if (H - patch_size) % stride != 0:
        for c in range(0, W - patch_size + 1, stride):
            coords.append((H - patch_size, c))
    if (W - patch_size) % stride != 0:
        for r in range(0, H - patch_size + 1, stride):
            coords.append((r, W - patch_size))
    if (H - patch_size) % stride != 0 and (W - patch_size) % stride != 0:
        coords.append((H - patch_size, W - patch_size))
    coords = list(set(coords))

    # 批量推理
    for i in tqdm(range(0, len(coords), batch_size), desc="Inference"):
        batch_coords = coords[i:i + batch_size]
        patches = []
        for r, c in batch_coords:
            p = img[:, r:r + patch_size, c:c + patch_size]
            patches.append(p)
        batch_tensor = torch.from_numpy(np.stack(patches)).to(dev)  # (B,C,H,W)

        output = model(batch_tensor)                # (B, 1, ps, ps)
        preds = output[:, 0].cpu().numpy()          # (B, ps, ps)

        for idx, (r, c) in enumerate(batch_coords):
            depth_sum[r:r + patch_size, c:c + patch_size] += preds[idx]
            count_map[r:r + patch_size, c:c + patch_size] += 1.0

    # 取平均（重叠区域）
    mask = count_map > 0
    depth_map = np.full((H, W), np.nan, dtype=np.float32)
    depth_map[mask] = (depth_sum[mask] / count_map[mask]).astype(np.float32)

    # 写 GeoTIFF
    profile.update(dtype="float32", count=1, nodata=np.nan, compress="lzw")
    with rasterio.open(output_path, "w", **profile) as dst:
        dst.write(depth_map, 1)

    print(f"[Inference] 水深图已保存: {output_path}", flush=True)
    return output_path
