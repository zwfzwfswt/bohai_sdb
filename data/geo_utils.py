# -*- coding: utf-8 -*-
"""
地理工具：影像读取、坐标转换、patch 提取。
迁移自 bathymetry_keras/utils/geo.py，适配任意波段数。
"""

import numpy as np
import rasterio
from rasterio.windows import Window
import geopandas as gpd


def open_raster(tif_path: str, expected_bands: int | None = 8):
    """打开 GeoTIFF，可选校验波段数。"""
    ds = rasterio.open(tif_path)
    if expected_bands is not None and ds.count != expected_bands:
        raise ValueError(f"期望 {expected_bands} 波段，实际 {ds.count}：{tif_path}")
    if ds.crs is None:
        raise ValueError(f"影像缺少 CRS：{tif_path}")
    return ds


def reproject_gdf_to(gdf: gpd.GeoDataFrame, dst_crs) -> gpd.GeoDataFrame:
    """将矢量重投影到目标 CRS。"""
    if gdf.crs is None:
        raise ValueError("Shapefile 未定义坐标系（.prj 丢失？）")
    if str(gdf.crs) == str(dst_crs):
        return gdf
    return gdf.to_crs(dst_crs)


def world2pixel(ds, x, y):
    """投影坐标 (x, y) → 行列索引 (row, col)。"""
    row, col = ds.index(x, y)
    return int(row), int(col)


def read_patch(ds, row: int, col: int, patch_size: int,
               band_indexes: list[int] | None = None):
    """
    读取以 (row, col) 为中心的 patch。
    返回 (C, H, W) float32；越界部分用 NaN 填充。
    """
    rad = patch_size // 2
    r0, c0 = row - rad, col - rad
    r1, c1 = r0 + patch_size, c0 + patch_size

    # 裁剪到影像范围
    r0c, c0c = max(0, r0), max(0, c0)
    r1c, c1c = min(ds.height, r1), min(ds.width, c1)
    h, w = r1c - r0c, c1c - c0c
    if h <= 0 or w <= 0:
        return None

    win = Window.from_slices((r0c, r1c), (c0c, c1c))
    if band_indexes is not None:
        patch = ds.read(band_indexes, window=win).astype(np.float32)
    else:
        patch = ds.read(window=win).astype(np.float32)

    # 如果越界，pad 到目标尺寸
    C = patch.shape[0]
    if h != patch_size or w != patch_size:
        out = np.full((C, patch_size, patch_size), np.nan, dtype=np.float32)
        rs, cs = r0c - r0, c0c - c0
        out[:, rs:rs + h, cs:cs + w] = patch
        patch = out

    # NoData → NaN
    nodatas = ds.nodatavals
    if nodatas and any(v is not None for v in nodatas):
        nod = nodatas[0]
        if nod is not None:
            patch = np.where(patch == nod, np.nan, patch)

    return patch
