# -*- coding: utf-8 -*-
"""
从 USV 测深点采样 Planet 影像 patch → 生成 train/val/test npz。
迁移自 bathymetry_keras/datasets/prepare_samples_from_shp.py。

用法：
  python -m scripts.prepare_samples \
      --tif data/planet.tif --shp data/usv.shp \
      --depth-field "改正水" --patch-size 64 --suffix ps64
"""

import sys
import time
import argparse
import traceback
from pathlib import Path

import numpy as np
import geopandas as gpd
import pandas as pd
from sklearn.model_selection import train_test_split

# 允许从项目根导入
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from data.geo_utils import open_raster, reproject_gdf_to, world2pixel, read_patch


def parse_args():
    ap = argparse.ArgumentParser(description="Planet + USV → patch 样本 npz")
    ap.add_argument("--tif", required=True, help="Planet 8 波段影像路径")
    ap.add_argument("--shp", required=True, help="USV 测深 Shapefile 路径")
    ap.add_argument("--depth-field", default="改正水", help="水深字段名")
    ap.add_argument("--patch-size", type=int, default=64)
    ap.add_argument("--suffix", default=None, help="输出后缀（默认 ps{patch_size}）")
    ap.add_argument("--train-ratio", type=float, default=0.7)
    ap.add_argument("--val-ratio", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--valid-min", type=float, default=0.0,
                    help="反射率合法下限（Planet SR 通常 0.0）")
    ap.add_argument("--valid-max", type=float, default=10000.0,
                    help="反射率合法上限（Planet SR 0-1 则设 1.2；DN 产品设 10000）")
    ap.add_argument("--output-dir", default="outputs/samples")
    return ap.parse_args()


def make_stratify_labels(y, max_bins=10, min_count=2):
    y = np.asarray(y).ravel()
    if np.allclose(y.min(), y.max()) or y.size < min_count * 2:
        return None
    bins = max_bins
    while bins >= 2:
        try:
            labels = pd.qcut(y, q=bins, labels=False, duplicates="drop").astype(int)
            counts = pd.Series(labels).value_counts().values
            if counts.min() >= min_count and len(np.unique(labels)) >= 2:
                return labels
        except Exception:
            pass
        bins -= 1
    return None


def main():
    args = parse_args()
    t0 = time.time()
    ps = args.patch_size
    suffix = args.suffix or f"ps{ps}"
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] tif={args.tif}  shp={args.shp}  patch_size={ps}", flush=True)

    # 1) 打开数据
    ds = open_raster(args.tif)
    gdf = gpd.read_file(args.shp)
    if args.depth_field not in gdf.columns:
        print(f"[ERROR] 字段 '{args.depth_field}' 不存在。可用：{list(gdf.columns)}")
        sys.exit(1)

    gdf = reproject_gdf_to(gdf, ds.crs)
    gdf[args.depth_field] = pd.to_numeric(gdf[args.depth_field], errors="coerce")
    gdf = gdf.dropna(subset=[args.depth_field]).copy()

    # 2) 遍历采样
    X_list, y_list = [], []
    for _, row in gdf.iterrows():
        geom = row.geometry
        if geom is None or geom.is_empty:
            continue
        r, c = world2pixel(ds, geom.x, geom.y)
        patch = read_patch(ds, r, c, ps)
        if patch is None:
            continue
        if np.isnan(patch).any():
            continue
        pmin, pmax = np.nanmin(patch), np.nanmax(patch)
        if not (args.valid_min <= pmin and pmax <= args.valid_max):
            continue

        patch = np.moveaxis(patch, 0, -1).astype(np.float32)  # (H,W,C)
        depth = float(row[args.depth_field])
        if np.isnan(depth) or depth <= 0:
            continue
        X_list.append(patch)
        y_list.append(depth)

    if not y_list:
        print("[ERROR] 未采集到有效样本！")
        sys.exit(1)

    X = np.stack(X_list)
    y = np.array(y_list, dtype=np.float32)
    print(f"[INFO] 有效样本: {len(y)}，形状: {X.shape}", flush=True)

    # 3) 划分
    test_ratio = 1.0 - args.train_ratio - args.val_ratio
    strat = make_stratify_labels(y)
    try:
        X_tr, X_tmp, y_tr, y_tmp = train_test_split(
            X, y, test_size=1.0 - args.train_ratio,
            random_state=args.seed, stratify=strat)
    except ValueError:
        X_tr, X_tmp, y_tr, y_tmp = train_test_split(
            X, y, test_size=1.0 - args.train_ratio, random_state=args.seed)

    rel = args.val_ratio / (1.0 - args.train_ratio)
    try:
        X_val, X_te, y_val, y_te = train_test_split(
            X_tmp, y_tmp, test_size=1.0 - rel, random_state=args.seed)
    except ValueError:
        X_val, X_te, y_val, y_te = train_test_split(
            X_tmp, y_tmp, test_size=1.0 - rel, random_state=args.seed)

    # 4) 保存
    out_npz = out_dir / f"samples_{suffix}.npz"
    np.savez_compressed(
        out_npz,
        X_train=X_tr, y_train=y_tr,
        X_val=X_val, y_val=y_val,
        X_test=X_te, y_test=y_te,
        patch_size=ps, in_ch=X.shape[-1],
    )
    print(f"[DONE] train={len(y_tr)} val={len(y_val)} test={len(y_te)}", flush=True)
    print(f"[DONE] 保存: {out_npz}  耗时: {time.time()-t0:.1f}s", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        traceback.print_exc()
        sys.exit(1)
