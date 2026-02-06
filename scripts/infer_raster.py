# -*- coding: utf-8 -*-
"""
全图推理：用训练好的模型对 Planet 影像进行水深推理，输出 GeoTIFF。

用法：
  python -m scripts.infer_raster \
      --model swin_bathyunet_pp \
      --checkpoint outputs/models/swin_bathyunet_pp/best.pt \
      --tif data/planet.tif \
      --scaler outputs/scaler/band_mean_std.npy \
      --output outputs/predictions/depth_map.tif \
      --config configs/default.yaml
"""

import sys
import argparse
from pathlib import Path

import yaml
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from models.unet_baseline import UNetBaseline
from models.swin_bathyunet import SwinBathyUNet
from models.swin_bathyunet_pp import SwinBathyUNetPP
from engine.inference import sliding_window_inference


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--model", required=True,
                    help="unet | swin_bathyunet | swin_bathyunet_pp")
    ap.add_argument("--checkpoint", required=True, help="best.pt 路径")
    ap.add_argument("--tif", required=True, help="Planet 影像路径")
    ap.add_argument("--scaler", required=True, help="band_mean_std.npy 路径")
    ap.add_argument("--output", default="outputs/predictions/depth_map.tif")
    ap.add_argument("--patch-size", type=int, default=64)
    ap.add_argument("--stride", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=64)
    return ap.parse_args()


def build_model(name: str, cfg: dict) -> torch.nn.Module:
    mc = cfg["model"]
    kw = dict(in_channels=mc["in_channels"], base_channels=mc["base_channels"],
              dropout=mc["dropout"])
    if name == "unet":
        return UNetBaseline(**kw)
    swin_kw = {**kw, "swin_depth": mc["swin_depth"],
               "swin_heads": mc["swin_heads"],
               "swin_window_size": mc["swin_window_size"]}
    if name == "swin_bathyunet":
        return SwinBathyUNet(**swin_kw)
    if name == "swin_bathyunet_pp":
        return SwinBathyUNetPP(**swin_kw)
    raise ValueError(f"未知模型: {name}")


def main():
    args = parse_args()
    cfg = yaml.safe_load(open(args.config))

    # 构建模型 & 加载权重
    model = build_model(args.model, cfg)
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    print(f"[INFO] 已加载权重: {args.checkpoint} (epoch={ckpt['epoch']})")

    # 归一化参数
    scaler = np.load(args.scaler)
    band_mean = scaler[0]   # (1, 1, C)
    band_std = scaler[1]    # (1, 1, C)

    # 推理
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    sliding_window_inference(
        model=model,
        raster_path=args.tif,
        output_path=args.output,
        patch_size=args.patch_size,
        stride=args.stride,
        band_mean=band_mean,
        band_std=band_std,
        batch_size=args.batch_size,
        device=cfg.get("device", "cuda"),
    )


if __name__ == "__main__":
    main()
