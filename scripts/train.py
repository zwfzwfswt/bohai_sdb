# -*- coding: utf-8 -*-
"""
统一训练入口：支持所有模型类型（Stumpf / ML / DL）。

用法：
  # 深度学习模型
  python -m scripts.train --config configs/default.yaml --model swin_bathyunet_pp \
      --samples outputs/samples/samples_ps64.npz

  # 机器学习模型
  python -m scripts.train --config configs/default.yaml --model rf \
      --samples outputs/samples/samples_ps64.npz
"""

import sys
import argparse
import pickle
from pathlib import Path

import yaml
import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from data.patch_dataset import PatchDataset, compute_band_stats
from data.pixel_dataset import extract_pixel_features
from models.stumpf import StumpfModel, MultiLinearModel
from models.ml_models import build_ml_model
from models.unet_baseline import UNetBaseline
from models.swin_bathyunet import SwinBathyUNet
from models.swin_bathyunet_pp import SwinBathyUNetPP
from engine.trainer import Trainer
from engine.evaluator import evaluate, evaluate_by_depth, format_summary

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


DL_MODELS = {"unet", "swin_bathyunet", "swin_bathyunet_pp"}
ML_MODELS = {"rf", "xgboost", "lgbm"}
CLASSIC_MODELS = {"stumpf", "stumpf_multi"}


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--model", required=True,
                    help="stumpf|stumpf_multi|rf|xgboost|lgbm|unet|swin_bathyunet|swin_bathyunet_pp")
    ap.add_argument("--samples", required=True, help="采样 npz 路径")
    return ap.parse_args()


def seed_everything(seed: int):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def build_dl_model(name: str, cfg: dict) -> torch.nn.Module:
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
    raise ValueError(f"未知 DL 模型: {name}")


def train_dl(cfg: dict, model_name: str, npz_path: str):
    """训练深度学习模型。"""
    # 归一化参数
    mean, std = compute_band_stats(npz_path)
    np.save(Path(cfg["output"]["scaler_npy"]), np.stack([mean, std]))

    aug = cfg["data"].get("augmentation", True)
    train_ds = PatchDataset(npz_path, "train", mean, std, augmentation=aug)
    val_ds = PatchDataset(npz_path, "val", mean, std, augmentation=False)
    test_ds = PatchDataset(npz_path, "test", mean, std, augmentation=False)

    bs = cfg["train"]["batch_size"]
    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True,
                              num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=bs * 2, shuffle=False,
                            num_workers=4, pin_memory=True)

    model = build_dl_model(model_name, cfg)
    save_dir = Path(cfg["output"]["model_dir"]) / model_name
    trainer = Trainer(model, train_loader, val_loader, cfg, save_dir=str(save_dir))

    print(f"\n{'='*60}")
    print(f" 训练模型: {model_name}")
    print(f" 参数量: {sum(p.numel() for p in model.parameters()):,}")
    print(f"{'='*60}\n")

    history = trainer.train()

    # 测试评估
    test_loader = DataLoader(test_ds, batch_size=bs * 2, shuffle=False,
                             num_workers=4, pin_memory=True)
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for patches, depths in test_loader:
            patches = patches.to(trainer.device)
            output = model(patches)
            h, w = output.shape[2], output.shape[3]
            pred = output[:, 0, h // 2, w // 2].cpu().numpy()
            preds.append(pred)
            trues.append(depths[:, 0].numpy())

    y_pred = np.concatenate(preds)
    y_true = np.concatenate(trues)
    metrics = evaluate(y_true, y_pred)
    print(f"\n[TEST] {format_summary(metrics, model_name)}")
    print(evaluate_by_depth(y_true, y_pred).to_string(index=False))

    # 保存曲线
    plots_dir = Path(cfg["output"]["plots_dir"]) / model_name
    plots_dir.mkdir(parents=True, exist_ok=True)
    _save_curves(history, plots_dir)

    # 保存预测
    import pandas as pd
    pred_dir = Path(cfg["output"]["pred_csv"]).parent
    pred_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"y_true": y_true, "y_pred": y_pred}).to_csv(
        pred_dir / f"{model_name}_test.csv", index=False)

    return metrics


def _save_pred_csv(cfg: dict, model_name: str,
                   y_true: np.ndarray, y_pred: np.ndarray):
    """保存预测结果到 CSV。"""
    import pandas as pd
    pred_dir = Path(cfg["output"]["pred_csv"]).parent
    pred_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"y_true": y_true, "y_pred": y_pred}).to_csv(
        pred_dir / f"{model_name}_test.csv", index=False)


def train_ml(cfg: dict, model_name: str, npz_path: str):
    """训练机器学习 / 经典模型。"""
    feat_list = cfg.get("ml", {}).get("features", ["raw_bands"])
    X_tr, y_tr, names = extract_pixel_features(npz_path, "train", feat_list)
    X_val, y_val, _ = extract_pixel_features(npz_path, "val", feat_list)
    X_te, y_te, _ = extract_pixel_features(npz_path, "test", feat_list)

    print(f"\n[{model_name}] 特征维度: {X_tr.shape[1]}  特征: {names[:5]}...")

    if model_name == "stumpf":
        # Stumpf 只用 raw_bands 的波段索引
        X_tr_raw, y_tr, _ = extract_pixel_features(npz_path, "train", ["raw_bands"])
        X_te_raw, y_te, _ = extract_pixel_features(npz_path, "test", ["raw_bands"])
        model = StumpfModel()
        model.fit(X_tr_raw, y_tr)
        y_pred = model.predict(X_te_raw)
        metrics = evaluate(y_te, y_pred)
        print(f"[TEST] {format_summary(metrics, model_name)}")
        print(evaluate_by_depth(y_te, y_pred).to_string(index=False))
        _save_pred_csv(cfg, model_name, y_te, y_pred)
        return metrics
    elif model_name == "stumpf_multi":
        # 多波段回归只需 raw bands
        X_tr_raw, y_tr, _ = extract_pixel_features(npz_path, "train", ["raw_bands"])
        X_te_raw, y_te, _ = extract_pixel_features(npz_path, "test", ["raw_bands"])
        model = MultiLinearModel()
        model.fit(X_tr_raw, y_tr)
        y_pred = model.predict(X_te_raw)
        metrics = evaluate(y_te, y_pred)
        print(f"[TEST] {format_summary(metrics, model_name)}")
        print(evaluate_by_depth(y_te, y_pred).to_string(index=False))
        _save_pred_csv(cfg, model_name, y_te, y_pred)
        return metrics
    else:
        ml_cfg = cfg.get("ml", {})
        model = build_ml_model(model_name, ml_cfg)

    model.fit(X_tr, y_tr)
    y_pred = model.predict(X_te)
    metrics = evaluate(y_te, y_pred)
    print(f"\n[TEST] {format_summary(metrics, model_name)}")
    print(evaluate_by_depth(y_te, y_pred).to_string(index=False))

    # 保存模型
    save_dir = Path(cfg["output"]["model_dir"]) / model_name
    save_dir.mkdir(parents=True, exist_ok=True)
    with open(save_dir / "model.pkl", "wb") as f:
        pickle.dump(model, f)

    # 保存预测
    import pandas as pd
    pred_dir = Path(cfg["output"]["pred_csv"]).parent
    pred_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"y_true": y_te, "y_pred": y_pred}).to_csv(
        pred_dir / f"{model_name}_test.csv", index=False)

    return metrics


def _save_curves(history: dict, plots_dir: Path):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].plot(history["train_loss"], label="train_loss")
    axes[0].set_xlabel("epoch"); axes[0].set_ylabel("MSE"); axes[0].legend()
    axes[1].plot(history["val_rmse"], label="val_RMSE")
    axes[1].set_xlabel("epoch"); axes[1].set_ylabel("RMSE"); axes[1].legend()
    axes[2].plot(history["val_r2"], label="val_R²")
    axes[2].set_xlabel("epoch"); axes[2].set_ylabel("R²"); axes[2].legend()
    plt.tight_layout()
    plt.savefig(plots_dir / "training_curves.png", dpi=150)
    plt.close()


def main():
    args = parse_args()
    cfg = yaml.safe_load(open(args.config))
    seed_everything(cfg.get("seed", 42))

    model_name = args.model
    if model_name in DL_MODELS:
        train_dl(cfg, model_name, args.samples)
    elif model_name in ML_MODELS or model_name in CLASSIC_MODELS:
        train_ml(cfg, model_name, args.samples)
    else:
        print(f"[ERROR] 未知模型: {model_name}")
        sys.exit(1)


if __name__ == "__main__":
    main()
