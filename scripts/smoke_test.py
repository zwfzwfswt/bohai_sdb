# -*- coding: utf-8 -*-
"""
烟雾测试：验证所有模型的 forward 通路和 shape 正确性。
不依赖真实数据，使用随机张量。

用法：
  python -m scripts.smoke_test
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from models.unet_baseline import UNetBaseline
from models.swin_bathyunet import SwinBathyUNet
from models.swin_bathyunet_pp import SwinBathyUNetPP
from models.stumpf import StumpfModel, MultiLinearModel
from models.ml_models import RFModel
from data.pixel_dataset import extract_pixel_features
from engine.evaluator import evaluate


def test_dl_models():
    """测试深度学习模型 forward pass。"""
    B, C, H, W = 4, 8, 64, 64
    x = torch.randn(B, C, H, W)

    models = {
        "UNetBaseline": UNetBaseline(in_channels=C, base_channels=32, dropout=0.1),
        "SwinBathyUNet": SwinBathyUNet(
            in_channels=C, base_channels=32,
            swin_depth=1, swin_heads=4, swin_window_size=8, dropout=0.1),
        "SwinBathyUNetPP": SwinBathyUNetPP(
            in_channels=C, base_channels=32,
            swin_depth=1, swin_heads=4, swin_window_size=8, dropout=0.1),
    }

    all_ok = True
    for name, model in models.items():
        model.eval()
        try:
            with torch.no_grad():
                out = model(x)
            expected = (B, 1, H, W)
            if out.shape != expected:
                print(f"  [FAIL] {name}: 输出 {out.shape}, 期望 {expected}")
                all_ok = False
            else:
                n_params = sum(p.numel() for p in model.parameters())
                print(f"  [OK]   {name}: {out.shape}  参数量={n_params:,}")
        except Exception as e:
            print(f"  [FAIL] {name}: {e}")
            all_ok = False

    return all_ok


def test_center_pixel_extraction():
    """测试中心像素提取（训练时的 loss 计算方式）。"""
    B, C, H, W = 4, 8, 64, 64
    x = torch.randn(B, C, H, W)
    model = UNetBaseline(in_channels=C, base_channels=32)
    model.eval()

    with torch.no_grad():
        out = model(x)  # (B, 1, H, W)
    h, w = out.shape[2], out.shape[3]
    center = out[:, 0, h // 2, w // 2]  # (B,)

    assert center.shape == (B,), f"中心像素 shape 错误: {center.shape}"
    print(f"  [OK]   中心像素提取: {center.shape}")
    return True


def test_ml_models():
    """测试 ML/经典模型。"""
    np.random.seed(42)
    N = 100
    X = np.random.rand(N, 8).astype(np.float32)
    y = np.random.rand(N).astype(np.float32) * 10

    all_ok = True

    # Stumpf
    m = StumpfModel()
    m.fit(X, y)
    pred = m.predict(X)
    assert pred.shape == (N,), f"Stumpf 输出 shape 错误: {pred.shape}"
    print(f"  [OK]   StumpfModel: pred.shape={pred.shape}")

    # MultiLinear
    m2 = MultiLinearModel()
    m2.fit(X, y)
    pred2 = m2.predict(X)
    assert pred2.shape == (N,), f"MultiLinear 输出 shape 错误: {pred2.shape}"
    print(f"  [OK]   MultiLinearModel: pred.shape={pred2.shape}")

    # RF
    m3 = RFModel(n_estimators=10, max_depth=5)
    m3.fit(X, y)
    pred3 = m3.predict(X)
    assert pred3.shape == (N,), f"RF 输出 shape 错误: {pred3.shape}"
    print(f"  [OK]   RFModel: pred.shape={pred3.shape}")

    return all_ok


def test_evaluator():
    """测试评估模块。"""
    y_true = np.array([1.0, 2.0, 3.0, 5.0, 10.0])
    y_pred = np.array([1.1, 2.2, 2.8, 5.5, 9.5])
    metrics = evaluate(y_true, y_pred)
    assert "rmse" in metrics and "r2" in metrics
    print(f"  [OK]   评估: RMSE={metrics['rmse']:.4f} R²={metrics['r2']:.4f}")
    return True


def main():
    print("=" * 50)
    print(" Bohai SDB 烟雾测试")
    print("=" * 50)

    sections = [
        ("深度学习模型 Forward Pass", test_dl_models),
        ("中心像素提取", test_center_pixel_extraction),
        ("ML/经典模型", test_ml_models),
        ("评估模块", test_evaluator),
    ]

    all_pass = True
    for name, fn in sections:
        print(f"\n[{name}]")
        try:
            ok = fn()
            if not ok:
                all_pass = False
        except Exception as e:
            print(f"  [FAIL] 异常: {e}")
            import traceback
            traceback.print_exc()
            all_pass = False

    print(f"\n{'=' * 50}")
    if all_pass:
        print(" ✓ 全部通过！")
    else:
        print(" ✗ 存在失败项，请检查。")
        sys.exit(1)


if __name__ == "__main__":
    main()
