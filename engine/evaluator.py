# -*- coding: utf-8 -*-
"""
评估模块：统一指标计算 + 深度分段分析。
"""

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score


def evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """
    计算全量评估指标。
    返回 dict: rmse, mae, r2, bias, mre
    """
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    diff = y_pred - y_true
    rmse = float(np.sqrt(np.mean(diff ** 2)))
    mae = float(np.mean(np.abs(diff)))
    r2 = float(r2_score(y_true, y_pred))
    bias = float(np.mean(diff))
    mre = float(np.mean(np.abs(diff) / (np.abs(y_true) + 1e-8)) * 100)
    return {"rmse": rmse, "mae": mae, "r2": r2, "bias": bias, "mre": mre}


def evaluate_by_depth(y_true: np.ndarray, y_pred: np.ndarray,
                      bins: list[float] | None = None) -> pd.DataFrame:
    """
    按深度区间分段评估。
    默认区间：0-2m, 2-5m, 5-10m, >10m
    """
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    if bins is None:
        bins = [0, 2, 5, 10, float("inf")]
    labels = [f"{bins[i]:.0f}-{bins[i+1]:.0f}m" for i in range(len(bins) - 1)]
    labels[-1] = f">{bins[-2]:.0f}m"

    rows = []
    for i in range(len(bins) - 1):
        mask = (y_true >= bins[i]) & (y_true < bins[i + 1])
        n = int(mask.sum())
        if n == 0:
            rows.append({"range": labels[i], "n": 0,
                         "rmse": np.nan, "mae": np.nan,
                         "r2": np.nan, "bias": np.nan})
            continue
        metrics = evaluate(y_true[mask], y_pred[mask])
        metrics["range"] = labels[i]
        metrics["n"] = n
        rows.append(metrics)

    return pd.DataFrame(rows)[["range", "n", "rmse", "mae", "r2", "bias"]]


def format_summary(metrics: dict, model_name: str = "") -> str:
    """将指标字典格式化为可读字符串。"""
    parts = [f"[{model_name}]"] if model_name else []
    parts.append(f"RMSE={metrics['rmse']:.4f}")
    parts.append(f"MAE={metrics['mae']:.4f}")
    parts.append(f"R²={metrics['r2']:.4f}")
    parts.append(f"Bias={metrics['bias']:.4f}")
    parts.append(f"MRE={metrics['mre']:.2f}%")
    return "  ".join(parts)
