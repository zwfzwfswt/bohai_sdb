# -*- coding: utf-8 -*-
"""
多模型对比评估：加载各模型的测试预测 csv，生成对比表与可视化。

用法：
  python -m scripts.eval_compare --pred-dir outputs/predictions --output-dir outputs/plots
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from engine.evaluator import evaluate, evaluate_by_depth


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred-dir", default="outputs/predictions")
    ap.add_argument("--output-dir", default="outputs/plots/compare")
    return ap.parse_args()


def main():
    args = parse_args()
    pred_dir = Path(args.pred_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_files = sorted(pred_dir.glob("*_test.csv"))
    if not csv_files:
        print("[WARN] 未找到预测 csv，请先运行 train.py。")
        return

    # ── 1) 全量指标对比 ──
    rows = []
    model_data = {}
    for f in csv_files:
        name = f.stem.replace("_test", "")
        df = pd.read_csv(f)
        y_true, y_pred = df["y_true"].values, df["y_pred"].values
        m = evaluate(y_true, y_pred)
        m["model"] = name
        rows.append(m)
        model_data[name] = (y_true, y_pred)

    summary = pd.DataFrame(rows)[["model", "rmse", "mae", "r2", "bias", "mre"]]
    summary = summary.sort_values("rmse")
    print("\n========== 模型对比 ==========")
    print(summary.to_string(index=False))
    summary.to_csv(out_dir / "comparison_table.csv", index=False)

    # ── 2) 分段精度 ──
    for name, (yt, yp) in model_data.items():
        seg = evaluate_by_depth(yt, yp)
        seg.to_csv(out_dir / f"{name}_depth_segments.csv", index=False)

    # ── 3) 散点图 ──
    n = len(model_data)
    cols = min(4, n)
    fig_rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(fig_rows, cols, figsize=(5 * cols, 5 * fig_rows),
                             squeeze=False)
    for idx, (name, (yt, yp)) in enumerate(model_data.items()):
        ax = axes[idx // cols][idx % cols]
        ax.scatter(yt, yp, s=2, alpha=0.3)
        lims = [min(yt.min(), yp.min()), max(yt.max(), yp.max())]
        ax.plot(lims, lims, "r--", linewidth=1)
        m = evaluate(yt, yp)
        ax.set_title(f"{name}\nRMSE={m['rmse']:.3f} R²={m['r2']:.3f}", fontsize=10)
        ax.set_xlabel("实测水深 (m)")
        ax.set_ylabel("预测水深 (m)")
    # 隐藏多余子图
    for idx in range(n, fig_rows * cols):
        axes[idx // cols][idx % cols].set_visible(False)
    plt.tight_layout()
    plt.savefig(out_dir / "scatter_compare.png", dpi=200)
    plt.close()
    print(f"\n[DONE] 对比结果已保存: {out_dir}")


if __name__ == "__main__":
    main()
