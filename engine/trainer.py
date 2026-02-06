# -*- coding: utf-8 -*-
"""
统一 PyTorch 训练循环。
支持：早停、学习率调度、checkpoint、混合精度、日志。
"""

import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.amp import GradScaler, autocast

from .evaluator import evaluate


class Trainer:
    """
    通用训练器。

    参数：
        model:       nn.Module（输出 (B,1,H,W)）
        train_loader: DataLoader
        val_loader:   DataLoader
        cfg:          配置 dict（包含 train / loss 子项）
        save_dir:     模型保存路径
    """

    def __init__(self, model: nn.Module, train_loader: DataLoader,
                 val_loader: DataLoader, cfg: dict,
                 save_dir: str = "outputs/models"):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.cfg = cfg
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        self.device = torch.device(cfg.get("device", "cuda")
                                   if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        # 优化器
        train_cfg = cfg["train"]
        if train_cfg.get("optimizer", "AdamW") == "AdamW":
            self.optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=train_cfg["lr"],
                weight_decay=train_cfg.get("weight_decay", 1e-4),
            )
        else:
            self.optimizer = torch.optim.Adam(
                model.parameters(), lr=train_cfg["lr"],
            )

        # 学习率调度
        sched_name = train_cfg.get("scheduler", "cosine_warm_restarts")
        if sched_name == "cosine_warm_restarts":
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                self.optimizer, T_0=train_cfg.get("T_0", 20), T_mult=2,
            )
        elif sched_name == "reduce_on_plateau":
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode="min", factor=0.5,
                patience=max(2, train_cfg.get("patience", 10) // 3),
            )
        else:
            self.scheduler = None

        # 损失函数
        self.loss_cfg = cfg.get("loss", {})
        self.loss_type = self.loss_cfg.get("type", "mse")

        # 混合精度
        self.scaler = GradScaler("cuda") if self.device.type == "cuda" else None

        # 早停
        self.patience = train_cfg.get("patience", 30)
        self.epochs = train_cfg.get("epochs", 200)

    # ---------- 损失 ----------
    def _compute_loss(self, output: torch.Tensor,
                      target: torch.Tensor) -> torch.Tensor:
        """从 (B,1,H,W) 输出中提取中心像素，计算损失。"""
        # 取中心像素
        h, w = output.shape[2], output.shape[3]
        pred = output[:, 0, h // 2, w // 2].unsqueeze(1)   # (B, 1)

        if self.loss_type == "depth_weighted_mse":
            st = self.loss_cfg.get("shallow_threshold", 5.0)
            sw = self.loss_cfg.get("shallow_weight", 2.0)
            weight = torch.where(target < st, sw, 1.0)
            return (weight * (pred - target) ** 2).mean()
        # 默认 MSE
        return nn.functional.mse_loss(pred, target)

    # ---------- 单 epoch ----------
    def _train_one_epoch(self) -> float:
        self.model.train()
        losses = []
        for patches, depths in self.train_loader:
            patches = patches.to(self.device)
            depths = depths.to(self.device)

            self.optimizer.zero_grad()
            if self.scaler is not None:
                with autocast("cuda"):
                    output = self.model(patches)
                    loss = self._compute_loss(output, depths)
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                output = self.model(patches)
                loss = self._compute_loss(output, depths)
                loss.backward()
                self.optimizer.step()

            losses.append(loss.item())
        return float(np.mean(losses))

    @torch.no_grad()
    def _validate(self) -> tuple[float, dict]:
        self.model.eval()
        all_pred, all_true = [], []
        for patches, depths in self.val_loader:
            patches = patches.to(self.device)
            output = self.model(patches)
            h, w = output.shape[2], output.shape[3]
            pred = output[:, 0, h // 2, w // 2].cpu().numpy()
            all_pred.append(pred)
            all_true.append(depths[:, 0].numpy())
        y_pred = np.concatenate(all_pred)
        y_true = np.concatenate(all_true)
        metrics = evaluate(y_true, y_pred)
        return metrics["rmse"], metrics

    # ---------- 完整训练 ----------
    def train(self) -> dict:
        """
        执行完整训练流程。
        返回 history dict: {train_loss, val_rmse, val_mae, val_r2, lr}
        """
        history = {"train_loss": [], "val_rmse": [], "val_mae": [],
                    "val_r2": [], "lr": []}
        best_rmse = float("inf")
        no_improve = 0

        for epoch in range(1, self.epochs + 1):
            t0 = time.time()
            train_loss = self._train_one_epoch()
            val_rmse, val_metrics = self._validate()
            elapsed = time.time() - t0

            # 学习率调度
            lr = self.optimizer.param_groups[0]["lr"]
            if isinstance(self.scheduler,
                          torch.optim.lr_scheduler.ReduceLROnPlateau):
                self.scheduler.step(val_rmse)
            elif self.scheduler is not None:
                self.scheduler.step()

            # 记录
            history["train_loss"].append(train_loss)
            history["val_rmse"].append(val_rmse)
            history["val_mae"].append(val_metrics["mae"])
            history["val_r2"].append(val_metrics["r2"])
            history["lr"].append(lr)

            print(f"Epoch {epoch:03d}/{self.epochs} | "
                  f"loss={train_loss:.4f} | "
                  f"val_RMSE={val_rmse:.4f} | "
                  f"val_R²={val_metrics['r2']:.4f} | "
                  f"lr={lr:.2e} | {elapsed:.1f}s", flush=True)

            # checkpoint
            if val_rmse < best_rmse:
                best_rmse = val_rmse
                no_improve = 0
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": self.model.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "val_rmse": val_rmse,
                }, self.save_dir / "best.pt")
            else:
                no_improve += 1

            # 早停
            if no_improve >= self.patience:
                print(f"[EarlyStop] 连续 {self.patience} 轮无改善，停止训练。",
                      flush=True)
                break

        # 恢复最优权重
        ckpt = torch.load(self.save_dir / "best.pt", map_location=self.device,
                          weights_only=False)
        self.model.load_state_dict(ckpt["model_state_dict"])
        print(f"[Done] 最优模型: epoch={ckpt['epoch']}, "
              f"val_RMSE={ckpt['val_rmse']:.4f}", flush=True)

        return history
