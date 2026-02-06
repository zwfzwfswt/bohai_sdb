# -*- coding: utf-8 -*-
"""
经典水深反演模型：
  - StumpfModel:     Stumpf (2003) 双波段对数比模型
  - MultiLinearModel: 多波段线性回归扩展
"""

import numpy as np
from sklearn.linear_model import LinearRegression


class StumpfModel:
    """
    Stumpf 对数比模型：
        depth = a0 + a1 * ln(Rw(λ1)) / ln(Rw(λ2))
    默认选取 green(idx=3) / blue(idx=1) 波段。
    """

    def __init__(self, band_i: int = 3, band_j: int = 1):
        self.band_i = band_i
        self.band_j = band_j
        self.model = LinearRegression()

    def _ratio(self, X: np.ndarray) -> np.ndarray:
        """计算 Stumpf 对数比值特征，X 形状 (N, bands)。"""
        eps = 1e-8
        ln_i = np.log1p(np.clip(X[:, self.band_i], 0, None))
        ln_j = np.log1p(np.clip(X[:, self.band_j], 0, None))
        return (ln_i / (ln_j + eps)).reshape(-1, 1)

    def fit(self, X: np.ndarray, y: np.ndarray):
        feat = self._ratio(X)
        self.model.fit(feat, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        feat = self._ratio(X)
        return self.model.predict(feat).ravel()


class MultiLinearModel:
    """
    多波段线性回归：
        depth = a0 + Σ aᵢ · ln(Rw(λᵢ))
    使用全部波段的 log 值作为自变量。
    """

    def __init__(self):
        self.model = LinearRegression()

    @staticmethod
    def _features(X: np.ndarray) -> np.ndarray:
        return np.log1p(np.clip(X, 0, None))

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.model.fit(self._features(X), y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(self._features(X)).ravel()
