# -*- coding: utf-8 -*-
"""
机器学习水深反演模型统一封装：RF / XGBoost / LightGBM。
接口：fit(X, y), predict(X), get_importance()
"""

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV

try:
    import xgboost as xgb
except ImportError:
    xgb = None

try:
    import lightgbm as lgb
except ImportError:
    lgb = None


class _BaseMLModel:
    """统一接口基类。"""

    def __init__(self):
        self.model = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.model.fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X).ravel()

    def get_importance(self) -> np.ndarray | None:
        if hasattr(self.model, "feature_importances_"):
            return self.model.feature_importances_
        return None


class RFModel(_BaseMLModel):
    """Random Forest 回归。"""

    def __init__(self, n_estimators: int = 500, max_depth: int = 20,
                 random_state: int = 42, **kwargs):
        super().__init__()
        self.model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
            n_jobs=-1,
            **kwargs,
        )


class XGBModel(_BaseMLModel):
    """XGBoost 回归。"""

    def __init__(self, n_estimators: int = 500, max_depth: int = 8,
                 learning_rate: float = 0.05, random_state: int = 42,
                 **kwargs):
        super().__init__()
        if xgb is None:
            raise ImportError("请安装 xgboost: pip install xgboost")
        self.model = xgb.XGBRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            random_state=random_state,
            n_jobs=-1,
            tree_method="hist",
            **kwargs,
        )


class LGBModel(_BaseMLModel):
    """LightGBM 回归。"""

    def __init__(self, n_estimators: int = 500, max_depth: int = 8,
                 learning_rate: float = 0.05, random_state: int = 42,
                 **kwargs):
        super().__init__()
        if lgb is None:
            raise ImportError("请安装 lightgbm: pip install lightgbm")
        self.model = lgb.LGBMRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            random_state=random_state,
            n_jobs=-1,
            verbose=-1,
            **kwargs,
        )


def build_ml_model(name: str, cfg: dict) -> _BaseMLModel:
    """根据名称和配置构建 ML 模型。"""
    if name == "rf":
        return RFModel(**cfg.get("rf", {}))
    if name == "xgboost":
        return XGBModel(**cfg.get("xgboost", {}))
    if name == "lgbm":
        return LGBModel(**cfg.get("lgbm", {}))
    raise ValueError(f"未知 ML 模型: {name}")
