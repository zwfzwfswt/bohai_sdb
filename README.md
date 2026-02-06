# 渤海近岸水深反演系统 (Bohai SDB)

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1+-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

基于多光谱卫星影像（Planet）的浅海水深反演深度学习项目。集成传统方法（Stumpf模型）、机器学习（RF/XGBoost/LightGBM）和深度学习（UNet系列/Swin Transformer）多种方法。

## 🎯 项目概述

本项目旨在使用Planet 8波段多光谱卫星影像结合USV实测水深点，训练深度学习模型进行浅海水深反演（Satellite-Derived Bathymetry, SDB）。项目特别针对渤海近岸海域（0-10米水深）进行优化，实现了多种先进算法的集成与对比。

**核心特点：**
- 🚀 自研 **Swin-BathyUNet++** 模型（多尺度光谱嵌入 + SE注意力 + 密集跳连）
- 📊 支持6种模型：Stumpf / RF / XGBoost / LightGBM / UNet / Swin-BathyUNet / Swin-BathyUNet++
- 🎯 深度分段加权损失（浅水区精度提升）
- 🔧 完整的训练/评估/推理工具链
- 🌊 针对浅海复杂水色环境优化

## 📁 项目结构

```
bohai_sdb/
├── configs/
│   └── default.yaml          # 统一配置文件（数据/训练/模型参数）
├── data/
│   ├── geo_utils.py          # 地理数据处理（影像/矢量/坐标转换）
│   ├── patch_dataset.py      # 深度学习数据集（64×64 patch + 数据增强）
│   └── pixel_dataset.py      # 机器学习特征提取（波段/NDWI/比值）
├── models/
│   ├── stumpf.py             # Stumpf 经验模型（对数波段比）
│   ├── ml_models.py          # 机器学习模型（RF/XGBoost/LightGBM）
│   ├── unet_baseline.py      # UNet 基线模型
│   ├── swin_bathyunet.py     # Swin Transformer + UNet 混合架构
│   └── swin_bathyunet_pp.py  # ⭐ 自研改进模型（Swin-BathyUNet++）
├── engine/
│   ├── trainer.py            # 统一训练流程（混合精度/早停/学习率调度）
│   ├── evaluator.py          # 评估指标（RMSE/MAE/R²/分段评估）
│   └── inference.py          # 模型推理与预测
├── scripts/
│   ├── prepare_samples.py    # 样本准备（从TIF+SHP生成NPZ）
│   ├── train.py              # 统一训练入口
│   ├── infer_raster.py       # 全图水深推理
│   ├── eval_compare.py       # 模型对比评估
│   └── smoke_test.py         # 快速测试脚本
└── requirements.txt          # Python 依赖包
```

## 🔬 核心技术

### 1. Swin-BathyUNet++ 自研模型

**改进点：**

✅ **多尺度光谱嵌入（Multi-Scale Spectral Embedding）**
- 并行提取 1×1 / 3×3 / 5×5 感受野特征
- 融合光谱与空间多尺度信息
- 适应水色复杂变化

✅ **SE 通道注意力（Squeeze-and-Excitation）**
- 自适应学习通道权重
- 强化有效光谱波段
- 抑制冗余信息

✅ **UNet++ 密集跳连（Dense Skip Connections）**
- 跨层特征融合
- 多尺度上下文聚合
- 缓解深层网络梯度消失

✅ **深度分段加权损失**
- 浅水区（<5m）损失权重 ×2
- 针对性提升浅水精度
- 平衡不同深度段表现

### 2. 混合架构优势

```
输入：8波段 Planet 影像 (64×64 patch)
  ↓
多尺度光谱嵌入（1×1/3×3/5×5 并行卷积 + SE）
  ↓
Swin Transformer 编码器（全局上下文建模）
  ├─ Level 1: 64×64
  ├─ Level 2: 32×32
  ├─ Level 3: 16×16
  └─ Level 4: 8×8 (Bottleneck)
  ↓
UNet++ 密集解码器（多尺度特征融合）
  ├─ 上采样 + 密集跳连
  ├─ SE 通道注意力
  └─ 逐层特征重用
  ↓
输出：水深预测图 (64×64)
```

### 3. 数据增强策略

- **几何变换**：随机翻转（水平/垂直）、90°旋转
- **辐射扰动**：高斯噪声（σ=0.01~0.03）、亮度微调（±5%）
- **归一化**：Per-band Z-score 标准化
- **分层采样**：按深度分段均衡采样（避免深度偏斜）

### 4. 损失函数

**深度加权 MSE：**
```python
weight = 2.0 if depth < 5.0m else 1.0
loss = mean(weight × (pred - true)²)
```

## 🚀 快速开始

### 环境安装

```bash
# 克隆仓库
git clone https://github.com/your-repo/bohai_sdb.git
cd bohai_sdb

# 创建虚拟环境
conda create -n bathymetry python=3.9
conda activate bathymetry

# 安装依赖
pip install -r requirements.txt
```

### 数据准备

**输入数据要求：**
1. **Planet 影像**：8波段 GeoTIFF（蓝/绿/红/近红外 + 4个多光谱波段）
2. **实测水深点**：Shapefile 格式，需包含水深字段（如"改正水深"）

```bash
# 生成训练样本（NPZ格式）
python -m scripts.prepare_samples \
    --config configs/default.yaml \
    --planet data/planet_image.tif \
    --usv data/usv_depth_points.shp \
    --output outputs/samples/samples_ps64.npz
```

### 模型训练

**训练自研模型（Swin-BathyUNet++）：**
```bash
python -m scripts.train \
    --config configs/default.yaml \
    --model swin_bathyunet_pp \
    --samples outputs/samples/samples_ps64.npz
```

**训练其他模型：**
```bash
# 机器学习模型
python -m scripts.train --model rf --samples outputs/samples/samples_ps64.npz
python -m scripts.train --model xgboost --samples outputs/samples/samples_ps64.npz

# 深度学习模型
python -m scripts.train --model unet --samples outputs/samples/samples_ps64.npz
python -m scripts.train --model swin_bathyunet --samples outputs/samples/samples_ps64.npz
```

### 模型推理

**全图水深反演：**
```bash
python -m scripts.infer_raster \
    --config configs/default.yaml \
    --model outputs/models/best_model.pth \
    --input data/planet_image.tif \
    --output outputs/depth_map.tif
```

### 模型评估

**对比多个模型：**
```bash
python -m scripts.eval_compare \
    --config configs/default.yaml \
    --models outputs/models/*.pth \
    --samples outputs/samples/samples_ps64.npz
```

## 📊 配置说明

主要配置项（[configs/default.yaml](configs/default.yaml)）：

```yaml
# 数据配置
data:
  patch_size: 64              # Patch 尺寸
  train_ratio: 0.7            # 训练集比例
  val_ratio: 0.15             # 验证集比例
  augmentation: true          # 数据增强

# 训练配置
train:
  epochs: 200                 # 训练轮数
  batch_size: 32              # 批大小
  optimizer: AdamW
  lr: 1.0e-4                  # 学习率
  patience: 30                # 早停耐心值

# 损失函数
loss:
  type: depth_weighted_mse    # mse / depth_weighted_mse
  shallow_threshold: 5.0      # 浅水阈值
  shallow_weight: 2.0         # 浅水权重

# 模型配置
model:
  name: swin_bathyunet_pp     # 模型名称
  in_channels: 8              # 输入波段数
  base_channels: 64           # 基础通道数
  dropout: 0.1                # Dropout 概率
```

## 📈 评估指标

项目采用以下指标评估模型性能：

- **RMSE**（Root Mean Square Error）：均方根误差
- **MAE**（Mean Absolute Error）：平均绝对误差
- **R²**（Coefficient of Determination）：决定系数
- **MAPE**（Mean Absolute Percentage Error）：平均绝对百分比误差

**分深度段评估：**
- 0-2m（极浅水区）
- 2-5m（浅水区）
- 5-10m（中等水深）

## 🎓 应用场景

- 🌊 **海洋测绘**：快速获取大范围水深数据，替代传统声呐测量
- 🐟 **浅海生态监测**：海草床/珊瑚礁栖息地制图
- ⚓ **港口航道安全**：动态监测航道淤积与变化
- 🏖️ **海岸带管理**：滩涂演变/侵蚀沉积分析
- 🚢 **应急响应**：灾后水深变化快速评估

## 📦 依赖包

**核心依赖：**
- PyTorch >= 2.1
- rasterio >= 1.3（遥感影像处理）
- geopandas >= 0.14（矢量数据处理）
- scikit-learn >= 1.3（机器学习）
- xgboost >= 2.0 / lightgbm >= 4.0（集成学习）
- einops >= 0.7（张量操作）

详见 [requirements.txt](requirements.txt)

## 🔧 故障排查

**常见问题：**

1. **CUDA 内存不足**
   - 减小 `batch_size`（如改为16或8）
   - 减小 `patch_size`（如改为32）

2. **训练不收敛**
   - 检查数据归一化是否正确
   - 调整学习率（尝试 5e-5 或 2e-4）
   - 确认损失函数配置

3. **推理速度慢**
   - 使用混合精度推理（FP16）
   - 减小 overlap（降低重叠度）
   - 使用 GPU 加速

## 📝 引用

如果本项目对您的研究有帮助，请引用：

```bibtex
@software{bohai_sdb_2026,
  title={Bohai Satellite-Derived Bathymetry System},
  author={Your Name},
  year={2026},
  url={https://github.com/your-repo/bohai_sdb}
}
```

## 📄 许可证

本项目采用 MIT 许可证 - 详见 [LICENSE](LICENSE) 文件

## 🤝 贡献

欢迎提交 Issue 和 Pull Request！

**开发计划：**
- [ ] 支持 Sentinel-2 多源影像
- [ ] 增加水质参数联合反演（浊度/叶绿素a）
- [ ] 时序变化检测模块
- [ ] Web 可视化界面

## 📧 联系方式

如有问题或合作意向，请联系：
- Email: your.email@example.com
- 项目主页: https://github.com/your-repo/bohai_sdb

---

**⚠️ 免责声明**：本项目水深反演结果仅供科研参考，不作为海图导航依据。实际应用请结合专业测量数据验证。
