# MNIST手写数字识别 — 基于卷积神经网络（CNN）

《人工智能》课程论文项目：使用 PyTorch 实现 CNN 对 MNIST 手写数字数据集进行分类。

## 项目结构

```
├── src/
│   ├── model.py          # CNN模型定义（BasicCNN + MiniResNet）
│   ├── train.py          # 训练主脚本（含三组对照实验）
│   ├── data_loader.py    # 数据加载与预处理
│   ├── evaluate.py       # 评估与可视化
│   └── utils.py          # 辅助函数
├── figures/              # 论文用图（训练后生成）
├── outputs/              # 模型权重与训练日志（训练后生成）
├── requirements.txt      # Python依赖
└── README.md
```

## 环境要求

- Python 3.8+
- PyTorch 2.0+ (CUDA 可选但推荐)
- 详见 `requirements.txt`

## 安装

```bash
pip install -r requirements.txt
```

## 运行

### 训练

```bash
python src/train.py
```

将依次执行三组对照实验：
1. BasicCNN + Adam 优化器
2. BasicCNN + SGD 优化器
3. MiniResNet + Adam 优化器

模型权重与训练日志保存至 `outputs/` 目录。

### 评估与可视化

```bash
python src/evaluate.py
```

生成以下论文用图至 `figures/` 目录：
- 数据集样本展示
- 训练/验证 Loss 与 Accuracy 曲线
- 测试集混淆矩阵
- 正确/错误分类样本可视化
- 卷积核与特征图可视化
- 模型对照实验对比图

## 模型架构

### BasicCNN
```
Conv2d(1,32,3) → BN → ReLU → MaxPool(2)
Conv2d(32,64,3) → BN → ReLU → MaxPool(2)
Conv2d(64,128,3) → BN → ReLU → MaxPool(2)
FC(1152,256) → ReLU → Dropout(0.5)
FC(256,10) → LogSoftmax
```

### MiniResNet (对照实验)
```
Conv2d(1,32,3) → BN → ReLU
ResidualBlock(32→32) ×2
ResidualBlock(32→64, stride=2) ×2
GlobalAvgPool → FC(64,10) → LogSoftmax
```

## 实验设置

| 参数 | 值 |
|------|-----|
| Epochs | 20 (早停 patience=5) |
| Batch Size | 64 |
| 优化器 | Adam (lr=0.001) / SGD (lr=0.01, momentum=0.9) |
| 学习率调度 | ReduceLROnPlateau (factor=0.5, patience=3) |
| 数据增强 | 随机旋转 ±10° |

## 数据集

[MNIST](http://yann.lecun.com/exdb/mnist/) — 手写数字灰度图像，28×28像素，10类（0-9），训练集60,000张，测试集10,000张。代码运行后自动下载至 `data/` 目录。
