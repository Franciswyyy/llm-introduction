# 🎬 电影评论情感分析 - Mac M2版

轻量化的电影评论情感分析项目，使用 Rotten Tomatoes 数据集和 twitter-roberta-base-sentiment-latest 模型，专为 Mac M2 优化。

## 📋 项目概述

- **任务**: 电影评论二分类（好评/差评）
- **模型**: twitter-roberta-base-sentiment-latest (~500MB)
- **数据集**: Rotten Tomatoes (~1MB)
- **优化**: Mac M2 Apple Silicon MPS 加速

## 💾 存储信息
1
### 文件大小
- **预训练模型**: ~500 MB
- **数据集**: ~1 MB
- **训练后模型**: ~500 MB
- **总存储需求**: ~1 GB

### 本地文件结构
```
ai-hello/
├── requirements.txt           # Python依赖
├── evaluate_pretrained.py    # ⭐️ 快速评估脚本（推荐新手）
├── train_model.py            # 训练微调脚本
├── predict.py                # 预测脚本
├── README.md                 # 说明文档
├── resources/                # 资源文件夹
│   ├── datasets/             # 数据集（自动创建）
│   │   └── rotten_tomatoes/
│   ├── pretrained_models/    # 预训练模型（自动创建）
│   │   └── twitter-roberta-base-sentiment-latest/
│   └── trained_models/       # 训练后模型（自动创建）
├── confusion_matrix.png      # 训练模型混淆矩阵（自动生成）
└── pretrained_confusion_matrix.png  # 预训练模型混淆矩阵（自动生成）
```

## 🚀 快速开始

### 方案一：快速评估（推荐新手） ⭐️
无需训练，直接评估预训练模型性能：
```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 快速评估预训练模型
python evaluate_pretrained.py
```
**优点：** 快速看到模型效果，只需几分钟

### 方案二：完整训练流程
训练微调模型以获得更好性能：
```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 训练模型
python train_model.py

# 3. 预测测试
python predict.py
```
**首次运行会自动下载：**
- twitter-roberta-base-sentiment-latest 模型 (~500MB)
- Rotten Tomatoes 数据集 (~1MB)

**训练时间：** 在 Mac M2 上约 10-20 分钟

## 🔧 Mac M2 优化特性

✅ **完全兼容 Mac M2**
- 自动检测并使用 MPS (Metal Performance Shaders) 加速
- 针对 Apple Silicon 优化的训练参数
- 内存使用优化配置
- 3-5倍于CPU的性能提升

## 📊 预期性能

- **测试准确率**: 85-90%
- **训练时间**: 10-20分钟 (Mac M2)
- **推理速度**: 每秒数百条评论
- **内存使用**: ~3GB

## 🎯 使用示例

### 快速评估输出（方案一）
```
🎬 预训练模型快速评估
🚀 基于 twitter-roberta-base-sentiment-latest
============================================================
📥 下载 Rotten Tomatoes 数据集...
✅ 数据集信息:
   训练集: 8,530 条
   验证集: 1,066 条
   测试集: 1,066 条
🔧 创建推理Pipeline:
   模型: cardiffnlp/twitter-roberta-base-sentiment-latest
   设备: mps
✅ Pipeline创建成功

🚀 开始推理...
推理进度: 100%|████████████| 1066/1066 [00:45<00:00, 23.67it/s]
✅ 推理完成

📊 模型性能评估:
🎯 整体准确率: 0.8146
```

### 训练输出（方案二）
```
🎬 电影评论情感分析模型训练
==================================================
✅ 使用 Apple MPS 加速
📥 下载模型: cardiffnlp/twitter-roberta-base-sentiment-latest (~500MB)
📥 下载 Rotten Tomatoes 数据集 (~1MB)
✅ 数据集已下载到: ./datasets
   训练集: 8,530 条
   验证集: 1,066 条
   测试集: 1,066 条
🚀 开始训练...
✅ 训练完成！模型已保存到: ./trained_model
✅ 测试准确率: 0.8736
```

### 预测示例
```
🎬 电影评论情感分析预测器
========================================
✅ 使用 MPS 加速预测
🎯 使用训练后的模型: ./trained_model

请输入电影评论: This movie is absolutely amazing!
📊 预测结果:
   情感: 好评 👍
   置信度: 0.952
   🎯 高置信度
```

## 🛠️ 常见问题

### MPS 不可用
```
⚠️  使用 CPU
```
**解决方案:**
- 确保 PyTorch 版本 >= 1.12
- 确保 macOS 版本 >= 12.3
- 重新安装：`pip install torch torchvision torchaudio`

### 内存不足
**解决方案:**
- 关闭其他占用内存的应用
- 降低 batch size（修改 train_model.py 中的 `per_device_train_batch_size`）

### 下载失败
**解决方案:**
- 检查网络连接
- 重新运行脚本（支持断点续传）
- 使用代理：`export https_proxy=http://your-proxy:port`

## 📈 模型详细信息

### twitter-roberta-base-sentiment-latest
- **架构**: RoBERTa-base
- **参数量**: 125M
- **预训练数据**: Twitter 文本
- **语言**: 英文为主，部分中文支持
- **输出**: 二分类情感分析

### Rotten Tomatoes 数据集
- **训练集**: 8,530 条电影评论
- **验证集**: 1,066 条评论
- **测试集**: 1,066 条评论
- **标签**: 0=差评, 1=好评
- **语言**: 英文

## 🧹 清理文件

如需清理下载的文件以释放空间：
```bash
# 删除模型文件
rm -rf resources/
！
# 删除生成的图片
rm -f confusion_matrix.png
```
@
---

**🎉 开始你的情感分析之旅！** 
