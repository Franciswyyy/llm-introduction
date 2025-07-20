#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
预训练模型评估脚本
直接使用 twitter-roberta-base-sentiment-latest 在 Rotten Tomatoes 数据集上评估性能
无需训练，快速验证模型效果 - 基于 Google Colab 代码优化
"""

import sys
from pathlib import Path

# 添加项目根目录到路径
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.append(str(PROJECT_ROOT))

# 导入配置和core loader
from utils import config, load_model_pipeline
from core.data.loaders import HuggingFaceLoader
import numpy as np
from tqdm import tqdm
from transformers.pipelines.pt_utils import KeyDataset
from sklearn.metrics import classification_report


def main():
    
    # 1. 加载数据集 调用core loader的静态方法
    data = HuggingFaceLoader.load_dataset("rotten_tomatoes")

    if data is None:
        raise Exception("数据集加载失败")

    # 2. 加载模型 (自动缓存和设备选择)
    pipe = load_model_pipeline("cardiffnlp/twitter-roberta-base-sentiment-latest")

    y_pred = []
    for output in tqdm(pipe(KeyDataset(data["test"], "text")), total=len(data["test"])):
        negative_score = output[0]["score"]
        positive_score = output[2]["score"]
        assignment = np.argmax([negative_score, positive_score])
        y_pred.append(assignment)

    y_true = data["test"]["label"]
    evaluate_performance(y_true, y_pred)


def evaluate_performance(y_true, y_pred):
    """Create and print the classification report"""
    performance = classification_report(
        y_true, y_pred,
        target_names=["Negative Review", "Positive Review"]
    )
    print(performance) 


if __name__ == "__main__":
    main() 