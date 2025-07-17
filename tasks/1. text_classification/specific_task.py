#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
预训练模型评估脚本
直接使用 twitter-roberta-base-sentiment-latest 在 Rotten Tomatoes 数据集上评估性能
无需训练，快速验证模型效果 - 基于 Google Colab 代码优化
"""

# 导入数据管理工具
from utils import get_dataset
from transformers import pipeline
import numpy as np
from tqdm import tqdm
from transformers.pipelines.pt_utils import KeyDataset
from sklearn.metrics import classification_report


def main():
    """主函数"""
    
    # 1. 加载数据集
    data = get_dataset()  
    if data is None:
        raise Exception("数据集加载失败")

    # 2. 加载模型
    model_path = "cardiffnlp/twitter-roberta-base-sentiment-latest"
    # Load model into pipeline
    pipe = pipeline(
        model=model_path,
        tokenizer=model_path,
        return_all_scores=True,
        device="mps"
    )

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