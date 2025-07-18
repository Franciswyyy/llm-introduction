#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
零样本分类任务
使用预训练的嵌入模型，无需训练数据，直接通过标签描述和文本相似度进行分类
"""

import sys
import numpy as np
from pathlib import Path

# 添加项目根目录到Python路径
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.append(str(PROJECT_ROOT))

# 导入必要的库和模块
from sklearn.metrics import classification_report
from sklearn.metrics.pairwise import cosine_similarity
from core.data.loaders import HuggingFaceLoader
from utils import load_embedding_model


def evaluate_performance(y_true, y_pred):
    """
    评估分类性能
    
    Args:
        y_true: 真实标签
        y_pred: 预测标签
    """
    print("📊 零样本分类性能评估:")
    print("=" * 50)
    
    # 详细分类报告
    report = classification_report(
        y_true, y_pred,
        target_names=["负面评价", "正面评价"],
        digits=4
    )
    print(report)


def main():
    try:
        # 1. 加载数据集
        print("📊 加载数据集...")
        data = HuggingFaceLoader.load_dataset("rotten_tomatoes")
        
        # 2. 加载嵌入模型（使用配置化缓存系统）
        print("🤖 加载嵌入模型...")
        model = load_embedding_model('sentence-transformers/all-mpnet-base-v2')
        
        # 3. Create embeddings for our labels
        print("🏷️  创建标签嵌入...")
        label_embeddings = model.encode(["A negative review", "A positive review"])
        
        # 4. 生成测试文本嵌入
        print("📝 生成测试文本嵌入...")
        test_texts = list(data["test"]["text"])
        test_embeddings = model.encode(test_texts, show_progress_bar=True)
        
        # 5. 核心零样本分类算法
        print("🔮 执行零样本分类...")
        sim_matrix = cosine_similarity(test_embeddings, label_embeddings)
        y_pred = np.argmax(sim_matrix, axis=1)
        
        # 6. 评估性能
        evaluate_performance(data["test"]["label"], y_pred)
        
    except Exception as e:
        print(f"❌ 执行过程中出现错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 