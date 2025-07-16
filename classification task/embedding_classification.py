#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
嵌入模型分类任务
使用Sentence Transformer将文本转换为嵌入向量，然后训练分类器进行情感分析

包含两种分类方法：
1. 逻辑回归分类器
2. 基于余弦相似度的分类
"""

import sys
import os
import numpy as np
import pandas as pd
from pathlib import Path

# 添加项目根目录到Python路径，以便导入utils模块
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.append(str(PROJECT_ROOT))

# 导入必要的库
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import seaborn as sns

# 导入数据管理工具
from utils import get_dataset

def load_data():
    """
    加载Rotten Tomatoes数据集
    
    Returns:
        dict: 包含train、validation、test分割的数据集
    """
    print("🔄 正在加载数据集...")
    data = get_dataset()
    
    if data is None:
        raise Exception("数据集加载失败")
    
    print("✅ 数据集加载成功")
    print(f"训练集大小: {len(data['train'])}")
    print(f"验证集大小: {len(data['validation'])}")  
    print(f"测试集大小: {len(data['test'])}")
    
    return data

def load_embedding_model():
    """
    加载预训练的Sentence Transformer模型
    
    Returns:
        SentenceTransformer: 已加载的模型实例
    """
    print("🤖 正在加载Sentence Transformer模型...")
    print("模型: sentence-transformers/all-mpnet-base-v2")
    
    # 加载预训练的多语言句子嵌入模型
    # all-mpnet-base-v2 是一个高质量的句子嵌入模型
    model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
    
    print("✅ 模型加载成功")
    return model

def generate_embeddings(model, data):
    """
    将文本转换为嵌入向量
    
    Args:
        model: Sentence Transformer模型
        data: 数据集
        
    Returns:
        tuple: (训练集嵌入, 测试集嵌入)
    """
    print("🔄 正在生成文本嵌入...")
    
    # 将训练集文本转换为嵌入向量
    print("  正在处理训练集...")
    train_embeddings = model.encode(data["train"]["text"], show_progress_bar=True)
    
    # 将测试集文本转换为嵌入向量  
    print("  正在处理测试集...")
    test_embeddings = model.encode(data["test"]["text"], show_progress_bar=True)
    
    print(f"✅ 嵌入生成完成")
    print(f"训练集嵌入形状: {train_embeddings.shape}")
    print(f"测试集嵌入形状: {test_embeddings.shape}")
    
    return train_embeddings, test_embeddings

def train_logistic_regression(train_embeddings, train_labels):
    """
    训练逻辑回归分类器
    
    Args:
        train_embeddings: 训练集嵌入向量
        train_labels: 训练集标签
        
    Returns:
        LogisticRegression: 训练好的分类器
    """
    print("🎯 正在训练逻辑回归分类器...")
    
    # 创建逻辑回归分类器
    # random_state=42 确保结果可重现
    clf = LogisticRegression(random_state=42, max_iter=1000)
    
    # 在训练集嵌入上训练分类器
    clf.fit(train_embeddings, train_labels)
    
    print("✅ 逻辑回归训练完成")
    return clf

def cosine_similarity_classification(train_embeddings, train_labels, test_embeddings):
    """
    基于余弦相似度的分类方法
    
    该方法计算每个类别的平均嵌入向量，然后将测试样本与最相似的类别匹配
    
    Args:
        train_embeddings: 训练集嵌入向量
        train_labels: 训练集标签
        test_embeddings: 测试集嵌入向量
        
    Returns:
        numpy.ndarray: 预测标签
    """
    print("🔄 正在执行基于余弦相似度的分类...")
    
    # 将嵌入和标签合并为DataFrame以便分组操作
    # 假设嵌入维度为768（all-mpnet-base-v2的输出维度）
    embedding_dim = train_embeddings.shape[1]
    
    # 创建包含嵌入和标签的DataFrame
    df_data = np.hstack([train_embeddings, np.array(train_labels).reshape(-1, 1)])
    df = pd.DataFrame(df_data)
    
    # 按标签分组，计算每个类别的平均嵌入向量
    # 最后一列是标签，前面的列是嵌入特征
    averaged_target_embeddings = df.groupby(embedding_dim).mean().iloc[:, :-1].values
    
    print(f"类别平均嵌入形状: {averaged_target_embeddings.shape}")
    
    # 计算测试嵌入与每个类别平均嵌入的余弦相似度
    sim_matrix = cosine_similarity(test_embeddings, averaged_target_embeddings)
    
    # 选择相似度最高的类别作为预测结果
    y_pred = np.argmax(sim_matrix, axis=1)
    
    print("✅ 余弦相似度分类完成")
    return y_pred

def evaluate_performance(y_true, y_pred, method_name=""):
    """
    评估分类性能
    
    Args:
        y_true: 真实标签
        y_pred: 预测标签  
        method_name: 方法名称（用于输出标识）
    """
    print(f"\n📊 {method_name}性能评估:")
    print("=" * 50)
    
    # 打印详细的分类报告
    report = classification_report(y_true, y_pred, 
                                 target_names=['负面评价', '正面评价'],
                                 digits=4)
    print(report)
    
    # 计算混淆矩阵
    cm = confusion_matrix(y_true, y_pred)
    
    # 可视化混淆矩阵
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['负面评价', '正面评价'],
                yticklabels=['负面评价', '正面评价'])
    plt.title(f'{method_name}混淆矩阵')
    plt.ylabel('实际标签')
    plt.xlabel('预测标签')
    
    # 保存混淆矩阵图像
    output_path = Path(__file__).parent / f'{method_name}_confusion_matrix.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"混淆矩阵已保存至: {output_path}")
    plt.show()

def main():
    """
    主函数：执行完整的嵌入分类流程
    """
    print("🚀 开始执行嵌入模型分类任务")
    print("=" * 60)
    
    try:
        # 1. 加载数据集
        data = load_data()
        
        # 2. 加载嵌入模型
        model = load_embedding_model()
        
        # 3. 生成嵌入向量
        train_embeddings, test_embeddings = generate_embeddings(model, data)
        
        # 4. 方法一：逻辑回归分类
        print("\n" + "="*60)
        print("方法一：逻辑回归分类")
        print("="*60)
        
        clf = train_logistic_regression(train_embeddings, data["train"]["label"])
        lr_predictions = clf.predict(test_embeddings)
        evaluate_performance(data["test"]["label"], lr_predictions, "逻辑回归")
        
        # 5. 方法二：余弦相似度分类  
        print("\n" + "="*60)
        print("方法二：余弦相似度分类")
        print("="*60)
        
        cosine_predictions = cosine_similarity_classification(
            train_embeddings, data["train"]["label"], test_embeddings
        )
        evaluate_performance(data["test"]["label"], cosine_predictions, "余弦相似度")
        
        print("\n🎉 分类任务完成！")
        
    except Exception as e:
        print(f"❌ 执行过程中出现错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 