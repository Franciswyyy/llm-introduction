#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Similarity Classifier - 相似度分类器实现
"""

from typing import Dict, Any
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

from .base import BaseClassifier


class SimilarityClassifier(BaseClassifier):
    """基于相似度的分类器"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.metric = config.get('metric', 'cosine')
        self.class_centers = None
    
    def train(self, X: np.ndarray, y: np.ndarray, **kwargs) -> None:
        """训练相似度分类器（实际是计算类别中心）"""
        print("🎯 正在准备相似度分类器...")
        
        # 直接使用numpy计算类别中心，避免DataFrame的key type问题
        unique_labels = np.unique(y)
        self.class_centers = []
        
        for label in unique_labels:
            # 找到属于当前标签的所有样本
            mask = (y == label)
            label_embeddings = X[mask]
            # 计算该标签的平均嵌入
            center = np.mean(label_embeddings, axis=0)
            self.class_centers.append(center)
        
        self.class_centers = np.array(self.class_centers)
        
        self.is_trained = True
        print(f"✅ 相似度分类器准备完成，计算了 {len(self.class_centers)} 个类别中心")
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """基于相似度进行预测"""
        if not self.is_trained:
            raise ValueError("模型尚未训练")
        
        if self.metric == 'cosine':
            # 计算余弦相似度
            similarities = cosine_similarity(X, self.class_centers)
        elif self.metric == 'euclidean':
            # 计算欧几里得距离（转换为相似度）
            from sklearn.metrics.pairwise import euclidean_distances
            distances = euclidean_distances(X, self.class_centers)
            similarities = -distances  # 负距离作为相似度
        elif self.metric == 'dot':
            # 点积相似度
            similarities = np.dot(X, self.class_centers.T)
        else:
            raise ValueError(f"不支持的相似度度量: {self.metric}")
        
        # 选择相似度最高的类别
        predictions = np.argmax(similarities, axis=1)
        
        return predictions
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """预测类别概率（基于相似度的软分类）"""
        if not self.is_trained:
            raise ValueError("模型尚未训练")
        
        if self.metric == 'cosine':
            similarities = cosine_similarity(X, self.class_centers)
        elif self.metric == 'dot':
            similarities = np.dot(X, self.class_centers.T)
        else:
            # 对于距离度量，先转换为相似度
            from sklearn.metrics.pairwise import euclidean_distances
            distances = euclidean_distances(X, self.class_centers)
            similarities = 1 / (1 + distances)  # 转换为相似度
        
        # 将相似度转换为概率（使用softmax）
        exp_similarities = np.exp(similarities - np.max(similarities, axis=1, keepdims=True))
        probabilities = exp_similarities / np.sum(exp_similarities, axis=1, keepdims=True)
        
        return probabilities
    
    def get_model_info(self) -> Dict[str, Any]:
        """获取模型信息"""
        info = super().get_model_info()
        info.update({
            'similarity_metric': self.metric,
            'num_class_centers': len(self.class_centers) if self.class_centers is not None else None,
            'class_centers_shape': self.class_centers.shape if self.class_centers is not None else None
        })
        return info 