#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Supervised Classifiers - 监督学习分类器实现
"""

from typing import Dict, Any
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier as RFClassifier

from .base import BaseClassifier


class LogisticRegressionClassifier(BaseClassifier):
    """逻辑回归分类器"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        # 默认参数
        self.model_params = {
            'random_state': 42,
            'max_iter': 1000,
            **config.get('params', {})
        }
    
    def train(self, X: np.ndarray, y: np.ndarray, **kwargs) -> None:
        """训练逻辑回归分类器"""
        print("🎯 正在训练逻辑回归分类器...")
        
        # 更新参数
        params = {**self.model_params, **kwargs}
        
        # 创建并训练模型
        self.model = LogisticRegression(**params)
        self.model.fit(X, y)
        
        self.is_trained = True
        print("✅ 逻辑回归分类器训练完成")
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """进行预测"""
        if not self.is_trained:
            raise ValueError("模型尚未训练")
        
        return self.model.predict(X)


class SVMClassifier(BaseClassifier):
    """支持向量机分类器"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        # 默认参数
        self.model_params = {
            'kernel': 'rbf',
            'random_state': 42,
            'probability': True,  # 启用概率预测
            **config.get('params', {})
        }
    
    def train(self, X: np.ndarray, y: np.ndarray, **kwargs) -> None:
        """训练SVM分类器"""
        print("🎯 正在训练SVM分类器...")
        
        # 更新参数
        params = {**self.model_params, **kwargs}
        
        # 创建并训练模型
        self.model = SVC(**params)
        self.model.fit(X, y)
        
        self.is_trained = True
        print("✅ SVM分类器训练完成")
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """进行预测"""
        if not self.is_trained:
            raise ValueError("模型尚未训练")
        
        return self.model.predict(X)


class RandomForestClassifier(BaseClassifier):
    """随机森林分类器"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        # 默认参数
        self.model_params = {
            'n_estimators': 100,
            'random_state': 42,
            'n_jobs': -1,
            **config.get('params', {})
        }
    
    def train(self, X: np.ndarray, y: np.ndarray, **kwargs) -> None:
        """训练随机森林分类器"""
        print("🎯 正在训练随机森林分类器...")
        
        # 更新参数
        params = {**self.model_params, **kwargs}
        
        # 创建并训练模型
        self.model = RFClassifier(**params)
        self.model.fit(X, y)
        
        self.is_trained = True
        print("✅ 随机森林分类器训练完成")
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """进行预测"""
        if not self.is_trained:
            raise ValueError("模型尚未训练")
        
        return self.model.predict(X) 