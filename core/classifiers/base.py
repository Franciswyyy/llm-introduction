#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Base Classifier - 分类器基类
定义分类器的统一接口
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Union
import numpy as np
import pickle
from pathlib import Path


class BaseClassifier(ABC):
    """分类器基类"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化分类器
        
        Args:
            config: 配置字典，包含分类器相关参数
        """
        self.config = config
        self.model = None
        self.is_trained = False
        self.label_encoder = None
        self.class_names = None
    
    @abstractmethod
    def train(self, X: np.ndarray, y: np.ndarray, **kwargs) -> None:
        """
        训练分类器
        
        Args:
            X: 训练特征，形状为 (n_samples, n_features)
            y: 训练标签，形状为 (n_samples,)
            **kwargs: 训练参数
        """
        pass
    
    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        进行预测
        
        Args:
            X: 测试特征，形状为 (n_samples, n_features)
            
        Returns:
            预测标签，形状为 (n_samples,)
        """
        pass
    
    def predict_proba(self, X: np.ndarray) -> Optional[np.ndarray]:
        """
        预测类别概率
        
        Args:
            X: 测试特征，形状为 (n_samples, n_features)
            
        Returns:
            类别概率，形状为 (n_samples, n_classes)，如果不支持则返回None
        """
        if hasattr(self.model, 'predict_proba'):
            return self.model.predict_proba(X)
        return None
    
    def fit_transform_labels(self, labels: List[Any]) -> np.ndarray:
        """
        拟合并转换标签
        
        Args:
            labels: 原始标签列表
            
        Returns:
            编码后的标签数组
        """
        from sklearn.preprocessing import LabelEncoder
        
        if self.label_encoder is None:
            self.label_encoder = LabelEncoder()
            encoded_labels = self.label_encoder.fit_transform(labels)
            self.class_names = self.label_encoder.classes_.tolist()
        else:
            encoded_labels = self.label_encoder.transform(labels)
        
        return encoded_labels
    
    def inverse_transform_labels(self, encoded_labels: np.ndarray) -> List[Any]:
        """
        反向转换标签
        
        Args:
            encoded_labels: 编码后的标签
            
        Returns:
            原始标签列表
        """
        if self.label_encoder is None:
            return encoded_labels.tolist()
        return self.label_encoder.inverse_transform(encoded_labels).tolist()
    
    def save_model(self, filepath: str) -> None:
        """
        保存模型到文件
        
        Args:
            filepath: 保存路径
        """
        model_data = {
            'model': self.model,
            'config': self.config,
            'is_trained': self.is_trained,
            'label_encoder': self.label_encoder,
            'class_names': self.class_names
        }
        
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
    
    def load_model(self, filepath: str) -> None:
        """
        从文件加载模型
        
        Args:
            filepath: 模型文件路径
        """
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.model = model_data['model']
        self.config = model_data['config']
        self.is_trained = model_data['is_trained']
        self.label_encoder = model_data['label_encoder']
        self.class_names = model_data['class_names']
    
    def get_feature_importance(self) -> Optional[np.ndarray]:
        """
        获取特征重要性
        
        Returns:
            特征重要性数组，如果不支持则返回None
        """
        if hasattr(self.model, 'feature_importances_'):
            return self.model.feature_importances_
        elif hasattr(self.model, 'coef_'):
            return np.abs(self.model.coef_[0])
        return None
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        获取模型信息
        
        Returns:
            模型信息字典
        """
        info = {
            'classifier_type': self.__class__.__name__,
            'config': self.config,
            'is_trained': self.is_trained,
            'class_names': self.class_names,
            'num_classes': len(self.class_names) if self.class_names else None
        }
        
        # 添加模型特定信息
        if hasattr(self.model, 'get_params'):
            info['model_params'] = self.model.get_params()
        
        return info
    
    def cross_validate(self, X: np.ndarray, y: np.ndarray, 
                      cv: int = 5, scoring: str = 'accuracy') -> Dict[str, Any]:
        """
        交叉验证
        
        Args:
            X: 特征数据
            y: 标签数据
            cv: 交叉验证折数
            scoring: 评分方法
            
        Returns:
            交叉验证结果
        """
        from sklearn.model_selection import cross_validate
        from sklearn.base import clone
        
        if not self.is_trained:
            raise ValueError("模型尚未训练，无法进行交叉验证")
        
        model_copy = clone(self.model)
        cv_results = cross_validate(model_copy, X, y, cv=cv, scoring=scoring, 
                                   return_train_score=True)
        
        return {
            'test_scores': cv_results['test_score'],
            'train_scores': cv_results['train_score'],
            'mean_test_score': cv_results['test_score'].mean(),
            'std_test_score': cv_results['test_score'].std(),
            'mean_train_score': cv_results['train_score'].mean(),
            'std_train_score': cv_results['train_score'].std()
        } 