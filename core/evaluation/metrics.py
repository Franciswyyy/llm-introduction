#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Evaluation Metrics - 评估指标
提供各种模型评估功能
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, roc_auc_score
)


class BaseEvaluator(ABC):
    """评估器基类"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化评估器
        
        Args:
            config: 配置字典
        """
        self.config = config
    
    @abstractmethod
    def evaluate(self, y_true: np.ndarray, y_pred: np.ndarray, 
                **kwargs) -> Dict[str, Any]:
        """
        评估模型性能
        
        Args:
            y_true: 真实标签
            y_pred: 预测标签
            **kwargs: 额外参数
            
        Returns:
            评估结果字典
        """
        pass


class ClassificationEvaluator(BaseEvaluator):
    """分类任务评估器"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.metrics = config.get('metrics', ['accuracy', 'precision', 'recall', 'f1'])
        self.average = config.get('average', 'weighted')
        self.class_names = config.get('class_names', None)
    
    def evaluate(self, y_true: np.ndarray, y_pred: np.ndarray, 
                y_proba: Optional[np.ndarray] = None,
                class_names: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        评估分类模型性能
        
        Args:
            y_true: 真实标签
            y_pred: 预测标签
            y_proba: 预测概率（可选）
            class_names: 类别名称（可选）
            
        Returns:
            评估结果字典
        """
        results = {}
        
        # 使用传入的类别名称或配置中的类别名称
        if class_names is not None:
            self.class_names = class_names
        
        # 基础指标
        if 'accuracy' in self.metrics:
            results['accuracy'] = accuracy_score(y_true, y_pred)
        
        if 'precision' in self.metrics:
            results['precision'] = precision_score(y_true, y_pred, average=self.average, zero_division=0)
        
        if 'recall' in self.metrics:
            results['recall'] = recall_score(y_true, y_pred, average=self.average, zero_division=0)
        
        if 'f1' in self.metrics:
            results['f1'] = f1_score(y_true, y_pred, average=self.average, zero_division=0)
        
        # 详细分类报告
        if 'classification_report' in self.metrics:
            results['classification_report'] = classification_report(
                y_true, y_pred, 
                target_names=self.class_names,
                output_dict=True,
                zero_division=0
            )
        
        # 混淆矩阵
        if 'confusion_matrix' in self.metrics:
            results['confusion_matrix'] = confusion_matrix(y_true, y_pred)
        
        # ROC AUC (如果有概率预测)
        if y_proba is not None and 'roc_auc' in self.metrics:
            try:
                if len(np.unique(y_true)) == 2:  # 二分类
                    results['roc_auc'] = roc_auc_score(y_true, y_proba[:, 1])
                else:  # 多分类
                    results['roc_auc'] = roc_auc_score(y_true, y_proba, multi_class='ovr', average=self.average)
            except Exception as e:
                results['roc_auc_error'] = str(e)
        
        # 每个类别的详细指标
        if 'per_class_metrics' in self.metrics:
            results['per_class_metrics'] = self._compute_per_class_metrics(y_true, y_pred)
        
        return results
    
    def _compute_per_class_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
        """
        计算每个类别的详细指标
        
        Args:
            y_true: 真实标签
            y_pred: 预测标签
            
        Returns:
            每个类别的指标字典
        """
        unique_labels = np.unique(np.concatenate([y_true, y_pred]))
        per_class_results = {}
        
        for label in unique_labels:
            class_name = self.class_names[label] if self.class_names and label < len(self.class_names) else f"Class_{label}"
            
            # 计算二分类指标
            y_true_binary = (y_true == label).astype(int)
            y_pred_binary = (y_pred == label).astype(int)
            
            per_class_results[class_name] = {
                'precision': precision_score(y_true_binary, y_pred_binary, zero_division=0),
                'recall': recall_score(y_true_binary, y_pred_binary, zero_division=0),
                'f1': f1_score(y_true_binary, y_pred_binary, zero_division=0),
                'support': np.sum(y_true == label)
            }
        
        return per_class_results
    
    def compare_models(self, results_list: List[Dict[str, Any]], 
                      model_names: List[str]) -> Dict[str, Any]:
        """
        比较多个模型的性能
        
        Args:
            results_list: 多个模型的评估结果列表
            model_names: 模型名称列表
            
        Returns:
            模型比较结果
        """
        comparison = {}
        
        # 提取公共指标
        common_metrics = set(results_list[0].keys())
        for results in results_list[1:]:
            common_metrics &= set(results.keys())
        
        # 排除非数值指标
        numeric_metrics = []
        for metric in common_metrics:
            if isinstance(results_list[0][metric], (int, float)):
                numeric_metrics.append(metric)
        
        # 构建比较表
        for metric in numeric_metrics:
            comparison[metric] = {}
            values = [results[metric] for results in results_list]
            
            for i, (name, value) in enumerate(zip(model_names, values)):
                comparison[metric][name] = value
            
            # 找出最佳模型
            if metric in ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']:
                best_idx = np.argmax(values)
            else:
                best_idx = np.argmin(values)  # 对于损失函数等
            
            comparison[metric]['best_model'] = model_names[best_idx]
            comparison[metric]['best_value'] = values[best_idx]
        
        return comparison
    
    def generate_summary(self, results: Dict[str, Any]) -> str:
        """
        生成评估结果摘要
        
        Args:
            results: 评估结果字典
            
        Returns:
            格式化的摘要字符串
        """
        summary_lines = ["📊 模型评估结果摘要", "=" * 50]
        
        # 主要指标
        main_metrics = ['accuracy', 'precision', 'recall', 'f1']
        for metric in main_metrics:
            if metric in results:
                summary_lines.append(f"{metric.upper()}: {results[metric]:.4f}")
        
        # ROC AUC
        if 'roc_auc' in results:
            summary_lines.append(f"ROC AUC: {results['roc_auc']:.4f}")
        
        # 混淆矩阵形状
        if 'confusion_matrix' in results:
            cm = results['confusion_matrix']
            summary_lines.append(f"混淆矩阵形状: {cm.shape}")
            summary_lines.append(f"总预测样本数: {cm.sum()}")
        
        return "\n".join(summary_lines) 