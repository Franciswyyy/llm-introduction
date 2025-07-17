#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Result Visualizer - 结果可视化器
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from typing import Dict, Any, List


class ResultVisualizer:
    """结果可视化器"""
    
    def plot_confusion_matrix(self, cm: np.ndarray, class_names: List[str], 
                             title: str = "混淆矩阵"):
        """绘制混淆矩阵"""
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=class_names,
                    yticklabels=class_names)
        plt.title(title)
        plt.ylabel('实际标签')
        plt.xlabel('预测标签')
        plt.show()
    
    def plot_metrics_comparison(self, results: Dict[str, Dict[str, Any]], 
                               title: str = "模型性能对比"):
        """绘制指标对比图"""
        metrics = ['accuracy', 'precision', 'recall', 'f1']
        model_names = list(results.keys())
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        x = np.arange(len(metrics))
        width = 0.35
        
        for i, model_name in enumerate(model_names):
            values = [results[model_name].get(metric, 0) for metric in metrics]
            ax.bar(x + i * width, values, width, label=model_name)
        
        ax.set_xlabel('指标')
        ax.set_ylabel('分数')
        ax.set_title(title)
        ax.set_xticks(x + width / 2)
        ax.set_xticklabels(metrics)
        ax.legend()
        
        plt.tight_layout()
        plt.show()
