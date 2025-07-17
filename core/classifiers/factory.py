#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Classifier Factory - 分类器工厂
"""

from typing import Dict, Any
from .supervised import LogisticRegressionClassifier, SVMClassifier, RandomForestClassifier
from .similarity import SimilarityClassifier


def create_classifier(classifier_type: str, config: Dict[str, Any]):
    """创建分类器"""
    if classifier_type == "LogisticRegressionClassifier":
        return LogisticRegressionClassifier(config)
    elif classifier_type == "SVMClassifier":
        return SVMClassifier(config)
    elif classifier_type == "RandomForestClassifier":
        return RandomForestClassifier(config)
    elif classifier_type == "SimilarityClassifier":
        return SimilarityClassifier(config)
    else:
        raise ValueError(f"不支持的分类器类型: {classifier_type}")
