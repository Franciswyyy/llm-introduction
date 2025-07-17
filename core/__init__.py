#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LLM Introduction - Core Module
提供文本分类的核心功能模块
"""

__version__ = "2.0.0"
__author__ = "AI Assistant"
__description__ = "模块化文本分类框架"

# 导入核心组件
from .data.base import BaseDataLoader
from .embeddings.base import BaseEmbedding
from .classifiers.base import BaseClassifier
from .evaluation.metrics import BaseEvaluator
from .pipeline.text_classification import TextClassificationPipeline

__all__ = [
    'BaseDataLoader',
    'BaseEmbedding', 
    'BaseClassifier',
    'BaseEvaluator',
    'TextClassificationPipeline'
] 