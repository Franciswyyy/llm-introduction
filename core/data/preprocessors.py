#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Data Preprocessors - 数据预处理器实现
"""

from typing import Dict, List, Any
import re
from .base import BaseDataPreprocessor


class TextPreprocessor(BaseDataPreprocessor):
    """基础文本预处理器"""
    
    def preprocess_texts(self, texts: List[str]) -> List[str]:
        """基础文本清洗"""
        processed = []
        for text in texts:
            # 简单的文本清洗
            text = re.sub(r'\s+', ' ', text)  # 合并多个空格
            text = text.strip()
            processed.append(text)
        return processed
    
    def preprocess_labels(self, labels: List[Any]) -> List[Any]:
        """标签预处理（这里直接返回）"""
        return labels
