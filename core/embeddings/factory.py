#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Embedding Factory - 嵌入模型工厂
"""

from typing import Dict, Any
from .sentence_transformer import SentenceTransformerEmbedding


def create_embedding(embedding_type: str, config: Dict[str, Any]):
    """创建嵌入模型"""
    if embedding_type == "SentenceTransformerEmbedding":
        return SentenceTransformerEmbedding(config)
    else:
        raise ValueError(f"不支持的嵌入模型类型: {embedding_type}")
