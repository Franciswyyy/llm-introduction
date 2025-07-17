#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Base Embedding - 嵌入模型基类
定义文本嵌入的统一接口
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Union
import numpy as np


class BaseEmbedding(ABC):
    """嵌入模型基类"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化嵌入模型
        
        Args:
            config: 配置字典，包含模型相关参数
        """
        self.config = config
        self.model = None
        self.embedding_dim = None
        self._is_loaded = False
    
    @abstractmethod
    def load_model(self) -> None:
        """加载嵌入模型"""
        pass
    
    @abstractmethod
    def encode(self, texts: Union[str, List[str]], **kwargs) -> np.ndarray:
        """
        将文本编码为嵌入向量
        
        Args:
            texts: 单个文本或文本列表
            **kwargs: 编码参数
            
        Returns:
            嵌入向量数组，形状为 (n_texts, embedding_dim)
        """
        pass
    
    def encode_batch(self, texts: List[str], batch_size: int = 32, 
                    show_progress: bool = True) -> np.ndarray:
        """
        批量编码文本
        
        Args:
            texts: 文本列表
            batch_size: 批处理大小
            show_progress: 是否显示进度条
            
        Returns:
            嵌入向量数组
        """
        if not self._is_loaded:
            self.load_model()
        
        all_embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            batch_embeddings = self.encode(batch_texts, show_progress_bar=False)
            all_embeddings.append(batch_embeddings)
        
        return np.vstack(all_embeddings)
    
    def get_embedding_dimension(self) -> int:
        """
        获取嵌入向量维度
        
        Returns:
            嵌入向量维度
        """
        if self.embedding_dim is None:
            # 用一个样本文本测试获取维度
            sample_embedding = self.encode(["test"])
            self.embedding_dim = sample_embedding.shape[1]
        
        return self.embedding_dim
    
    def save_embeddings(self, embeddings: np.ndarray, filepath: str) -> None:
        """
        保存嵌入向量到文件
        
        Args:
            embeddings: 嵌入向量数组
            filepath: 保存路径
        """
        np.save(filepath, embeddings)
    
    def load_embeddings(self, filepath: str) -> np.ndarray:
        """
        从文件加载嵌入向量
        
        Args:
            filepath: 文件路径
            
        Returns:
            嵌入向量数组
        """
        return np.load(filepath)
    
    def compute_similarity(self, embeddings1: np.ndarray, 
                          embeddings2: np.ndarray, 
                          metric: str = "cosine") -> np.ndarray:
        """
        计算嵌入向量间的相似度
        
        Args:
            embeddings1: 第一组嵌入向量
            embeddings2: 第二组嵌入向量  
            metric: 相似度度量方法 ("cosine", "euclidean", "dot")
            
        Returns:
            相似度矩阵
        """
        if metric == "cosine":
            from sklearn.metrics.pairwise import cosine_similarity
            return cosine_similarity(embeddings1, embeddings2)
        elif metric == "euclidean":
            from sklearn.metrics.pairwise import euclidean_distances
            return -euclidean_distances(embeddings1, embeddings2)  # 负距离作为相似度
        elif metric == "dot":
            return np.dot(embeddings1, embeddings2.T)
        else:
            raise ValueError(f"不支持的相似度度量方法: {metric}")
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        获取模型信息
        
        Returns:
            模型信息字典
        """
        return {
            'model_name': self.config.get('model_name', 'unknown'),
            'embedding_dimension': self.get_embedding_dimension(),
            'config': self.config,
            'is_loaded': self._is_loaded
        } 