#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Sentence Transformer Embedding - Sentence Transformer嵌入实现
"""

from typing import Dict, List, Any, Union
import numpy as np
from sentence_transformers import SentenceTransformer

from .base import BaseEmbedding


class SentenceTransformerEmbedding(BaseEmbedding):
    """Sentence Transformer嵌入模型实现"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.model_name = config.get('model_name', 'sentence-transformers/all-mpnet-base-v2')
        self.device = config.get('device', None)
        self.normalize_embeddings = config.get('normalize_embeddings', False)
    
    def load_model(self) -> None:
        """加载Sentence Transformer模型"""
        print(f"🤖 正在加载Sentence Transformer模型: {self.model_name}")
        
        try:
            self.model = SentenceTransformer(self.model_name, device=self.device)
            self._is_loaded = True
            print("✅ 模型加载成功")
        except Exception as e:
            print(f"❌ 模型加载失败: {e}")
            raise
    
    def encode(self, texts: Union[str, List[str]], **kwargs) -> np.ndarray:
        """
        将文本编码为嵌入向量
        
        Args:
            texts: 单个文本或文本列表
            **kwargs: 编码参数
            
        Returns:
            嵌入向量数组
        """
        if not self._is_loaded:
            self.load_model()
        
        # 处理单个文本
        if isinstance(texts, str):
            texts = [texts]
        
        # 设置默认参数
        encode_kwargs = {
            'show_progress_bar': kwargs.get('show_progress_bar', False),
            'normalize_embeddings': self.normalize_embeddings,
            'convert_to_numpy': True
        }
        encode_kwargs.update(kwargs)
        
        # 编码
        embeddings = self.model.encode(texts, **encode_kwargs)
        
        return embeddings
    
    def get_embedding_dimension(self) -> int:
        """获取嵌入向量维度"""
        if self.embedding_dim is None:
            if not self._is_loaded:
                self.load_model()
            self.embedding_dim = self.model.get_sentence_embedding_dimension()
        
        return self.embedding_dim
    
    def get_model_info(self) -> Dict[str, Any]:
        """获取模型信息"""
        info = super().get_model_info()
        
        if self._is_loaded:
            info.update({
                'device': str(self.model.device),
                'normalize_embeddings': self.normalize_embeddings,
                'max_seq_length': getattr(self.model[0], 'max_seq_length', None) if hasattr(self.model, '__getitem__') else None
            })
        
        return info 