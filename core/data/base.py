#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Base Data Loader - 数据加载器基类
定义数据加载的统一接口
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Any, Optional
import pandas as pd


class BaseDataLoader(ABC):
    """数据加载器基类"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化数据加载器
        
        Args:
            config: 配置字典，包含数据源相关参数
        """
        self.config = config
        self.data = None
        self._is_loaded = False
    
    @abstractmethod
    def load(self) -> Dict[str, Any]:
        """
        加载数据
        
        Returns:
            包含训练、验证、测试集的字典
        """
        pass
    
    @abstractmethod
    def get_texts(self, split: str = "train") -> List[str]:
        """
        获取文本数据
        
        Args:
            split: 数据集分割 ("train", "validation", "test")
            
        Returns:
            文本列表
        """
        pass
    
    @abstractmethod
    def get_labels(self, split: str = "train") -> List[Any]:
        """
        获取标签数据
        
        Args:
            split: 数据集分割 ("train", "validation", "test")
            
        Returns:
            标签列表
        """
        pass
    
    def get_label_names(self) -> List[str]:
        """
        获取标签名称
        
        Returns:
            标签名称列表
        """
        return getattr(self, 'label_names', None)
    
    def get_dataset_info(self) -> Dict[str, Any]:
        """
        获取数据集信息
        
        Returns:
            数据集统计信息
        """
        if not self._is_loaded:
            self.load()
            
        info = {
            'splits': list(self.data.keys()) if self.data else [],
            'total_samples': 0,
            'label_distribution': {}
        }
        
        if self.data:
            for split_name, split_data in self.data.items():
                if hasattr(split_data, '__len__'):
                    info[f'{split_name}_size'] = len(split_data)
                    info['total_samples'] += len(split_data)
        
        return info
    
    def validate_data(self) -> bool:
        """
        验证数据完整性
        
        Returns:
            数据是否有效
        """
        if not self._is_loaded:
            return False
            
        required_splits = ['train']
        for split in required_splits:
            if split not in self.data:
                return False
                
            try:
                texts = self.get_texts(split)
                labels = self.get_labels(split)
                
                if len(texts) != len(labels):
                    return False
                    
                if len(texts) == 0:
                    return False
                    
            except Exception:
                return False
                
        return True


class BaseDataPreprocessor(ABC):
    """数据预处理器基类"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化预处理器
        
        Args:
            config: 配置字典
        """
        self.config = config
    
    @abstractmethod
    def preprocess_texts(self, texts: List[str]) -> List[str]:
        """
        预处理文本数据
        
        Args:
            texts: 原始文本列表
            
        Returns:
            预处理后的文本列表
        """
        pass
    
    @abstractmethod
    def preprocess_labels(self, labels: List[Any]) -> List[Any]:
        """
        预处理标签数据
        
        Args:
            labels: 原始标签列表
            
        Returns:
            预处理后的标签列表
        """
        pass 