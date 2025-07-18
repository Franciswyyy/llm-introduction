#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Data Loaders - 具体的数据加载器实现
"""

from typing import Dict, List, Any
import sys
from pathlib import Path

# 添加项目根目录到路径，以便导入utils
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.append(str(PROJECT_ROOT))

from .base import BaseDataLoader
from utils import get_dataset  # 使用原有的工具


class HuggingFaceLoader(BaseDataLoader):
    """Hugging Face数据集加载器"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.dataset_name = config.get('dataset_name', 'rotten_tomatoes')
        
        # 根据数据集名称设置标签名称
        self.label_names = self._get_label_names_for_dataset(self.dataset_name)
    
    @staticmethod
    def load_dataset(dataset_name: str) -> Dict[str, Any]:
        """
        静态方法：直接通过数据集名称加载数据集
        
        Args:
            dataset_name: 数据集名称，如 "rotten_tomatoes", "imdb" 等
            
        Returns:
            加载的数据集字典
        """
        from utils.data_builder import get_dataset_by_name
        return get_dataset_by_name(dataset_name)
    
    def load(self) -> Dict[str, Any]:
        """加载Hugging Face数据集"""
        print(f"🔄 正在加载数据集: {self.dataset_name}")
        
        # 使用配置中的数据集名称
        from utils.data_builder import get_dataset_by_name
        self.data = get_dataset_by_name(self.dataset_name)
        
        if self.data is None:
            raise ValueError(f"无法加载数据集: {self.dataset_name}")
        
        self._is_loaded = True
        return self.data
    
    def get_texts(self, split: str = "train") -> List[str]:
        """获取文本数据"""
        if not self._is_loaded:
            self.load()
        
        if split not in self.data:
            raise ValueError(f"数据集中不存在分割: {split}")
        
        return self.data[split]["text"]
    
    def get_labels(self, split: str = "train") -> List[Any]:
        """获取标签数据"""
        if not self._is_loaded:
            self.load()
        
        if split not in self.data:
            raise ValueError(f"数据集中不存在分割: {split}")
        
        return self.data[split]["label"]
    
    def _get_label_names_for_dataset(self, dataset_name: str) -> List[str]:
        """根据数据集名称获取标签名称"""
        # 常见情感分析数据集的标签映射
        label_mappings = {
            'rotten_tomatoes': ['negative', 'positive'],
            'imdb': ['negative', 'positive'],
            'sst2': ['negative', 'positive'],
            'amazon_polarity': ['negative', 'positive'],
            'yelp_polarity': ['negative', 'positive'],
            'ag_news': ['World', 'Sports', 'Business', 'Technology'],
            'dbpedia_14': ['Company', 'Educational Institution', 'Artist', 'Athlete', 
                          'Office Holder', 'Mean of Transportation', 'Building', 
                          'Natural Place', 'Village', 'Animal', 'Plant', 'Album', 
                          'Film', 'Written Work'],
        }
        
        return label_mappings.get(dataset_name, ['label_0', 'label_1'])  # 默认二分类标签
    
    def get_label_names(self) -> List[str]:
        """获取标签名称"""
        return self.label_names


class CSVLoader(BaseDataLoader):
    """CSV文件数据加载器"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.file_path = config.get('file_path')
        self.text_column = config.get('text_column', 'text')
        self.label_column = config.get('label_column', 'label')
        self.test_size = config.get('test_size', 0.2)
    
    def load(self) -> Dict[str, Any]:
        """加载CSV文件"""
        import pandas as pd
        from sklearn.model_selection import train_test_split
        
        print(f"🔄 正在加载CSV文件: {self.file_path}")
        
        df = pd.read_csv(self.file_path)
        
        # 分割数据
        train_df, test_df = train_test_split(df, test_size=self.test_size, random_state=42)
        
        self.data = {
            'train': train_df,
            'test': test_df
        }
        
        # 获取标签名称
        self.label_names = df[self.label_column].unique().tolist()
        
        self._is_loaded = True
        return self.data
    
    def get_texts(self, split: str = "train") -> List[str]:
        """获取文本数据"""
        if not self._is_loaded:
            self.load()
        
        return self.data[split][self.text_column].tolist()
    
    def get_labels(self, split: str = "train") -> List[Any]:
        """获取标签数据"""
        if not self._is_loaded:
            self.load()
        
        return self.data[split][self.label_column].tolist()


class JSONLoader(BaseDataLoader):
    """JSON文件数据加载器"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.file_path = config.get('file_path')
        self.text_field = config.get('text_field', 'text')
        self.label_field = config.get('label_field', 'label')
    
    def load(self) -> Dict[str, Any]:
        """加载JSON文件"""
        import json
        from sklearn.model_selection import train_test_split
        
        print(f"🔄 正在加载JSON文件: {self.file_path}")
        
        with open(self.file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # 假设数据格式为 [{"text": "...", "label": "..."}, ...]
        texts = [item[self.text_field] for item in data]
        labels = [item[self.label_field] for item in data]
        
        # 分割数据
        train_texts, test_texts, train_labels, test_labels = train_test_split(
            texts, labels, test_size=0.2, random_state=42
        )
        
        self.data = {
            'train': {'text': train_texts, 'label': train_labels},
            'test': {'text': test_texts, 'label': test_labels}
        }
        
        # 获取标签名称
        self.label_names = list(set(labels))
        
        self._is_loaded = True
        return self.data
    
    def get_texts(self, split: str = "train") -> List[str]:
        """获取文本数据"""
        if not self._is_loaded:
            self.load()
        
        return self.data[split]['text']
    
    def get_labels(self, split: str = "train") -> List[Any]:
        """获取标签数据"""
        if not self._is_loaded:
            self.load()
        
        return self.data[split]['label'] 