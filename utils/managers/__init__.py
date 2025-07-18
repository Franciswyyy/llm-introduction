#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Managers 子包 - 按职责组织的管理器
遵循单一职责原则，每个管理器负责特定领域

🏗️ 架构优势:
- 单一职责: 每个类专注一个领域
- 易于扩展: 新功能添加新管理器
- 便于测试: 独立的功能模块
- 代码清晰: 相关功能聚合在一起
"""

from .data_manager import DataManager
from .model_manager import ModelManager
from .cache_manager import CacheManager

# 创建全局实例（单例模式）
data_mgr = DataManager()
model_mgr = ModelManager()
cache_mgr = CacheManager()

# 提供简化的接口函数
def get_dataset():
    """获取数据集 - 简化接口"""
    return data_mgr.load_rotten_tomatoes()

def get_sentiment_model():
    """获取情感分析模型 - 简化接口"""
    return model_mgr.get_sentiment_pipeline()

def get_embedding_model(model_name="sentence-transformers/all-mpnet-base-v2"):
    """获取嵌入模型 - 简化接口"""
    return model_mgr.get_embedding_model(model_name)

def list_cached_models():
    """列出缓存模型 - 简化接口"""
    return cache_mgr.list_models()

def clean_cache():
    """清理所有缓存 - 简化接口"""
    cache_mgr.clean_all()

__all__ = [
    # 管理器类
    'DataManager',
    'ModelManager', 
    'CacheManager',
    
    # 管理器实例
    'data_mgr',
    'model_mgr',
    'cache_mgr',
    
    # 简化接口
    'get_dataset',
    'get_sentiment_model',
    'get_embedding_model',
    'list_cached_models',
    'clean_cache'
] 