#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Utils包 - 项目工具模块
包含数据集管理、模型工具等功能

⚡ 新功能: 统一配置管理
现在可以通过 utils.config 访问所有路径和配置
"""

# 🆕 导入统一配置管理（优先推荐）
from .config import (
    config,
    PROJECT_ROOT,
    RESOURCES_DIR,
    DATASETS_DIR,
    PRETRAINED_MODELS_DIR,
    TRAINED_MODELS_DIR,
    MODELS_DIR,
    TRAINED_MODEL_DIR,
    get_config,
    load_task_config,
    setup_directories,
    get_device,
    get_dataset_info,
    list_available_datasets
)

# 导入主要功能
from .data_builder import (
    get_dataset,
    get_dataset_by_name,
    download_dataset_if_needed,
    check_dataset_exists,
    clean_cache,
    get_dataset_info
)

# 导入模型管理功能
from .model_manager import (
    get_sentiment_model,
    get_embedding_model,
    get_generation_model,
    get_model_path,
    load_model_pipeline,
    load_embedding_model,
    load_generation_pipeline,
    list_cached_models,
    clear_model_cache
)

# 定义包的公开接口
__all__ = [
    # 🆕 统一配置管理（推荐使用）
    'config',
    'get_config',
    'load_task_config',
    'setup_directories',
    'get_device',
    'get_dataset_info',
    'list_available_datasets',
    
    # 路径常量
    'PROJECT_ROOT',
    'RESOURCES_DIR',
    'DATASETS_DIR',
    'PRETRAINED_MODELS_DIR',
    'TRAINED_MODELS_DIR',
    'MODELS_DIR',
    'TRAINED_MODEL_DIR',
    
    # 数据管理
    'get_dataset',
    'get_dataset_by_name',
    'download_dataset_if_needed', 
    'check_dataset_exists',
    'clean_cache',
    'get_dataset_info',
    
    # 模型管理
    'get_sentiment_model',
    'get_embedding_model',
    'get_generation_model',
    'get_model_path',
    'load_model_pipeline',
    'load_embedding_model',
    'load_generation_pipeline',
    'list_cached_models',
    'clear_model_cache'
]

# 包信息
__version__ = "1.0.0"
__author__ = "AI Helper"
__description__ = "数据集和模型管理工具包"
