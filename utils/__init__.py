#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Utils包 - 项目工具模块
包含数据集管理、模型工具等功能
"""

# 导入主要功能
from .data_builder import (
    get_dataset,
    download_dataset_if_needed,
    check_dataset_exists,
    setup_directories,
    clean_cache,
    get_dataset_info,
    PROJECT_ROOT,
    DATASETS_DIR,
    MODELS_DIR,
    TRAINED_MODEL_DIR
)

# 定义包的公开接口
__all__ = [
    'get_dataset',
    'download_dataset_if_needed', 
    'check_dataset_exists',
    'setup_directories',
    'clean_cache',
    'get_dataset_info',
    'PROJECT_ROOT',
    'DATASETS_DIR',
    'MODELS_DIR',
    'TRAINED_MODEL_DIR'
]

# 包信息
__version__ = "1.0.0"
__author__ = "AI Helper"
__description__ = "数据集和模型管理工具包"
