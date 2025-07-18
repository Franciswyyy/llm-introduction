#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
统一配置管理模块
集中管理项目的所有路径配置、模型配置和其他设置
避免配置分散，提高维护性
"""

import os
import yaml
from pathlib import Path
from typing import Dict, Any, Optional


class ProjectConfig:
    """项目配置管理类"""
    
    def __init__(self):
        # 项目根目录 - 基于这个文件的位置计算
        self.PROJECT_ROOT = Path(__file__).parent.parent
        
        # 基础目录结构
        self._setup_base_paths()

        # 确保目录存在
        self.create_directories()
        
        # 默认配置
        self._setup_default_config()
    
    def _setup_base_paths(self):
        """设置基础路径"""
        # 资源目录 (新结构)
        self.RESOURCES_DIR = self.PROJECT_ROOT / "resources"
        self.DATASETS_DIR = self.RESOURCES_DIR / "datasets"
        self.PRETRAINED_MODELS_DIR = self.RESOURCES_DIR / "pretrained_models"
        self.TRAINED_MODELS_DIR = self.RESOURCES_DIR / "trained_models"
        
        # # 其他目录
        # self.UTILS_DIR = self.PROJECT_ROOT / "utils"
        # self.CORE_DIR = self.PROJECT_ROOT / "core"
        # self.TASKS_DIR = self.PROJECT_ROOT / "tasks"
        # self.EXAMPLES_DIR = self.PROJECT_ROOT / "examples"
        
        # # 缓存和临时目录
        # self.CACHE_DIR = self.PROJECT_ROOT / "cache"
        # self.RESULTS_DIR = self.PROJECT_ROOT / "results"
        # self.LOGS_DIR = self.PROJECT_ROOT / "logs"
    
    def _setup_default_config(self):
        """设置默认配置"""
        self.DEFAULT_CONFIG = {
            # 数据配置
            "data": {
                # "dataset_name": "rotten_tomatoes",  # 默认数据集
                "cache_dir": str(self.DATASETS_DIR),
                "batch_size": 32,
                
                # 支持的数据集配置
                "available_datasets": {
                    "rotten_tomatoes": {
                        "name": "rotten_tomatoes",
                        "description": "电影评论情感分析",
                        "task_type": "sentiment_analysis",
                        "num_classes": 2,
                        "labels": ["negative", "positive"]
                    },
                    "imdb": {
                        "name": "imdb", 
                        "description": "IMDB电影评论情感分析",
                        "task_type": "sentiment_analysis",
                        "num_classes": 2,
                        "labels": ["negative", "positive"]
                    },
                    "ag_news": {
                        "name": "ag_news",
                        "description": "AG News新闻分类",
                        "task_type": "text_classification", 
                        "num_classes": 4,
                        "labels": ["World", "Sports", "Business", "Technology"]
                    },
                    "sst2": {
                        "name": "sst2",
                        "description": "Stanford情感树库",
                        "task_type": "sentiment_analysis",
                        "num_classes": 2,
                        "labels": ["negative", "positive"]
                    }
                }
            },
            
            # 模型配置
            "models": {
                "sentiment_model": "cardiffnlp/twitter-roberta-base-sentiment-latest",
                "embedding_model": "sentence-transformers/all-mpnet-base-v2",
                "models_cache_dir": str(self.PRETRAINED_MODELS_DIR)
            },
            
            # 训练配置
            "training": {
                "output_dir": str(self.TRAINED_MODELS_DIR),
                "cache_dir": str(self.PROJECT_ROOT / "cache"),
                "results_dir": str(self.PROJECT_ROOT / "results"),
                "logs_dir": str(self.PROJECT_ROOT / "logs")
            },
            
            # 设备配置
            "device": {
                "auto_select": True,
                "preferred": ["mps", "cuda", "cpu"]
            }
        }
    
    def get_path(self, path_name: str) -> Path:
        """
        获取指定的路径
        
        Args:
            path_name: 路径名称
            
        Returns:
            Path: 路径对象
        """
        if hasattr(self, path_name.upper()):
            return getattr(self, path_name.upper())
        else:
            raise ValueError(f"未知的路径名称: {path_name}")
    
    def get_config(self, section: Optional[str] = None) -> Dict[str, Any]:
        """
        获取配置
        
        Args:
            section: 配置节名称，为None时返回全部配置
            
        Returns:
            Dict: 配置字典
        """
        if section is None:
            return self.DEFAULT_CONFIG
        elif section in self.DEFAULT_CONFIG:
            return self.DEFAULT_CONFIG[section]
        else:
            raise ValueError(f"未知的配置节: {section}")
    
    def load_yaml_config(self, config_file: str) -> Dict[str, Any]:
        """
        加载YAML配置文件
        
        Args:
            config_file: 配置文件路径
            
        Returns:
            Dict: 配置字典
        """
        config_path = self.UTILS_DIR / config_file
        
        if not config_path.exists():
            raise FileNotFoundError(f"配置文件不存在: {config_path}")
        
        with open(config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    
    def merge_config(self, yaml_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        合并YAML配置和默认配置
        
        Args:
            yaml_config: YAML配置字典
            
        Returns:
            Dict: 合并后的配置
        """
        merged_config = self.DEFAULT_CONFIG.copy()
        
        # 深度合并配置
        def deep_merge(base_dict: Dict, update_dict: Dict):
            for key, value in update_dict.items():
                if key in base_dict and isinstance(base_dict[key], dict) and isinstance(value, dict):
                    deep_merge(base_dict[key], value)
                else:
                    base_dict[key] = value
        
        deep_merge(merged_config, yaml_config)
        return merged_config
    
    def create_directories(self, directories: Optional[list] = None):
        """
        创建必要的目录
        
        Args:
            directories: 要创建的目录列表，为None时创建所有标准目录
        """
        if directories is None:
            directories = [
                self.RESOURCES_DIR,
                self.DATASETS_DIR,
                self.PRETRAINED_MODELS_DIR,
                self.TRAINED_MODELS_DIR,
                # 注释掉的目录不再自动创建
                # self.CACHE_DIR,
                # self.RESULTS_DIR,
                # self.LOGS_DIR
            ]
        
        for directory in directories:
            directory.mkdir(exist_ok=True, parents=True)
        
        print(f"📁 目录结构已创建: {self.PROJECT_ROOT}")
    
    def get_dataset_info(self, dataset_name: str) -> Dict[str, Any]:
        """
        获取指定数据集的配置信息
        
        Args:
            dataset_name: 数据集名称
            
        Returns:
            Dict: 数据集配置信息
        """
        available_datasets = self.DEFAULT_CONFIG["data"]["available_datasets"]
        if dataset_name in available_datasets:
            return available_datasets[dataset_name]
        else:
            # 返回默认配置
            return {
                "name": dataset_name,
                "description": f"未知数据集: {dataset_name}",
                "task_type": "text_classification",
                "num_classes": 2,
                "labels": ["label_0", "label_1"]
            }
    
    def list_available_datasets(self) -> list:
        """
        列出所有可用的数据集
        
        Returns:
            list: 数据集名称列表
        """
        return list(self.DEFAULT_CONFIG["data"]["available_datasets"].keys())
    
    def get_device(self) -> str:
        """
        自动选择最佳设备
        
        Returns:
            str: 设备名称
        """
        try:
            import torch
            
            if torch.backends.mps.is_available():
                return "mps"
            elif torch.cuda.is_available():
                return "cuda"
            else:
                return "cpu"
        except ImportError:
            return "cpu"
    
    def __str__(self) -> str:
        """返回配置概览"""
        return f"""
项目配置概览:
  项目根目录: {self.PROJECT_ROOT}
  资源目录: {self.RESOURCES_DIR}
  数据集目录: {self.DATASETS_DIR}
  模型目录: {self.PRETRAINED_MODELS_DIR}
  训练输出目录: {self.TRAINED_MODELS_DIR}
  缓存目录: {self.CACHE_DIR}
  结果目录: {self.RESULTS_DIR}
"""


# 创建全局配置实例
config = ProjectConfig()

# 导出常用路径和配置（保持向后兼容）
PROJECT_ROOT = config.PROJECT_ROOT
RESOURCES_DIR = config.RESOURCES_DIR
DATASETS_DIR = config.DATASETS_DIR
PRETRAINED_MODELS_DIR = config.PRETRAINED_MODELS_DIR
TRAINED_MODELS_DIR = config.TRAINED_MODELS_DIR

# 兼容旧路径名称
MODELS_DIR = config.PRETRAINED_MODELS_DIR  # 兼容性别名
TRAINED_MODEL_DIR = config.TRAINED_MODELS_DIR  # 兼容性别名

# 导出配置访问函数
def get_config(section: Optional[str] = None) -> Dict[str, Any]:
    """获取配置"""
    return config.get_config(section)

def load_task_config(config_file: str = "sentiment_analysis.yaml") -> Dict[str, Any]:
    """加载任务配置"""
    yaml_config = config.load_yaml_config(config_file)
    return config.merge_config(yaml_config)

def setup_directories():
    """创建项目目录"""
    config.create_directories()

def get_device() -> str:
    """获取最佳设备"""
    return config.get_device()

def get_dataset_info(dataset_name: str) -> Dict[str, Any]:
    """获取数据集配置信息"""
    return config.get_dataset_info(dataset_name)

def list_available_datasets() -> list:
    """列出可用数据集"""
    return config.list_available_datasets() 