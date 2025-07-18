#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据管理器 - 专注数据集相关操作
遵循单一职责原则，只处理数据集的下载、缓存和加载
"""

import sys
from pathlib import Path
from typing import Dict, Any, Optional
from datasets import load_dataset
import warnings
warnings.filterwarnings('ignore')

# 项目路径配置
PROJECT_ROOT = Path(__file__).parent.parent.parent
RESOURCES_DIR = PROJECT_ROOT / "resources"
DATASETS_DIR = RESOURCES_DIR / "datasets"

class DataManager:
    """
    数据管理器类
    
    职责:
    - 数据集下载和缓存
    - 数据集加载和验证
    - 数据集信息查询
    """
    
    def __init__(self):
        self.cache_dir = DATASETS_DIR
        self._setup_directories()
    
    def _setup_directories(self):
        """创建必要的目录"""
        self.cache_dir.mkdir(exist_ok=True, parents=True)
    
    def check_dataset_exists(self, dataset_name: str = "rotten_tomatoes") -> bool:
        """
        检查数据集是否已缓存
        
        Args:
            dataset_name: 数据集名称
            
        Returns:
            bool: 数据集是否存在
        """
        dataset_path = self.cache_dir / dataset_name
        return dataset_path.exists() and any(dataset_path.iterdir())
    
    def download_dataset(self, dataset_name: str = "rotten_tomatoes") -> Dict[str, Any]:
        """
        下载数据集到本地缓存
        
        Args:
            dataset_name: 要下载的数据集名称
            
        Returns:
            Dict: 加载的数据集
        """
        print(f"🔄 正在下载数据集: {dataset_name}")
        print(f"📁 缓存位置: {self.cache_dir}")
        
        try:
            dataset = load_dataset(dataset_name, cache_dir=str(self.cache_dir))
            print(f"✅ 数据集下载成功")
            return dataset
        except Exception as e:
            print(f"❌ 数据集下载失败: {e}")
            raise
    
    def load_rotten_tomatoes(self) -> Dict[str, Any]:
        """
        加载Rotten Tomatoes数据集
        
        Returns:
            Dict: 包含train/validation/test的数据集
        """
        dataset_name = "rotten_tomatoes"
        print(f"🔍 检查数据集状态: {dataset_name}")
        
        if self.check_dataset_exists(dataset_name):
            print("✅ 发现本地数据集缓存")
            print("🚀 从本地缓存加载数据集")
        else:
            print("❌ 未发现本地数据集缓存")
            print("🌐 首次下载数据集到本地")
        
        # 加载数据集（自动处理缓存）
        dataset = load_dataset(dataset_name, cache_dir=str(self.cache_dir))
        
        # 显示数据集信息
        self._show_dataset_info(dataset)
        
        return dataset
    
    def load_custom_dataset(self, dataset_name: str) -> Dict[str, Any]:
        """
        加载自定义数据集
        
        Args:
            dataset_name: 数据集名称
            
        Returns:
            Dict: 加载的数据集
        """
        if self.check_dataset_exists(dataset_name):
            print(f"✅ 从缓存加载: {dataset_name}")
            return load_dataset(dataset_name, cache_dir=str(self.cache_dir))
        else:
            print(f"📥 下载数据集: {dataset_name}")
            return self.download_dataset(dataset_name)
    
    def get_dataset_info(self, dataset_name: str = "rotten_tomatoes") -> Dict[str, Any]:
        """
        获取数据集信息
        
        Args:
            dataset_name: 数据集名称
            
        Returns:
            Dict: 数据集信息
        """
        if not self.check_dataset_exists(dataset_name):
            return {"exists": False, "message": f"数据集 {dataset_name} 不存在"}
        
        try:
            dataset = load_dataset(dataset_name, cache_dir=str(self.cache_dir))
            info = {
                "exists": True,
                "splits": list(dataset.keys()),
                "features": dict(dataset[list(dataset.keys())[0]].features),
                "size": {split: len(data) for split, data in dataset.items()}
            }
            return info
        except Exception as e:
            return {"exists": False, "error": str(e)}
    
    def _show_dataset_info(self, dataset: Dict[str, Any]):
        """显示数据集信息"""
        print("✅ 数据集加载成功:")
        print(f"   📁 缓存位置: {self.cache_dir}")
        print("   📊 数据统计:")
        
        for split_name, split_data in dataset.items():
            split_size = len(split_data)
            print(f"      {split_name}: {split_size:,} 条")
    
    def clean_datasets_cache(self, dataset_name: Optional[str] = None):
        """
        清理数据集缓存
        
        Args:
            dataset_name: 要清理的数据集名称，None表示清理所有
        """
        import shutil
        
        if dataset_name:
            dataset_path = self.cache_dir / dataset_name
            if dataset_path.exists():
                shutil.rmtree(dataset_path)
                print(f"🗑️ 已删除数据集缓存: {dataset_name}")
            else:
                print(f"❌ 数据集缓存不存在: {dataset_name}")
        else:
            if self.cache_dir.exists():
                shutil.rmtree(self.cache_dir)
                self._setup_directories()
                print("🗑️ 已清理所有数据集缓存")
            else:
                print("❌ 没有数据集缓存需要清理")
    
    def list_cached_datasets(self) -> list:
        """
        列出所有已缓存的数据集
        
        Returns:
            list: 缓存的数据集列表
        """
        if not self.cache_dir.exists():
            return []
        
        cached_datasets = []
        for item in self.cache_dir.iterdir():
            if item.is_dir():
                cached_datasets.append(item.name)
        
        return cached_datasets 