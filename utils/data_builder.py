#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据集管理模块
负责下载、缓存和加载Rotten Tomatoes数据集
支持本地缓存，避免重复下载
"""

import os
from pathlib import Path
from datasets import load_dataset
import warnings
warnings.filterwarnings('ignore')

# 项目本地路径配置 - 指向项目根目录
PROJECT_ROOT = Path(__file__).parent.parent  # 从utils目录回到项目根目录
MODELS_DIR = PROJECT_ROOT / "models"
DATASETS_DIR = PROJECT_ROOT / "datasets"
PRETRAINED_MODEL_DIR = MODELS_DIR / "twitter-roberta-base-sentiment-latest"
TRAINED_MODEL_DIR = PROJECT_ROOT / "trained_model"

def setup_directories():
    """创建必要的目录"""
    directories = [MODELS_DIR, DATASETS_DIR, TRAINED_MODEL_DIR]
    for directory in directories:
        directory.mkdir(exist_ok=True, parents=True)
    print(f"📁 目录结构已创建: {PROJECT_ROOT}")

def check_dataset_exists(dataset_name="rotten_tomatoes"):
    """检查数据集是否已存在于本地"""
    # 从config读取缓存目录
    from .config import config
    cache_base_dir = Path(config.get_config('data')['cache_dir'])
    cache_dir = cache_base_dir / dataset_name
    
    # Hugging Face datasets的典型缓存结构
    if cache_dir.exists() and any(cache_dir.iterdir()):
        print(f"✅ 发现本地数据集缓存: {dataset_name}")
        return True
  
    print(f"❌ 未发现本地数据集缓存: {dataset_name}")
    return False

def download_dataset_if_needed(dataset_name):
    """智能下载：如果本地没有数据集则下载，否则从本地加载"""
    print(f"🔍 检查数据集状态: {dataset_name}")
    
    # 从config读取缓存目录
    from .config import config
    cache_dir = config.get_config('data')['cache_dir']
    cache_path = Path(cache_dir) / dataset_name
    
    
    try:
        # Hugging Face datasets的典型缓存结构
        if cache_path.exists() and any(cache_path.iterdir()):
            print(f"加载缓存数据集: {dataset_name}")
        dataset = load_dataset(dataset_name, cache_dir=str(cache_path))
        print(f"数据集加载成功,路径为: {cache_path}")

        return dataset
 
    except Exception as e:
        print(f"❌ 数据集加载失败: {e}")
        print("💡 请检查网络连接或数据集名称是否正确")
        return None

def get_dataset_by_name(dataset_name):
    """根据数据集名称获取数据集"""
    return download_dataset_if_needed(dataset_name)

def get_dataset():
    """获取数据集的主要接口函数（向后兼容）"""
    return download_dataset_if_needed("rotten_tomatoes")

def clean_cache():
    """清理数据集缓存（用于强制重新下载）"""
    import shutil
    
    if DATASETS_DIR.exists():
        print(f"🗑️  清理缓存目录: {DATASETS_DIR}")
        shutil.rmtree(DATASETS_DIR)
        print("✅ 缓存已清理，下次调用将重新下载")
    else:
        print("ℹ️  没有找到缓存目录")

def get_dataset_info():
    """获取数据集信息而不加载全部数据"""
    try:
        from datasets import get_dataset_infos
        infos = get_dataset_infos("rotten_tomatoes")
        return infos
    except Exception as e:
        print(f"⚠️  获取数据集信息失败: {e}")
        return None

# 为了向后兼容，保留原有函数名
def download_dataset():
    """下载数据集到本地（保持兼容性）"""
    return download_dataset_if_needed()

if __name__ == "__main__":
    # 测试模块
    print("🧪 测试数据集管理模块")
    dataset = get_dataset()
    if dataset:
        print("✅ 模块测试成功")
    else:
        print("❌ 模块测试失败")