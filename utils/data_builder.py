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

def check_dataset_exists():
    """检查数据集是否已存在于本地"""
    # 检查datasets缓存目录
    cache_dir = DATASETS_DIR / "rotten_tomatoes"
    
    # Hugging Face datasets的典型缓存结构
    if cache_dir.exists() and any(cache_dir.iterdir()):
        print("✅ 发现本地数据集缓存")
        return True
    
    # 检查环境变量指定的缓存位置
    hf_cache = os.environ.get('HF_DATASETS_CACHE', '')
    if hf_cache and Path(hf_cache).exists():
        cache_path = Path(hf_cache) / "rotten_tomatoes"
        if cache_path.exists() and any(cache_path.iterdir()):
            print("✅ 发现HF缓存中的数据集")
            return True
    
    print("❌ 未发现本地数据集缓存")
    return False

def download_dataset_if_needed():
    """智能下载：如果本地没有数据集则下载，否则从本地加载"""
    print("🔍 检查数据集状态...")
    
    # 创建必要目录
    setup_directories()
    
    # 设置缓存目录
    os.environ['HF_DATASETS_CACHE'] = str(DATASETS_DIR)
    
    try:
        # 尝试加载数据集（会自动检查缓存）
        print("📥 加载 Rotten Tomatoes 数据集...")
        
        if check_dataset_exists():
            print("🚀 从本地缓存加载数据集")
        else:
            print("🌐 首次下载数据集到本地 (~1MB)")
        
        dataset = load_dataset("rotten_tomatoes", cache_dir=str(DATASETS_DIR))
        
        print(f"✅ 数据集加载成功:")
        print(f"   📁 缓存位置: {DATASETS_DIR}")
        print(f"   📊 数据统计:")
        print(f"      训练集: {len(dataset['train']):,} 条")
        print(f"      验证集: {len(dataset['validation']):,} 条")
        print(f"      测试集: {len(dataset['test']):,} 条")
        
        return dataset
        
    except Exception as e:
        print(f"❌ 数据集加载失败: {e}")
        print("💡 请检查网络连接或尝试手动删除缓存重新下载")
        return None

def get_dataset():
    """获取数据集的主要接口函数"""
    return download_dataset_if_needed()

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