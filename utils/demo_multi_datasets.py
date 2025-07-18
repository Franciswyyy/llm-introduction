#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
多数据集配置演示脚本
展示如何使用 config.py 中配置的多个数据集
"""

import sys
from pathlib import Path

# 添加项目根目录到路径
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.append(str(PROJECT_ROOT))

from utils import (
    get_dataset_info, 
    list_available_datasets, 
    get_config
)
from core.data.loaders import HuggingFaceLoader


def show_available_datasets():
    """显示所有可用的数据集"""
    print("🗂️  可用数据集列表:")
    print("=" * 60)
    
    datasets = list_available_datasets()
    for dataset_name in datasets:
        info = get_dataset_info(dataset_name)
        print(f"📊 {dataset_name}")
        print(f"   描述: {info['description']}")
        print(f"   任务类型: {info['task_type']}")
        print(f"   类别数: {info['num_classes']}")
        print(f"   标签: {info['labels']}")
        print()


def test_cache_mechanism():
    """测试缓存机制"""
    print("🔧 测试缓存机制:")
    print("=" * 60)
    
    # 显示配置的缓存目录
    data_config = get_config('data')
    cache_dir = data_config['cache_dir']
    print(f"📁 配置的缓存目录: {cache_dir}")
    print(f"🔢 默认批大小: {data_config['batch_size']}")
    print(f"📊 默认数据集: {data_config['dataset_name']}")
    print()
    
    # 测试不同数据集的缓存
    test_datasets = ["rotten_tomatoes", "imdb"]
    
    for dataset_name in test_datasets:
        print(f"🧪 测试数据集: {dataset_name}")
        
        # 检查缓存目录结构
        cache_path = Path(cache_dir) / dataset_name
        if cache_path.exists():
            print(f"   ✅ 缓存已存在: {cache_path}")
            file_count = len(list(cache_path.rglob("*")))
            print(f"   📁 缓存文件数: {file_count}")
        else:
            print(f"   ❌ 缓存不存在: {cache_path}")
            print(f"   💡 首次加载时会自动下载到此位置")
        print()


def load_different_datasets():
    """加载不同的数据集进行测试"""
    print("🚀 加载不同数据集测试:")
    print("=" * 60)
    
    # 测试数据集列表（从小到大）
    test_datasets = [
        "rotten_tomatoes",  # 较小，适合测试
        # "imdb",           # 较大，取消注释以测试
        # "ag_news",        # 多分类，取消注释以测试
    ]
    
    for dataset_name in test_datasets:
        print(f"\n📊 加载数据集: {dataset_name}")
        print("-" * 40)
        
        try:
            # 获取数据集信息
            info = get_dataset_info(dataset_name)
            print(f"   信息: {info['description']}")
            print(f"   标签: {info['labels']}")
            
            # 使用 HuggingFaceLoader 加载
            dataset = HuggingFaceLoader.load_dataset(dataset_name)
            
            if dataset:
                print(f"   ✅ 加载成功!")
                print(f"   分割: {list(dataset.keys())}")
                
                # 显示数据统计
                for split in dataset.keys():
                    size = len(dataset[split])
                    print(f"   {split}: {size:,} 条")
                
                # 显示样本
                if 'train' in dataset and len(dataset['train']) > 0:
                    sample = dataset['train'][0]
                    print(f"   样本文本: {sample['text'][:80]}...")
                    print(f"   样本标签: {sample['label']} -> {info['labels'][sample['label']]}")
            else:
                print(f"   ❌ 加载失败")
                
        except Exception as e:
            print(f"   ❌ 错误: {e}")


def main():
    """主函数"""
    print("🎯 多数据集配置演示")
    print("✨ 展示 config.py 中的多数据集支持功能")
    print()
    
    # 1. 显示可用数据集
    show_available_datasets()
    
    # 2. 测试缓存机制
    test_cache_mechanism()
    
    # 3. 加载不同数据集
    load_different_datasets()
    
    print("\n✅ 演示完成!")
    print("\n💡 关键特性:")
    print("   🔧 缓存目录可在 config.py 中配置")
    print("   📊 支持多个预定义数据集")
    print("   🚀 自动检测缓存，避免重复下载")
    print("   📁 每个数据集有独立的缓存目录")


if __name__ == "__main__":
    main() 