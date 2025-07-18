#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试配置化数据集加载功能
演示如何使用不同的数据集配置
"""

import sys
from pathlib import Path

# 添加项目根目录到路径
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.append(str(PROJECT_ROOT))

from utils import get_dataset_by_name, load_task_config
from core.data.loaders import HuggingFaceLoader


def test_dataset_loading(dataset_name):
    """测试加载指定数据集"""
    print(f"\n🧪 测试加载数据集: {dataset_name}")
    print("=" * 50)
    
    try:
        # 方法1: 直接使用工具函数
        dataset = get_dataset_by_name(dataset_name)
        
        if dataset:
            print(f"✅ 成功加载数据集: {dataset_name}")
            print(f"   数据集分割: {list(dataset.keys())}")
            for split_name in dataset.keys():
                print(f"   {split_name}: {len(dataset[split_name]):,} 条")
            
            # 显示几个样本
            if 'train' in dataset:
                print(f"\n📝 样本预览:")
                for i in range(min(2, len(dataset['train']))):
                    text = dataset['train'][i]['text']
                    label = dataset['train'][i]['label']
                    print(f"   文本: {text[:100]}...")
                    print(f"   标签: {label}")
        else:
            print(f"❌ 无法加载数据集: {dataset_name}")
            
    except Exception as e:
        print(f"❌ 错误: {e}")


def test_config_based_loading(config_file):
    """测试基于配置文件的数据加载"""
    print(f"\n🧪 测试配置文件: {config_file}")
    print("=" * 50)
    
    try:
        # 加载配置
        config = load_task_config(config_file)
        data_config = config['data']
        
        print(f"📊 配置信息:")
        print(f"   数据集: {data_config.get('dataset_name')}")
        print(f"   加载器: {data_config.get('loader_type')}")
        
        # 使用HuggingFaceLoader
        loader = HuggingFaceLoader(data_config)
        dataset = loader.load()
        
        if dataset:
            print(f"✅ 成功通过配置加载数据集")
            print(f"   标签名称: {loader.get_label_names()}")
            
            # 测试获取文本和标签
            if 'train' in dataset:
                texts = loader.get_texts('train')
                labels = loader.get_labels('train')
                print(f"   训练集文本数: {len(texts)}")
                print(f"   训练集标签数: {len(labels)}")
        else:
            print(f"❌ 配置加载失败")
            
    except Exception as e:
        print(f"❌ 错误: {e}")


def main():
    """主函数 - 测试多个数据集"""
    print("🚀 开始测试配置化数据集加载功能")
    
    # 测试不同的数据集
    datasets_to_test = [
        "rotten_tomatoes",  # 默认数据集
        "imdb",            # 大型情感分析数据集
        # "ag_news",       # 新闻分类数据集 (取消注释以测试)
    ]
    
    for dataset_name in datasets_to_test:
        test_dataset_loading(dataset_name)
    
    # 测试配置文件
    print(f"\n🔧 测试配置文件加载")
    print("=" * 60)
    
    config_files = [
        "config.yaml",      # 默认配置
        "imdb_config.yaml", # IMDB配置
        # "ag_news_config.yaml",  # AG News配置 (取消注释以测试)
    ]
    
    for config_file in config_files:
        config_path = Path(__file__).parent / config_file
        if config_path.exists():
            test_config_based_loading(config_file)
        else:
            print(f"⚠️  配置文件不存在: {config_file}")
    
    print(f"\n✅ 测试完成!")


if __name__ == "__main__":
    main() 