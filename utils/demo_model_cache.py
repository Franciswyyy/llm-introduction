#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
模型缓存演示脚本
展示如何使用模型缓存机制，避免重复下载
"""

import sys
from pathlib import Path

# 添加项目根目录到路径
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.append(str(PROJECT_ROOT))

from utils import (
    load_model_pipeline,
    get_sentiment_model, 
    list_cached_models,
    get_config
)


def show_model_cache_info():
    """显示模型缓存信息"""
    print("🗂️  模型缓存信息:")
    print("=" * 60)
    
    # 显示配置的模型缓存目录
    model_config = get_config('models')
    cache_dir = model_config['models_cache_dir']
    default_model = model_config['sentiment_model']
    
    print(f"📁 缓存目录: {cache_dir}")
    print(f"🤖 默认情感模型: {default_model}")
    print()
    
    # 列出已缓存的模型
    print("📋 已缓存的模型:")
    cached = list_cached_models()
    
    if not cached:
        print("   💡 没有缓存的模型，首次加载时会自动下载")
    print()


def test_model_loading():
    """测试模型加载和缓存"""
    print("🧪 测试模型加载和缓存:")
    print("=" * 60)
    
    model_name = "cardiffnlp/twitter-roberta-base-sentiment-latest"
    
    print(f"🚀 第一次加载模型: {model_name}")
    print("   (如果没有缓存，会自动下载)")
    print("-" * 40)
    
    try:
        # 第一次加载（可能需要下载）
        pipe1 = load_model_pipeline(model_name)
        print("✅ 第一次加载成功!\n")
        
        # 测试模型
        test_text = "This movie is amazing!"
        result = pipe1(test_text)
        print(f"🧪 测试预测:")
        print(f"   输入: {test_text}")
        print(f"   结果: {result}\n")
        
        print("🚀 第二次加载同一模型:")
        print("   (应该从缓存快速加载)")
        print("-" * 40)
        
        # 第二次加载（应该从缓存）
        pipe2 = load_model_pipeline(model_name)
        print("✅ 第二次加载成功 (从缓存)!\n")
        
        # 显示更新后的缓存列表
        print("📋 更新后的缓存列表:")
        list_cached_models()
        
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")


def test_different_devices():
    """测试不同设备的模型加载"""
    print("\n🔧 测试设备配置:")
    print("=" * 60)
    
    model_name = "cardiffnlp/twitter-roberta-base-sentiment-latest"
    
    devices_to_test = ["auto", "cpu"]  # 可以添加 "cuda", "mps" 等
    
    for device in devices_to_test:
        print(f"🎯 测试设备: {device}")
        try:
            pipe = get_sentiment_model(model_name, device=device)
            print(f"   ✅ {device} 设备加载成功")
            
            # 简单测试
            result = pipe("Great movie!")
            print(f"   🧪 测试结果: {result[0]['label']}")
            
        except Exception as e:
            print(f"   ❌ {device} 设备加载失败: {e}")
        print()


def test_config_based_loading():
    """测试基于配置的模型加载"""
    print("⚙️  测试配置化模型加载:")
    print("=" * 60)
    
    print("📖 从config.py读取默认模型...")
    try:
        # 不指定模型名称，从config读取
        pipe = get_sentiment_model()  # 使用默认配置
        print("✅ 配置化加载成功!")
        
        # 测试
        test_texts = [
            "I love this product!",
            "This is terrible.",
            "It's okay, not great."
        ]
        
        print("\n🧪 批量测试:")
        for text in test_texts:
            result = pipe(text)
            sentiment = result[0]['label']
            confidence = result[0]['score']
            print(f"   '{text}' → {sentiment} ({confidence:.3f})")
            
    except Exception as e:
        print(f"❌ 配置化加载失败: {e}")


def main():
    """主函数"""
    print("🎯 模型缓存演示")
    print("✨ 展示模型自动缓存和配置化加载")
    print()
    
    # 1. 显示缓存信息
    show_model_cache_info()
    
    # 2. 测试模型加载和缓存
    test_model_loading()
    
    # 3. 测试不同设备
    test_different_devices()
    
    # 4. 测试配置化加载
    test_config_based_loading()
    
    print("\n✅ 演示完成!")
    print("\n💡 关键特性:")
    print("   🔧 模型自动缓存，避免重复下载")
    print("   📊 支持配置化模型选择") 
    print("   🚀 自动设备选择和优化")
    print("   📁 模型缓存路径可配置")
    print("   ⚡ 类似数据集的简洁加载接口")


if __name__ == "__main__":
    main() 