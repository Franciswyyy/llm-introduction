#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
配置管理演示
展示项目中统一配置管理的使用方法
演示YAML配置文件与Python配置的区别和使用场景
"""

import sys
from pathlib import Path

# 添加项目根目录到路径
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.append(str(PROJECT_ROOT))

from utils import config, get_config, load_task_config, PROJECT_ROOT


def demo_python_config():
    """演示Python配置的使用"""
    print("🐍 Python配置管理演示")
    print("=" * 50)
    
    # 1. 获取项目路径
    print("📁 项目路径信息:")
    print(f"  项目根目录: {config.PROJECT_ROOT}")
    print(f"  数据集目录: {config.DATASETS_DIR}")
    print(f"  模型目录: {config.PRETRAINED_MODELS_DIR}")
    print(f"  结果目录: {config.RESULTS_DIR}")
    
    # 2. 获取默认配置
    print("\n⚙️ 默认配置信息:")
    data_config = get_config("data")
    models_config = get_config("models")
    print(f"  数据配置: {data_config}")
    print(f"  模型配置: {models_config}")
    
    # 3. 设备自动选择
    print(f"\n💻 推荐设备: {config.get_device()}")
    
    # 4. 创建目录
    print(f"\n📂 创建项目目录...")
    config.create_directories()


def demo_yaml_config():
    """演示YAML配置的使用"""
    print("\n📄 YAML配置管理演示") 
    print("=" * 50)
    
    try:
        # 加载YAML配置文件
        task_config = load_task_config("sentiment_analysis.yaml")
        
        print("📋 YAML配置内容:")
        print(f"  任务名称: {task_config.get('task', {}).get('name', 'N/A')}")
        print(f"  数据集: {task_config.get('data', {}).get('dataset_name', 'N/A')}")
        print(f"  嵌入模型: {task_config.get('embedding', {}).get('model_name', 'N/A')}")
        
        # 显示分类器配置
        classifiers = task_config.get('classifiers', [])
        print(f"  分类器数量: {len(classifiers)}")
        for i, clf in enumerate(classifiers):
            print(f"    {i+1}. {clf.get('type', 'Unknown')} ({clf.get('name', 'unnamed')})")
        
        # 显示评估指标
        metrics = task_config.get('evaluation', {}).get('metrics', [])
        print(f"  评估指标: {', '.join(metrics)}")
        
    except FileNotFoundError:
        print("❌ 未找到sentiment_analysis.yaml配置文件")
    except Exception as e:
        print(f"❌ 加载YAML配置失败: {e}")


def compare_config_approaches():
    """对比不同配置方式的优缺点"""
    print("\n🔍 配置方式对比分析")
    print("=" * 50)
    
    print("📄 YAML配置文件的特点:")
    print("  ✅ 易读易写，结构清晰")
    print("  ✅ 支持注释，便于理解")
    print("  ✅ 非技术人员也能修改")
    print("  ✅ 版本控制友好")
    print("  ✅ 可以热更新（无需重启程序）")
    print("  ❌ 不支持复杂逻辑和计算")
    print("  ❌ 类型检查有限")
    
    print("\n🐍 Python配置的特点:")
    print("  ✅ 支持复杂逻辑和动态计算")
    print("  ✅ 完整的类型检查")
    print("  ✅ 可以调用函数和类")
    print("  ✅ IDE支持良好（自动补全、错误检查）")
    print("  ❌ 需要编程知识")
    print("  ❌ 修改后需要重启程序")


def usage_recommendations():
    """给出使用建议"""
    print("\n💡 使用建议")
    print("=" * 50)
    
    print("🎯 推荐的配置管理策略:")
    print("  1. 路径配置 → 使用Python配置模块 (utils.config)")
    print("     理由: 路径通常基于项目结构，很少改动")
    print()
    print("  2. 任务参数 → 使用YAML配置文件") 
    print("     理由: 实验参数经常调整，YAML更方便")
    print()
    print("  3. 模型默认配置 → Python配置")
    print("     理由: 相对稳定，需要类型检查")
    print()
    print("  4. 设备和环境配置 → Python自动检测")
    print("     理由: 需要动态判断，Python更灵活")
    
    print("\n🔧 实际应用:")
    print("  • 开发时: 主要修改YAML文件调整实验参数") 
    print("  • 生产环境: 使用Python配置确保稳定性")
    print("  • 团队协作: YAML文件便于非技术人员参与")


def show_migration_example():
    """展示如何从旧的分散配置迁移到新配置"""
    print("\n🔄 配置迁移示例")
    print("=" * 50)
    
    print("⚠️ 旧的方式 (不推荐):")
    print("```python")
    print("# 在每个文件中都要定义")
    print("PROJECT_ROOT = Path(__file__).parent.parent")
    print("DATASETS_DIR = PROJECT_ROOT / 'datasets'")
    print("MODELS_DIR = PROJECT_ROOT / 'models'")
    print("```")
    
    print("\n✅ 新的方式 (推荐):")
    print("```python")
    print("# 只需要一行导入")
    print("from utils import config, DATASETS_DIR, MODELS_DIR")
    print("# 或者")
    print("from utils.config import config")
    print("```")
    
    print("\n📈 迁移步骤:")
    print("  1. 替换路径导入: 从utils.config导入所需路径")
    print("  2. 删除重复定义: 移除各文件中的路径定义")
    print("  3. 统一配置加载: 使用load_task_config()加载YAML")
    print("  4. 测试验证: 确保所有功能正常工作")


def main():
    """主演示函数"""
    print("🎬 配置管理系统演示")
    print("解决项目中路径配置分散的问题")
    print("展示YAML配置与Python配置的最佳实践")
    
    # Python配置演示
    demo_python_config()
    
    # YAML配置演示
    demo_yaml_config()
    
    # 配置方式对比
    compare_config_approaches()
    
    # 使用建议
    usage_recommendations()
    
    # 迁移示例
    show_migration_example()
    
    print("\n🎉 演示完成!")
    print("建议查看 utils/config.py 了解更多配置选项")


if __name__ == "__main__":
    main() 