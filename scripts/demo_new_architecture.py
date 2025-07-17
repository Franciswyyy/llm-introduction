#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
New Architecture Demo - 新架构演示脚本
展示重构后的模块化文本分类框架
"""

import sys
from pathlib import Path

# 添加项目根目录到路径
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.append(str(PROJECT_ROOT))

from tasks.sentiment_analysis.run import run_sentiment_analysis


def main():
    """主演示函数"""
    print("🌟 LLM Introduction 项目 - 新架构演示")
    print("=" * 60)
    print()
    print("📋 新架构特点:")
    print("  ✅ 模块化设计 - 每个功能都有独立模块")
    print("  ✅ 可扩展性强 - 轻松添加新的数据源、模型、分类器")
    print("  ✅ 配置驱动 - 通过配置文件控制行为")
    print("  ✅ 接口统一 - 所有组件遵循统一接口")
    print("  ✅ 流水线化 - 自动化的端到端处理流程")
    print()
    
    try:
        # 运行情感分析任务
        results = run_sentiment_analysis()
        
        print("\n✨ 演示总结:")
        print("  📁 模块结构清晰，便于维护和扩展")
        print("  🔧 组件可插拔，支持多种算法组合")
        print("  📊 自动化评估，提供详细性能报告")
        print("  🎯 业务逻辑与核心功能分离")
        
    except Exception as e:
        print(f"❌ 演示运行失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 