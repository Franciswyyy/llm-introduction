#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Utils 快速参考手册
提供所有工具函数的快速查找和使用示例

🔍 快速查找技巧:
1. 使用 help(function_name) 查看详细文档
2. 使用 dir(module) 列出所有可用函数  
3. 使用 function_name? (在Jupyter中) 查看签名
4. 查看本文件获取使用示例

📦 主要模块分类:
- 数据管理: get_dataset, check_dataset_exists, clean_cache
- 模型管理: get_sentiment_model, get_embedding_model, list_cached_models
- 路径工具: PROJECT_ROOT, DATASETS_DIR, PRETRAINED_MODELS_DIR
"""

# 数据管理函数快速参考
DATA_FUNCTIONS = {
    "get_dataset": {
        "用途": "加载Rotten Tomatoes数据集",
        "返回": "包含train/validation/test的字典",
        "示例": "data = get_dataset()",
        "适用场景": "需要训练数据时"
    },
    
    "check_dataset_exists": {
        "用途": "检查数据集是否已缓存",
        "返回": "布尔值",
        "示例": "exists = check_dataset_exists()",
        "适用场景": "避免重复下载"
    },
    
    "clean_cache": {
        "用途": "清理数据集缓存",
        "返回": "无",
        "示例": "clean_cache()",
        "适用场景": "释放磁盘空间"
    }
}

# 模型管理函数快速参考
MODEL_FUNCTIONS = {
    "get_sentiment_model": {
        "用途": "获取预训练情感分析模型",
        "返回": "Pipeline对象",
        "示例": "model = get_sentiment_model()",
        "适用场景": "需要情感分析时"
    },
    
    "get_embedding_model": {
        "用途": "获取文本嵌入模型",
        "返回": "SentenceTransformer对象",
        "示例": "model = get_embedding_model()",
        "适用场景": "需要文本向量化时"
    },
    
    "list_cached_models": {
        "用途": "列出所有已缓存的模型",
        "返回": "模型名称列表",
        "示例": "models = list_cached_models()",
        "适用场景": "查看本地模型库存"
    },
    
    "clear_model_cache": {
        "用途": "清理模型缓存",
        "返回": "无",
        "示例": "clear_model_cache('model_name')",
        "适用场景": "释放磁盘空间"
    }
}

def show_function_help(category: str = "all"):
    """
    显示函数帮助信息
    
    Args:
        category: 函数类别 ("data", "model", "all")
    """
    print("🔍 Utils 函数快速参考")
    print("=" * 50)
    
    if category in ["data", "all"]:
        print("\n📊 数据管理函数:")
        for name, info in DATA_FUNCTIONS.items():
            print(f"\n🔸 {name}")
            print(f"   用途: {info['用途']}")
            print(f"   示例: {info['示例']}")
            print(f"   场景: {info['适用场景']}")
    
    if category in ["model", "all"]:
        print("\n🤖 模型管理函数:")
        for name, info in MODEL_FUNCTIONS.items():
            print(f"\n🔸 {name}")
            print(f"   用途: {info['用途']}")
            print(f"   示例: {info['示例']}")
            print(f"   场景: {info['适用场景']}")

def find_function_by_purpose(purpose_keyword: str):
    """
    根据用途关键词查找函数
    
    Args:
        purpose_keyword: 用途关键词 (如"情感", "数据", "缓存")
    """
    print(f"🔍 搜索包含 '{purpose_keyword}' 的函数:")
    print("-" * 40)
    
    found = False
    all_functions = {**DATA_FUNCTIONS, **MODEL_FUNCTIONS}
    
    for name, info in all_functions.items():
        if purpose_keyword in info['用途'] or purpose_keyword in info['适用场景']:
            print(f"✅ {name}: {info['用途']}")
            print(f"   示例: {info['示例']}")
            found = True
    
    if not found:
        print("❌ 未找到相关函数")
        print("💡 尝试这些关键词: 数据, 模型, 情感, 嵌入, 缓存")

def usage_examples():
    """显示常见使用场景的完整示例"""
    print("📚 常见使用场景示例")
    print("=" * 50)
    
    scenarios = {
        "🎬 情感分析任务": [
            "from utils import get_dataset, get_sentiment_model",
            "data = get_dataset()",
            "model = get_sentiment_model()",
            "result = model('这部电影很棒！')"
        ],
        
        "📝 文本向量化": [
            "from utils import get_embedding_model", 
            "model = get_embedding_model()",
            "vectors = model.encode(['文本1', '文本2'])"
        ],
        
        "🧹 清理缓存": [
            "from utils import clean_cache, clear_model_cache",
            "clean_cache()  # 清理数据集缓存",
            "clear_model_cache()  # 清理所有模型缓存"
        ],
        
        "📊 查看资源状态": [
            "from utils import check_dataset_exists, list_cached_models",
            "print('数据集存在:', check_dataset_exists())",
            "print('缓存模型:', list_cached_models())"
        ]
    }
    
    for scenario, code_lines in scenarios.items():
        print(f"\n{scenario}:")
        for line in code_lines:
            print(f"  {line}")

if __name__ == "__main__":
    # 交互式帮助
    import sys
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "search":
            if len(sys.argv) > 2:
                find_function_by_purpose(sys.argv[2])
            else:
                print("用法: python quick_reference.py search <关键词>")
        elif sys.argv[1] in ["data", "model"]:
            show_function_help(sys.argv[1])
        else:
            print("用法: python quick_reference.py [data|model|search <关键词>]")
    else:
        # 默认显示所有帮助
        show_function_help()
        print("\n" + "=" * 50)
        usage_examples()
        
        print("\n💡 快速使用技巧:")
        print("  python utils/quick_reference.py data     # 只看数据函数")
        print("  python utils/quick_reference.py model    # 只看模型函数") 
        print("  python utils/quick_reference.py search 情感  # 搜索相关函数") 