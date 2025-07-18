#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
模型管理模块
负责下载、缓存和加载各种预训练模型
支持本地缓存，避免重复下载
"""

import os
from pathlib import Path
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification, pipeline
import warnings
warnings.filterwarnings('ignore')

# 项目本地路径配置 - 从config读取
def get_models_cache_dir():
    """从config获取模型缓存目录"""
    from .config import config
    return Path(config.get_config('models')['models_cache_dir'])

def setup_model_directories():
    """创建模型相关目录"""
    models_dir = get_models_cache_dir()
    models_dir.mkdir(exist_ok=True, parents=True)
    print(f"📁 模型目录结构已创建: {models_dir}")

def check_model_exists(model_name: str) -> bool:
    """
    检查模型是否已存在于本地
    
    Args:
        model_name: 模型名称（如 'cardiffnlp/twitter-roberta-base-sentiment-latest'）
    
    Returns:
        bool: 模型是否存在
    """
    # 将模型名转换为本地目录名
    safe_model_name = model_name.replace('/', '_')
    models_dir = get_models_cache_dir()
    model_path = models_dir / safe_model_name
    
    # 检查是否有config.json文件（表示模型完整）
    config_file = model_path / "config.json"
    return config_file.exists()

def download_model(model_name: str, model_type: str = "auto") -> str:
    """
    下载并缓存模型到本地
    
    Args:
        model_name: 模型名称
        model_type: 模型类型 ('auto', 'classification', 'embedding')
    
    Returns:
        str: 本地模型路径
    """
    setup_model_directories()
    safe_model_name = model_name.replace('/', '_')
    models_dir = get_models_cache_dir()
    local_model_path = models_dir / safe_model_name
    
    print(f"🔄 正在下载模型: {model_name}")
    print(f"📁 保存位置: {local_model_path}")
    
    try:
        # 下载tokenizer
        print("   📥 下载分词器...")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.save_pretrained(local_model_path)
        
        # 根据模型类型下载对应的模型
        print("   📥 下载模型...")
        if model_type == "classification":
            model = AutoModelForSequenceClassification.from_pretrained(model_name)
        else:
            model = AutoModel.from_pretrained(model_name)
        
        model.save_pretrained(local_model_path)
        
        print(f"✅ 模型下载成功: {local_model_path}")
        return str(local_model_path)
        
    except Exception as e:
        print(f"❌ 模型下载失败: {e}")
        # 如果下载失败，删除不完整的文件
        if local_model_path.exists():
            import shutil
            shutil.rmtree(local_model_path)
        raise

def get_model_path(model_name: str, model_type: str = "auto") -> str:
    """
    获取模型路径，如果不存在则下载
    
    Args:
        model_name: 模型名称
        model_type: 模型类型
    
    Returns:
        str: 模型路径（本地或远程）
    """
    print(f"🔍 检查模型状态: {model_name}")
    
    if check_model_exists(model_name):
        safe_model_name = model_name.replace('/', '_')
        models_dir = get_models_cache_dir()
        local_path = models_dir / safe_model_name
        print(f"✅ 发现本地模型缓存: {local_path}")
        return str(local_path)
    else:
        print("❌ 未发现本地模型缓存")
        print("🌐 首次下载模型到本地")
        return download_model(model_name, model_type)

def get_sentiment_model(model_name: str = None, use_local: bool = True, device: str = "auto"):
    """
    获取情感分析模型 (支持配置化)
    
    Args:
        model_name: 模型名称，为None时从config读取
        use_local: 是否优先使用本地缓存
        device: 设备选择 ("auto", "cpu", "cuda", "mps")
    
    Returns:
        pipeline: 情感分析pipeline
    """
    # 如果没有指定模型名称，从config读取
    if model_name is None:
        from .config import config
        model_name = config.get_config('models')['sentiment_model']
    
    print(f"🤖 加载情感分析模型: {model_name}")
    
    if use_local:
        model_path = get_model_path(model_name, "classification")
    else:
        model_path = model_name
    
    # 自动设备选择
    if device == "auto":
        from .config import get_device
        device = get_device()
        print(f"🔧 自动选择设备: {device}")
    
    # 转换设备格式
    if device == "cpu":
        device_id = -1
    elif device == "mps":
        device_id = "mps"
    elif device == "cuda":
        device_id = 0
    else:
        device_id = -1
    
    print(f"🚀 创建Pipeline (设备: {device})...")
    try:
        pipe = pipeline(
            "sentiment-analysis",
            model=model_path,
            tokenizer=model_path,
            return_all_scores=True,
            device=device_id
        )
        print("✅ 模型加载成功")
        return pipe
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        # 如果本地模型失败，尝试直接从网络加载
        if use_local and model_path != model_name:
            print("🔄 尝试从网络直接加载...")
            return get_sentiment_model(model_name, use_local=False, device=device)
        raise

def load_model_pipeline(model_name: str, use_local: bool = True, device: str = "auto"):
    """
    通用的模型pipeline加载函数 (类似HuggingFaceLoader.load_dataset)
    
    Args:
        model_name: 模型名称
        use_local: 是否使用本地缓存
        device: 设备选择
        
    Returns:
        pipeline: 模型pipeline
    """
    return get_sentiment_model(model_name, use_local, device)

def get_embedding_model(model_name: str = None, use_local: bool = True, device: str = "auto"):
    """
    获取嵌入模型 (支持配置化)
    
    Args:
        model_name: 嵌入模型名称，为None时从config读取
        use_local: 是否优先使用本地缓存
        device: 设备选择 ("auto", "cpu", "cuda", "mps")
    
    Returns:
        SentenceTransformer: 嵌入模型
    """
    # 如果没有指定模型名称，从config读取
    if model_name is None:
        from .config import config
        model_name = config.get_config('models')['embedding_model']
    
    print(f"🤖 加载嵌入模型: {model_name}")
    
    if use_local:
        model_path = get_model_path(model_name, "embedding")
    else:
        model_path = model_name
    
    # 自动设备选择
    if device == "auto":
        from .config import get_device
        device = get_device()
        print(f"🔧 自动选择设备: {device}")
    
    print(f"🚀 创建SentenceTransformer (设备: {device})...")
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(model_path, device=device)
        print("✅ 嵌入模型加载成功")
        return model
    except Exception as e:
        print(f"❌ 嵌入模型加载失败: {e}")
        # 如果本地模型失败，尝试直接从网络加载
        if use_local and model_path != model_name:
            print("🔄 尝试从网络直接加载...")
            return get_embedding_model(model_name, use_local=False, device=device)
        raise

def load_embedding_model(model_name: str, use_local: bool = True, device: str = "auto"):
    """
    简洁的嵌入模型加载函数 (类似 load_model_pipeline)
    
    Args:
        model_name: 模型名称
        use_local: 是否使用本地缓存
        device: 设备选择
        
    Returns:
        SentenceTransformer: 嵌入模型
    """
    return get_embedding_model(model_name, use_local, device)

def list_cached_models():
    """列出所有已缓存的模型"""
    setup_model_directories()
    models_dir = get_models_cache_dir()
    
    print("📋 已缓存的模型:")
    if not models_dir.exists() or not list(models_dir.iterdir()):
        print("   (无)")
        return []
    
    cached_models = []
    for model_dir in models_dir.iterdir():
        if model_dir.is_dir() and (model_dir / "config.json").exists():
            # 将目录名转换回模型名
            model_name = model_dir.name.replace('_', '/')
            size = sum(f.stat().st_size for f in model_dir.rglob('*') if f.is_file())
            size_mb = size / (1024 * 1024)
            print(f"   📦 {model_name} ({size_mb:.1f}MB)")
            cached_models.append(model_name)
    
    return cached_models

def clear_model_cache(model_name: str = None):
    """
    清理模型缓存
    
    Args:
        model_name: 要清理的模型名称，None表示清理所有
    """
    import shutil
    
    if model_name:
        safe_model_name = model_name.replace('/', '_')
        model_path = PRETRAINED_MODELS_DIR / safe_model_name
        if model_path.exists():
            shutil.rmtree(model_path)
            print(f"🗑️ 已删除模型缓存: {model_name}")
        else:
            print(f"❌ 模型缓存不存在: {model_name}")
    else:
        if PRETRAINED_MODELS_DIR.exists():
            shutil.rmtree(PRETRAINED_MODELS_DIR)
            print("🗑️ 已清理所有模型缓存")
        else:
            print("❌ 没有模型缓存需要清理")

if __name__ == "__main__":
    # 测试模型管理功能
    print("🧪 测试模型管理器")
    print("=" * 50)
    
    # 列出当前缓存
    list_cached_models()
    
    # 测试获取情感分析模型
    try:
        model = get_sentiment_model()
        print("✅ 情感分析模型测试成功")
    except Exception as e:
        print(f"❌ 情感分析模型测试失败: {e}")
    
    # 再次列出缓存
    print("\n更新后的缓存:")
    list_cached_models() 