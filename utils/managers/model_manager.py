#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
模型管理器 - 专注模型相关操作
遵循单一职责原则，只处理模型的下载、缓存和加载
"""

from pathlib import Path
from typing import Dict, Any, Optional, Union
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification, pipeline
import warnings
warnings.filterwarnings('ignore')

# 项目路径配置
PROJECT_ROOT = Path(__file__).parent.parent.parent
RESOURCES_DIR = PROJECT_ROOT / "resources"
PRETRAINED_MODELS_DIR = RESOURCES_DIR / "pretrained_models"

class ModelManager:
    """
    模型管理器类
    
    职责:
    - 模型下载和缓存
    - 模型加载和初始化
    - 模型配置管理
    """
    
    def __init__(self):
        self.models_dir = PRETRAINED_MODELS_DIR
        self._setup_directories()
        self._loaded_models = {}  # 模型缓存
    
    def _setup_directories(self):
        """创建必要的目录"""
        self.models_dir.mkdir(exist_ok=True, parents=True)
    
    def _get_safe_model_name(self, model_name: str) -> str:
        """将模型名转换为安全的文件名"""
        return model_name.replace('/', '_').replace('\\', '_')
    
    def check_model_exists(self, model_name: str) -> bool:
        """
        检查模型是否已缓存
        
        Args:
            model_name: 模型名称
            
        Returns:
            bool: 模型是否存在
        """
        safe_name = self._get_safe_model_name(model_name)
        model_path = self.models_dir / safe_name
        config_file = model_path / "config.json"
        return config_file.exists()
    
    def download_model(self, model_name: str, model_type: str = "auto") -> str:
        """
        下载模型到本地缓存
        
        Args:
            model_name: 模型名称
            model_type: 模型类型 ("auto", "classification", "embedding")
            
        Returns:
            str: 本地模型路径
        """
        safe_name = self._get_safe_model_name(model_name)
        local_path = self.models_dir / safe_name
        
        print(f"🔄 正在下载模型: {model_name}")
        print(f"📁 保存位置: {local_path}")
        
        try:
            # 下载tokenizer
            print("   📥 下载分词器...")
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            tokenizer.save_pretrained(local_path)
            
            # 下载模型
            print("   📥 下载模型...")
            if model_type == "classification":
                model = AutoModelForSequenceClassification.from_pretrained(model_name)
            else:
                model = AutoModel.from_pretrained(model_name)
            
            model.save_pretrained(local_path)
            print(f"✅ 模型下载成功: {local_path}")
            return str(local_path)
            
        except Exception as e:
            print(f"❌ 模型下载失败: {e}")
            # 清理不完整的下载
            if local_path.exists():
                import shutil
                shutil.rmtree(local_path)
            raise
    
    def get_model_path(self, model_name: str, model_type: str = "auto") -> str:
        """
        获取模型路径，如果不存在则下载
        
        Args:
            model_name: 模型名称
            model_type: 模型类型
            
        Returns:
            str: 模型路径
        """
        print(f"🔍 检查模型状态: {model_name}")
        
        if self.check_model_exists(model_name):
            safe_name = self._get_safe_model_name(model_name)
            local_path = self.models_dir / safe_name
            print(f"✅ 发现本地模型缓存: {local_path}")
            return str(local_path)
        else:
            print("❌ 未发现本地模型缓存")
            print("🌐 首次下载模型到本地")
            return self.download_model(model_name, model_type)
    
    def get_sentiment_pipeline(self, use_local: bool = True, model_name: str = "cardiffnlp/twitter-roberta-base-sentiment-latest"):
        """
        获取情感分析Pipeline
        
        Args:
            use_local: 是否优先使用本地缓存
            model_name: 模型名称
            
        Returns:
            Pipeline: 情感分析pipeline
        """
        if use_local:
            model_path = self.get_model_path(model_name, "classification")
        else:
            model_path = model_name
        
        print(f"🤖 创建情感分析Pipeline...")
        
        try:
            pipe = pipeline(
                "sentiment-analysis",
                model=model_path,
                tokenizer=model_path,
                return_all_scores=True,
                device=-1  # 使用CPU
            )
            print("✅ 情感分析模型加载成功")
            return pipe
        except Exception as e:
            print(f"❌ 模型加载失败: {e}")
            # 如果本地模型失败，尝试网络加载
            if use_local and model_path != model_name:
                print("🔄 尝试从网络直接加载...")
                return self.get_sentiment_pipeline(use_local=False, model_name=model_name)
            raise
    
    def get_embedding_model(self, model_name: str = "sentence-transformers/all-mpnet-base-v2", use_local: bool = True):
        """
        获取嵌入模型
        
        Args:
            model_name: 模型名称
            use_local: 是否优先使用本地缓存
            
        Returns:
            SentenceTransformer: 嵌入模型
        """
        # 检查内存缓存
        cache_key = f"embedding_{model_name}"
        if cache_key in self._loaded_models:
            print(f"🚀 从内存缓存加载嵌入模型: {model_name}")
            return self._loaded_models[cache_key]
        
        if use_local:
            model_path = self.get_model_path(model_name, "embedding")
        else:
            model_path = model_name
        
        print(f"🤖 加载嵌入模型: {model_name}")
        
        try:
            from sentence_transformers import SentenceTransformer
            model = SentenceTransformer(model_path)
            print("✅ 嵌入模型加载成功")
            
            # 缓存到内存
            self._loaded_models[cache_key] = model
            return model
        except Exception as e:
            print(f"❌ 嵌入模型加载失败: {e}")
            if use_local and model_path != model_name:
                print("🔄 尝试从网络直接加载...")
                return self.get_embedding_model(model_name, use_local=False)
            raise
    
    def list_cached_models(self) -> list:
        """
        列出所有已缓存的模型
        
        Returns:
            list: 缓存的模型信息列表
        """
        if not self.models_dir.exists():
            return []
        
        cached_models = []
        for model_dir in self.models_dir.iterdir():
            if model_dir.is_dir() and (model_dir / "config.json").exists():
                # 将目录名转换回模型名
                model_name = model_dir.name.replace('_', '/')
                
                # 计算大小
                size = sum(f.stat().st_size for f in model_dir.rglob('*') if f.is_file())
                size_mb = size / (1024 * 1024)
                
                cached_models.append({
                    "name": model_name,
                    "size_mb": round(size_mb, 1),
                    "path": str(model_dir)
                })
        
        return cached_models
    
    def clear_model_cache(self, model_name: Optional[str] = None):
        """
        清理模型缓存
        
        Args:
            model_name: 要清理的模型名称，None表示清理所有
        """
        import shutil
        
        if model_name:
            safe_name = self._get_safe_model_name(model_name)
            model_path = self.models_dir / safe_name
            if model_path.exists():
                shutil.rmtree(model_path)
                print(f"🗑️ 已删除模型缓存: {model_name}")
                
                # 清理内存缓存
                cache_key = f"embedding_{model_name}"
                if cache_key in self._loaded_models:
                    del self._loaded_models[cache_key]
            else:
                print(f"❌ 模型缓存不存在: {model_name}")
        else:
            if self.models_dir.exists():
                shutil.rmtree(self.models_dir)
                self._setup_directories()
                print("🗑️ 已清理所有模型缓存")
                
                # 清理内存缓存
                self._loaded_models.clear()
            else:
                print("❌ 没有模型缓存需要清理")
    
    def get_model_info(self, model_name: str) -> Dict[str, Any]:
        """
        获取模型信息
        
        Args:
            model_name: 模型名称
            
        Returns:
            Dict: 模型信息
        """
        info = {
            "name": model_name,
            "cached": self.check_model_exists(model_name),
            "in_memory": f"embedding_{model_name}" in self._loaded_models
        }
        
        if info["cached"]:
            safe_name = self._get_safe_model_name(model_name)
            model_path = self.models_dir / safe_name
            size = sum(f.stat().st_size for f in model_path.rglob('*') if f.is_file())
            info["size_mb"] = round(size / (1024 * 1024), 1)
            info["path"] = str(model_path)
        
        return info