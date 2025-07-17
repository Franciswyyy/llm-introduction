#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Text Classification Pipeline - 文本分类流水线
组合数据加载、嵌入生成、分类器训练和评估的完整流程
"""

from typing import Dict, List, Any, Optional, Tuple
import numpy as np
from pathlib import Path
import yaml
import json
from datetime import datetime

from ..data.base import BaseDataLoader
from ..embeddings.base import BaseEmbedding
from ..classifiers.base import BaseClassifier
from ..evaluation.metrics import BaseEvaluator


class TextClassificationPipeline:
    """文本分类流水线"""
    
    def __init__(self, 
                 data_loader: BaseDataLoader,
                 embedding_model: BaseEmbedding,
                 classifier: BaseClassifier,
                 evaluator: BaseEvaluator,
                 config: Optional[Dict[str, Any]] = None):
        """
        初始化流水线
        
        Args:
            data_loader: 数据加载器
            embedding_model: 嵌入模型
            classifier: 分类器
            evaluator: 评估器
            config: 流水线配置
        """
        self.data_loader = data_loader
        self.embedding_model = embedding_model
        self.classifier = classifier
        self.evaluator = evaluator
        self.config = config or {}
        
        # 流水线状态
        self.is_data_loaded = False
        self.is_embeddings_generated = False
        self.is_model_trained = False
        
        # 缓存数据
        self.train_embeddings = None
        self.test_embeddings = None
        self.train_labels = None
        self.test_labels = None
        self.class_names = None
        
        # 结果存储
        self.results = {}
        self.predictions = {}
    
    @classmethod
    def from_config(cls, config_path: str) -> 'TextClassificationPipeline':
        """
        从配置文件创建流水线
        
        Args:
            config_path: 配置文件路径
            
        Returns:
            流水线实例
        """
        with open(config_path, 'r', encoding='utf-8') as f:
            if config_path.endswith('.yaml') or config_path.endswith('.yml'):
                config = yaml.safe_load(f)
            else:
                config = json.load(f)
        
        # 这里需要根据配置创建各个组件
        # 实际实现需要工厂方法
        raise NotImplementedError("需要实现组件工厂方法")
    
    def load_data(self) -> Dict[str, Any]:
        """
        加载数据
        
        Returns:
            数据加载结果
        """
        print("🔄 正在加载数据...")
        
        # 加载数据
        data = self.data_loader.load()
        
        # 验证数据
        if not self.data_loader.validate_data():
            raise ValueError("数据验证失败")
        
        # 获取类别名称
        self.class_names = self.data_loader.get_label_names()
        
        self.is_data_loaded = True
        
        # 获取数据集信息
        data_info = self.data_loader.get_dataset_info()
        print(f"✅ 数据加载完成: {data_info}")
        
        return data_info
    
    def generate_embeddings(self, cache_embeddings: bool = True) -> Dict[str, Any]:
        """
        生成文本嵌入
        
        Args:
            cache_embeddings: 是否缓存嵌入向量
            
        Returns:
            嵌入生成结果
        """
        if not self.is_data_loaded:
            self.load_data()
        
        print("🔄 正在生成文本嵌入...")
        
        # 加载嵌入模型
        if not self.embedding_model._is_loaded:
            self.embedding_model.load_model()
        
        # 生成训练集嵌入
        train_texts = self.data_loader.get_texts("train")
        self.train_embeddings = self.embedding_model.encode(train_texts, show_progress_bar=True)
        self.train_labels = self.data_loader.get_labels("train")
        
        # 生成测试集嵌入
        test_texts = self.data_loader.get_texts("test")
        self.test_embeddings = self.embedding_model.encode(test_texts, show_progress_bar=True)
        self.test_labels = self.data_loader.get_labels("test")
        
        # 缓存嵌入向量
        if cache_embeddings:
            cache_dir = Path(self.config.get('cache_dir', './cache'))
            cache_dir.mkdir(parents=True, exist_ok=True)
            
            self.embedding_model.save_embeddings(
                self.train_embeddings, 
                cache_dir / 'train_embeddings.npy'
            )
            self.embedding_model.save_embeddings(
                self.test_embeddings,
                cache_dir / 'test_embeddings.npy'
            )
        
        self.is_embeddings_generated = True
        
        embedding_info = {
            'train_embeddings_shape': self.train_embeddings.shape,
            'test_embeddings_shape': self.test_embeddings.shape,
            'embedding_dimension': self.embedding_model.get_embedding_dimension()
        }
        
        print(f"✅ 嵌入生成完成: {embedding_info}")
        return embedding_info
    
    def train_classifier(self, **kwargs) -> Dict[str, Any]:
        """
        训练分类器
        
        Args:
            **kwargs: 训练参数
            
        Returns:
            训练结果
        """
        if not self.is_embeddings_generated:
            self.generate_embeddings()
        
        print("🎯 正在训练分类器...")
        
        # 编码标签
        encoded_labels = self.classifier.fit_transform_labels(self.train_labels)
        
        # 训练分类器
        self.classifier.train(self.train_embeddings, encoded_labels, **kwargs)
        
        self.is_model_trained = True
        
        training_info = {
            'classifier_type': self.classifier.__class__.__name__,
            'training_samples': len(self.train_embeddings),
            'num_classes': len(self.classifier.class_names) if self.classifier.class_names else None
        }
        
        print(f"✅ 分类器训练完成: {training_info}")
        return training_info
    
    def predict(self, return_probabilities: bool = False) -> Dict[str, Any]:
        """
        进行预测
        
        Args:
            return_probabilities: 是否返回预测概率
            
        Returns:
            预测结果
        """
        if not self.is_model_trained:
            self.train_classifier()
        
        print("🔮 正在进行预测...")
        
        # 预测
        predictions = self.classifier.predict(self.test_embeddings)
        
        # 预测概率
        probabilities = None
        if return_probabilities:
            probabilities = self.classifier.predict_proba(self.test_embeddings)
        
        # 转换标签
        predicted_labels = self.classifier.inverse_transform_labels(predictions)
        
        self.predictions = {
            'predictions': predictions,
            'predicted_labels': predicted_labels,
            'probabilities': probabilities,
            'true_labels': self.test_labels
        }
        
        print("✅ 预测完成")
        return self.predictions
    
    def evaluate(self) -> Dict[str, Any]:
        """
        评估模型性能
        
        Returns:
            评估结果
        """
        if 'predictions' not in self.predictions:
            self.predict(return_probabilities=True)
        
        print("📊 正在评估模型性能...")
        
        # 编码真实标签用于评估
        true_labels_encoded = self.classifier.fit_transform_labels(self.test_labels)
        
        # 评估
        evaluation_results = self.evaluator.evaluate(
            y_true=true_labels_encoded,
            y_pred=self.predictions['predictions'],
            y_proba=self.predictions['probabilities'],
            class_names=self.class_names
        )
        
        self.results['evaluation'] = evaluation_results
        
        print("✅ 评估完成")
        return evaluation_results
    
    def run(self, save_results: bool = True) -> Dict[str, Any]:
        """
        运行完整的流水线
        
        Args:
            save_results: 是否保存结果
            
        Returns:
            完整的运行结果
        """
        print("🚀 开始运行文本分类流水线")
        print("=" * 60)
        
        start_time = datetime.now()
        
        try:
            # 1. 加载数据
            data_info = self.load_data()
            
            # 2. 生成嵌入
            embedding_info = self.generate_embeddings()
            
            # 3. 训练分类器
            training_info = self.train_classifier()
            
            # 4. 预测
            predictions = self.predict(return_probabilities=True)
            
            # 5. 评估
            evaluation_results = self.evaluate()
            
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            # 汇总结果
            pipeline_results = {
                'config': self.config,
                'data_info': data_info,
                'embedding_info': embedding_info,
                'training_info': training_info,
                'predictions': predictions,
                'evaluation': evaluation_results,
                'pipeline_info': {
                    'start_time': start_time.isoformat(),
                    'end_time': end_time.isoformat(),
                    'duration_seconds': duration
                }
            }
            
            self.results.update(pipeline_results)
            
            # 保存结果
            if save_results:
                self.save_results()
            
            print(f"\n🎉 流水线运行完成! 总耗时: {duration:.2f}秒")
            
            return pipeline_results
            
        except Exception as e:
            print(f"❌ 流水线运行失败: {e}")
            raise
    
    def save_results(self, output_dir: Optional[str] = None) -> str:
        """
        保存运行结果
        
        Args:
            output_dir: 输出目录
            
        Returns:
            保存路径
        """
        if output_dir is None:
            output_dir = self.config.get('output_dir', './results')
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # 生成文件名
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = output_path / f"classification_results_{timestamp}.json"
        
        # 保存结果
        with open(results_file, 'w', encoding='utf-8') as f:
            # 处理numpy数组等不能直接序列化的对象
            serializable_results = self._make_serializable(self.results)
            json.dump(serializable_results, f, indent=2, ensure_ascii=False)
        
        print(f"💾 结果已保存至: {results_file}")
        return str(results_file)
    
    def _make_serializable(self, obj):
        """将结果转换为可序列化的格式"""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: self._make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_serializable(item) for item in obj]
        elif isinstance(obj, (np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.float64, np.float32)):
            return float(obj)
        else:
            return obj
    
    def get_summary(self) -> str:
        """
        获取流水线运行摘要
        
        Returns:
            格式化的摘要字符串
        """
        if 'evaluation' not in self.results:
            return "流水线尚未运行完成"
        
        return self.evaluator.generate_summary(self.results['evaluation']) 