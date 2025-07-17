#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Sentiment Analysis Task Runner
情感分析任务运行器 - 演示新架构的使用
"""

import sys
from pathlib import Path
import yaml

# 添加项目根目录到路径
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.append(str(PROJECT_ROOT))

from core.data.loaders import HuggingFaceLoader
from core.embeddings.sentence_transformer import SentenceTransformerEmbedding
from core.classifiers.supervised import LogisticRegressionClassifier
from core.classifiers.similarity import SimilarityClassifier
from core.evaluation.metrics import ClassificationEvaluator
from core.pipeline.text_classification import TextClassificationPipeline


def run_sentiment_analysis(config_path: str = None):
    """
    运行情感分析任务
    
    Args:
        config_path: 配置文件路径
    """
    print("🚀 开始运行情感分析任务 (新架构)")
    print("=" * 60)
    
    # 默认配置文件
    if config_path is None:
        config_path = PROJECT_ROOT / "utils" / "sentiment_analysis.yaml"
    
    # 加载配置
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # 创建组件
    print("🔧 正在初始化组件...")
    
    # 1. 数据加载器
    data_loader = HuggingFaceLoader({
        'dataset_name': config['data']['dataset_name'],
        'cache_dir': config['data']['cache_dir']
    })
    
    # 2. 嵌入模型
    embedding_model = SentenceTransformerEmbedding({
        'model_name': config['embedding']['model_name'],
        'device': config['embedding']['device'],
        'normalize_embeddings': config['embedding']['normalize_embeddings']
    })
    
    # 3. 评估器
    evaluator = ClassificationEvaluator({
        'metrics': config['evaluation']['metrics'],
        'average': config['evaluation']['average']
    })
    
    # 运行多个分类器
    results = {}
    
    for classifier_config in config['classifiers']:
        classifier_name = classifier_config['name']
        classifier_type = classifier_config['type']
        classifier_params = classifier_config.get('params', {})
        
        print(f"\n🎯 运行分类器: {classifier_name} ({classifier_type})")
        print("-" * 40)
        
        # 创建分类器
        if classifier_type == "LogisticRegressionClassifier":
            classifier = LogisticRegressionClassifier({'params': classifier_params})
        elif classifier_type == "SimilarityClassifier":
            classifier = SimilarityClassifier(classifier_params)
        else:
            print(f"❌ 不支持的分类器类型: {classifier_type}")
            continue
        
        # 创建流水线
        pipeline = TextClassificationPipeline(
            data_loader=data_loader,
            embedding_model=embedding_model,
            classifier=classifier,
            evaluator=evaluator,
            config=config['pipeline']
        )
        
        # 运行流水线
        try:
            result = pipeline.run(save_results=False)
            results[classifier_name] = result
            
            # 显示摘要
            print(f"\n📊 {classifier_name} 结果摘要:")
            print(pipeline.get_summary())
            
        except Exception as e:
            print(f"❌ {classifier_name} 运行失败: {e}")
            results[classifier_name] = {'error': str(e)}
    
    # 比较结果
    if len(results) > 1:
        print(f"\n🏆 分类器性能对比:")
        print("=" * 60)
        
        comparison_data = []
        for name, result in results.items():
            if 'error' not in result:
                eval_results = result['evaluation']
                comparison_data.append({
                    'classifier': name,
                    'accuracy': eval_results.get('accuracy', 0),
                    'precision': eval_results.get('precision', 0),
                    'recall': eval_results.get('recall', 0),
                    'f1': eval_results.get('f1', 0)
                })
        
        # 显示对比表格
        if comparison_data:
            print(f"{'分类器':<20} {'准确率':<10} {'精确率':<10} {'召回率':<10} {'F1分数':<10}")
            print("-" * 70)
            for row in comparison_data:
                print(f"{row['classifier']:<20} {row['accuracy']:<10.4f} {row['precision']:<10.4f} {row['recall']:<10.4f} {row['f1']:<10.4f}")
    
    print(f"\n🎉 情感分析任务完成！")
    return results


if __name__ == "__main__":
    run_sentiment_analysis() 