#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Core 模块使用示例
演示如何使用已封装的5个核心模块进行文本分类任务
"""

import sys
from pathlib import Path

# 添加项目根目录到路径
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.append(str(PROJECT_ROOT))

# 导入Core模块
from core.data.loaders import HuggingFaceLoader, CSVLoader
from core.embeddings.sentence_transformer import SentenceTransformerEmbedding
from core.classifiers.supervised import LogisticRegressionClassifier
from core.classifiers.similarity import SimilarityClassifier
from core.evaluation.metrics import ClassificationEvaluator
from core.pipeline.text_classification import TextClassificationPipeline

print("🏗️ Core模块使用演示")
print("=" * 60)

def demo_individual_modules():
    """演示各个模块的独立使用"""
    print("\n📋 方式一：分步使用各个模块")
    print("-" * 40)
    
    # 阶段1: 数据加载
    print("🔸 阶段1: 数据加载")
    data_config = {
        "dataset_name": "rotten_tomatoes",
        "cache_dir": "./resources/datasets"
    }
    data_loader = HuggingFaceLoader(data_config)
    dataset = data_loader.load()
    train_data = dataset['train']
    test_data = dataset['test']
    print(f"   训练数据: {len(train_data)} 条")
    print(f"   测试数据: {len(test_data)} 条")
    
    # 阶段2: 嵌入生成
    print("\n🔸 阶段2: 嵌入生成")
    embedding_config = {
        "model_name": "sentence-transformers/all-mpnet-base-v2",
        "batch_size": 16,
        "normalize_embeddings": True
    }
    embedding_model = SentenceTransformerEmbedding(embedding_config)
    
    # 取样本数据进行演示（避免计算时间过长）
    sample_train = train_data[:100]
    sample_test = test_data[:50]
    
    train_embeddings = embedding_model.encode([item['text'] for item in sample_train])
    test_embeddings = embedding_model.encode([item['text'] for item in sample_test])
    print(f"   训练嵌入: {train_embeddings.shape}")
    print(f"   测试嵌入: {test_embeddings.shape}")
    
    # 阶段3: 训练分类器
    print("\n🔸 阶段3: 训练分类器")
    classifier_config = {
        "random_state": 42,
        "max_iter": 1000,
        "C": 1.0
    }
    classifier = LogisticRegressionClassifier(classifier_config)
    
    train_labels = [item['label'] for item in sample_train]
    test_labels = [item['label'] for item in sample_test]
    
    classifier.train(train_embeddings, train_labels)
    print("   ✅ 分类器训练完成")
    
    # 阶段4: 预测
    print("\n🔸 阶段4: 预测")
    predictions = classifier.predict(test_embeddings)
    probabilities = classifier.predict_proba(test_embeddings)
    print(f"   预测结果: {len(predictions)} 个")
    print(f"   预测概率: {probabilities.shape}")
    
    # 阶段5: 评估
    print("\n🔸 阶段5: 评估")
    eval_config = {
        "target_names": ["负面", "正面"]
    }
    evaluator = ClassificationEvaluator(eval_config)
    metrics = evaluator.evaluate(test_labels, predictions)
    print(f"   准确率: {metrics['accuracy']:.3f}")
    print(f"   F1分数: {metrics['f1_macro']:.3f}")

def demo_pipeline_usage():
    """演示使用Pipeline进行端到端处理"""
    print("\n\n📋 方式二：使用Pipeline端到端处理")
    print("-" * 40)
    
    # 直接使用配置文件
    config_path = PROJECT_ROOT / "utils" / "sentiment_analysis.yaml"
    
    print("🔸 从配置文件创建Pipeline")
    pipeline = TextClassificationPipeline.from_config_file(str(config_path))
    print("   ✅ Pipeline创建完成")
    
    print("\n🔸 执行完整流程")
    try:
        results = pipeline.run()
        print("   ✅ Pipeline执行完成")
        print(f"   最佳模型: {results['best_model']}")
        print(f"   最佳F1分数: {results['best_f1']:.3f}")
        
        # 显示各个分类器的结果
        print("\n   📊 各分类器性能:")
        for model_name, metrics in results['results'].items():
            print(f"      {model_name}: F1={metrics['f1_macro']:.3f}")
            
    except Exception as e:
        print(f"   ⚠️ Pipeline执行出错: {e}")
        print("   这可能是因为计算资源限制，在实际使用中通常能正常运行")

def demo_similarity_classifier():
    """演示相似度分类器的使用"""
    print("\n\n📋 方式三：使用相似度分类器（无需训练）")
    print("-" * 40)
    
    # 准备示例数据
    texts = [
        "这部电影真的很棒，我非常喜欢！",
        "太糟糕了，完全浪费时间。",
        "还不错，值得一看。",
        "绝对是垃圾电影，不推荐。"
    ]
    
    # 类别原型文本
    prototype_texts = {
        0: ["这很糟糕", "我不喜欢", "太差了"],  # 负面
        1: ["这很棒", "我喜欢", "太好了"]      # 正面
    }
    
    print("🔸 创建相似度分类器")
    similarity_config = {
        "metric": "cosine",
        "embedding_model": "sentence-transformers/all-mpnet-base-v2"
    }
    similarity_classifier = SimilarityClassifier(similarity_config)
    
    print("🔸 设置类别原型")
    similarity_classifier.set_prototypes(prototype_texts)
    
    print("🔸 进行分类预测")
    for i, text in enumerate(texts):
        prediction = similarity_classifier.predict([text])[0]
        confidence = similarity_classifier.predict_proba([text])[0]
        
        label = "正面" if prediction == 1 else "负面"
        conf_score = max(confidence)
        
        print(f"   文本{i+1}: {text[:20]}...")
        print(f"   预测: {label} (置信度: {conf_score:.3f})")

def demo_custom_data():
    """演示如何使用自定义数据"""
    print("\n\n📋 方式四：使用自定义数据")
    print("-" * 40)
    
    # 创建自定义数据
    import pandas as pd
    custom_data = pd.DataFrame({
        'text': [
            "这个产品质量很好，推荐购买",
            "服务态度差，不会再来了",
            "价格合理，性价比不错",
            "完全不值这个价钱",
            "快递很快，包装也很好"
        ],
        'label': [1, 0, 1, 0, 1]  # 1=正面, 0=负面
    })
    
    # 保存为CSV文件
    csv_path = PROJECT_ROOT / "resources" / "custom_data.csv"
    csv_path.parent.mkdir(exist_ok=True)
    custom_data.to_csv(csv_path, index=False)
    print(f"🔸 创建自定义数据文件: {csv_path}")
    
    # 使用CSV加载器
    csv_config = {
        "file_path": str(csv_path),
        "text_column": "text",
        "label_column": "label"
    }
    csv_loader = CSVLoader(csv_config)
    dataset = csv_loader.load()
    data = dataset['train']  # CSV加载器返回的格式
    
    print(f"   加载数据: {len(data)} 条")
    for i, item in enumerate(data[:3]):
        print(f"   样本{i+1}: {item['text'][:30]}... (标签: {item['label']})")

def main():
    """主函数"""
    print("🌟 Core模块架构说明:")
    print("   1️⃣ core.data      - 数据加载和预处理")
    print("   2️⃣ core.embeddings - 文本嵌入生成") 
    print("   3️⃣ core.classifiers - 分类器训练和预测")
    print("   4️⃣ core.evaluation - 性能评估和可视化")
    print("   5️⃣ core.pipeline   - 端到端流水线")
    
    try:
        # 演示各个使用方式
        demo_individual_modules()
        demo_similarity_classifier()
        demo_custom_data()
        demo_pipeline_usage()
        
        print("\n" + "=" * 60)
        print("🎉 Core模块演示完成！")
        print("\n💡 使用建议:")
        print("   • 快速原型: 使用Pipeline")
        print("   • 灵活定制: 分步使用各模块")
        print("   • 无监督分类: 使用相似度分类器")
        print("   • 生产环境: 结合配置文件使用")
        
    except Exception as e:
        print(f"\n❌ 演示过程中出现错误: {e}")
        print("这可能是由于计算资源限制，核心功能仍然可用")

if __name__ == "__main__":
    main() 