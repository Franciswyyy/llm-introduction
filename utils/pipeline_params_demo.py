#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pipeline 参数详解演示
详细解释 transformers.pipeline 中各个参数的含义和用法
"""

import sys
from pathlib import Path

# 添加项目根目录到路径
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.append(str(PROJECT_ROOT))

from transformers import pipeline


def demo_basic_pipeline():
    """演示基础 pipeline 参数"""
    print("🔧 基础 pipeline 参数演示")
    print("=" * 60)
    
    model_name = "cardiffnlp/twitter-roberta-base-sentiment-latest"
    test_text = "I love this amazing movie!"
    
    print(f"📝 测试文本: '{test_text}'\n")
    
    # 1. 最简单的用法
    print("1️⃣ 最简单的用法:")
    print("   pipeline('sentiment-analysis')")
    try:
        pipe1 = pipeline("sentiment-analysis")
        result1 = pipe1(test_text)
        print(f"   结果: {result1}")
    except Exception as e:
        print(f"   ❌ 错误: {e}")
    print()
    
    # 2. 指定模型
    print("2️⃣ 指定特定模型:")
    print(f"   pipeline('sentiment-analysis', model='{model_name}')")
    try:
        pipe2 = pipeline("sentiment-analysis", model=model_name)
        result2 = pipe2(test_text)
        print(f"   结果: {result2}")
    except Exception as e:
        print(f"   ❌ 错误: {e}")
    print()


def demo_return_all_scores():
    """演示 return_all_scores 参数的区别"""
    print("🎯 return_all_scores 参数对比")
    print("=" * 60)
    
    model_name = "cardiffnlp/twitter-roberta-base-sentiment-latest"
    test_text = "This movie is okay, not great."
    
    print(f"📝 测试文本: '{test_text}'\n")
    
    try:
        # return_all_scores=False (默认)
        print("1️⃣ return_all_scores=False (默认):")
        pipe_false = pipeline("sentiment-analysis", model=model_name, return_all_scores=False)
        result_false = pipe_false(test_text)
        print(f"   结果: {result_false}")
        print("   📊 只返回分数最高的类别\n")
        
        # return_all_scores=True
        print("2️⃣ return_all_scores=True:")
        pipe_true = pipeline("sentiment-analysis", model=model_name, return_all_scores=True)
        result_true = pipe_true(test_text)
        print(f"   结果: {result_true}")
        print("   📊 返回所有类别的分数")
        
        # 解释为什么我们使用 True
        print("\n💡 为什么我们的代码使用 return_all_scores=True?")
        print("   我们需要同时获取 NEGATIVE 和 POSITIVE 的分数")
        print("   然后比较两者，选择分数更高的作为最终预测")
        
        if result_true and len(result_true) >= 3:
            negative_score = result_true[0]['score']  # NEGATIVE
            positive_score = result_true[2]['score']   # POSITIVE
            prediction = "POSITIVE" if positive_score > negative_score else "NEGATIVE"
            print(f"   手动计算: NEGATIVE={negative_score:.4f}, POSITIVE={positive_score:.4f}")
            print(f"   最终预测: {prediction}")
        
    except Exception as e:
        print(f"❌ 错误: {e}")
    print()


def demo_device_parameters():
    """演示不同设备参数"""
    print("💻 设备参数演示")
    print("=" * 60)
    
    model_name = "cardiffnlp/twitter-roberta-base-sentiment-latest"
    test_text = "Great product!"
    
    devices_to_test = [
        (-1, "CPU"),
        (0, "GPU (如果可用)"),
        ("mps", "Apple Silicon GPU (如果可用)")
    ]
    
    for device_id, description in devices_to_test:
        print(f"🎯 测试设备: {description} (device={device_id})")
        try:
            pipe = pipeline(
                "sentiment-analysis", 
                model=model_name, 
                device=device_id,
                return_all_scores=True
            )
            
            import time
            start_time = time.time()
            result = pipe(test_text)
            end_time = time.time()
            
            print(f"   ✅ 成功! 用时: {(end_time - start_time)*1000:.1f}ms")
            print(f"   结果: {result[0]['label']} ({result[0]['score']:.4f})")
            
        except Exception as e:
            print(f"   ❌ 失败: {e}")
        print()


def demo_batch_processing():
    """演示批处理参数"""
    print("🔄 批处理演示")
    print("=" * 60)
    
    model_name = "cardiffnlp/twitter-roberta-base-sentiment-latest"
    
    # 多个文本的批处理
    test_texts = [
        "I love this product!",
        "This is terrible quality.",
        "It's okay, nothing special.",
        "Amazing experience!",
        "Worst purchase ever."
    ]
    
    print("📝 测试文本:")
    for i, text in enumerate(test_texts, 1):
        print(f"   {i}. '{text}'")
    print()
    
    try:
        pipe = pipeline(
            "sentiment-analysis", 
            model=model_name,
            return_all_scores=True,
            device=-1  # 使用CPU确保兼容性
        )
        
        print("🚀 批量处理中...")
        import time
        start_time = time.time()
        
        # 批量处理
        results = pipe(test_texts)
        
        end_time = time.time()
        print(f"⏱️  总用时: {(end_time - start_time):.2f}秒\n")
        
        print("📊 批量处理结果:")
        for i, (text, result) in enumerate(zip(test_texts, results), 1):
            # 从 return_all_scores=True 的结果中提取
            if isinstance(result, list) and len(result) > 0:
                top_result = max(result, key=lambda x: x['score'])
                label = top_result['label']
                score = top_result['score']
            else:
                label = result['label']
                score = result['score']
                
            print(f"   {i}. '{text[:30]}...' → {label} ({score:.4f})")
            
    except Exception as e:
        print(f"❌ 错误: {e}")
    print()


def demo_advanced_parameters():
    """演示高级参数"""
    print("🔬 高级参数演示")
    print("=" * 60)
    
    model_name = "cardiffnlp/twitter-roberta-base-sentiment-latest"
    
    print("📚 常用的高级参数:")
    print("   - max_length: 输入文本的最大长度")
    print("   - truncation: 是否截断过长文本")
    print("   - padding: 是否填充短文本")
    print("   - top_k: 返回前k个结果")
    print()
    
    try:
        # 创建带高级参数的pipeline
        pipe = pipeline(
            "sentiment-analysis",
            model=model_name,
            return_all_scores=True,
            device=-1,
            # 分词器参数
            max_length=512,      # 最大长度
            truncation=True,     # 截断长文本
            padding=True         # 填充短文本
        )
        
        # 测试长文本
        long_text = "This is a very long review. " * 20 + "Overall, I think it's great!"
        print(f"📝 长文本测试 (长度: {len(long_text)} 字符)")
        print(f"   内容: {long_text[:100]}...")
        
        result = pipe(long_text)
        top_result = max(result, key=lambda x: x['score'])
        print(f"   结果: {top_result['label']} ({top_result['score']:.4f})")
        print("   ✅ 长文本处理成功 (自动截断)")
        
    except Exception as e:
        print(f"❌ 错误: {e}")
    print()


def main():
    """主函数"""
    print("🎓 Pipeline 参数详解教程")
    print("🔍 深入理解 transformers.pipeline 的各个参数")
    print()
    
    # 1. 基础参数
    demo_basic_pipeline()
    
    # 2. return_all_scores 参数
    demo_return_all_scores()
    
    # 3. 设备参数
    demo_device_parameters()
    
    # 4. 批处理
    demo_batch_processing()
    
    # 5. 高级参数
    demo_advanced_parameters()
    
    print("✅ 教程完成!")
    print("\n📋 参数总结:")
    print("   🎯 任务类型: 'sentiment-analysis', 'text-classification' 等")
    print("   🤖 model: 模型名称或路径")
    print("   ✂️  tokenizer: 分词器名称或路径") 
    print("   📊 return_all_scores: True=所有分数, False=最高分")
    print("   💻 device: -1=CPU, 0=GPU, 'mps'=Apple GPU")
    print("   📏 max_length, truncation, padding: 文本处理参数")


if __name__ == "__main__":
    main() 