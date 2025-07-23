#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
生成模型文本分类任务
使用Text2Text生成模型(如FLAN-T5)通过prompt工程进行情感分析
将分类任务转换为文本生成任务，让模型生成"positive"或"negative"标签
"""

import sys
import numpy as np
from pathlib import Path
from tqdm import tqdm

# 添加项目根目录到Python路径
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.append(str(PROJECT_ROOT))

# 导入必要的库和模块
from transformers.pipelines.pt_utils import KeyDataset
from sklearn.metrics import classification_report
from core.data.loaders import HuggingFaceLoader
from utils import load_generation_pipeline


def preprocess_data_with_prompt(data, prompt="Is the following sentence positive or negative? "):
    """
    为数据添加prompt，将分类任务转换为生成任务
    
    Args:
        data: HuggingFace数据集
        prompt: 用于引导模型的提示文本
        
    Returns:
        处理后的数据集
    """
    print(f"🔧 添加prompt: '{prompt}'")
    
    # 为每个样本添加prompt
    def add_prompt(example):
        return {"t5_input": prompt + example['text']}
    
    processed_data = data.map(add_prompt)
    return processed_data


def parse_generated_text(generated_text):
    """
    解析生成的文本，提取分类结果
    
    Args:
        generated_text: 模型生成的文本
        
    Returns:
        int: 0表示负面，1表示正面
    """
    text = generated_text.lower().strip()
    
    # 处理各种可能的生成结果
    if "negative" in text:
        return 0
    elif "positive" in text:
        return 1
    elif "bad" in text or "poor" in text or "terrible" in text:
        return 0
    elif "good" in text or "great" in text or "excellent" in text:
        return 1
    else:
        # 如果无法明确判断，默认返回正面（可以根据需要调整）
        print(f"⚠️  无法解析的生成文本: '{generated_text}', 默认为正面")
        return 1


def evaluate_performance(y_true, y_pred):
    """
    评估分类性能
    
    Args:
        y_true: 真实标签
        y_pred: 预测标签
    """
    print("📊 生成模型分类性能评估:")
    print("=" * 50)
    
    # 详细分类报告
    report = classification_report(
        y_true, y_pred,
        target_names=["负面评价", "正面评价"],
        digits=4
    )
    print(report)


def main():
    try:
        # 1. 加载数据集
        print("📊 加载数据集...")
        data = HuggingFaceLoader.load_dataset("rotten_tomatoes")
        
        if data is None:
            raise Exception("数据集加载失败")
        
        # 2. 加载文本生成模型（使用配置化缓存系统）
        print("🤖 加载生成模型...")
        # 强制使用CPU避免MPS兼容性问题
        pipe = load_generation_pipeline("google/flan-t5-small", device="cpu")
        
        # 3. 预处理数据：添加prompt
        print("🔧 预处理数据...")
        prompt = "Classify this movie review as positive or negative: "
        processed_data = preprocess_data_with_prompt(data, prompt)
        
        # 4. 生成预测结果
        print("🔮 执行文本生成分类...")
        y_pred = []
        
        # 只使用测试集的一个子集进行快速测试（可以调整大小）
        test_size = min(100, len(processed_data["test"]))  # 限制测试样本数量以节省时间
        print(f"📝 处理 {test_size} 个测试样本...")
        
        # 使用进度条显示生成进度，并设置生成参数
        for i, output in enumerate(tqdm(
            pipe(KeyDataset(processed_data["test"], "t5_input"),
                 max_length=5,  # 限制生成长度 
                 temperature=0.1,  # 降低随机性
                 do_sample=True,  # 启用采样
                 num_return_sequences=1), 
            total=test_size,
            desc="生成中"
        )):
            if i >= test_size:
                break
                
            generated_text = output[0]["generated_text"]
            prediction = parse_generated_text(generated_text)
            y_pred.append(prediction)
        
        # 5. 评估性能
        evaluate_performance(data["test"]["label"][:test_size], y_pred)
        
        # 6. 显示一些示例生成结果
        print("\n💡 示例生成结果:")
        print("-" * 50)
        for i in range(min(5, len(y_pred))):
            original_text = data["test"]["text"][i][:100] + "..."
            prompt_text = processed_data["test"]["t5_input"][i]
            
            # 重新生成一个示例来显示
            sample_output = pipe(prompt_text, max_length=5, temperature=0.1, do_sample=True, num_return_sequences=1)
            generated = sample_output[0]["generated_text"]
            predicted_label = "正面" if y_pred[i] == 1 else "负面"
            true_label = "正面" if data["test"]["label"][i] == 1 else "负面"
            
            print(f"📝 原文: {original_text}")
            print(f"🤖 生成: {generated}")
            print(f"🎯 预测: {predicted_label} | 真实: {true_label}")
            print("-" * 30)
            
    except Exception as e:
        print(f"❌ 执行过程中出现错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 