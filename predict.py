#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
电影评论情感分析预测脚本
使用训练好的本地模型进行预测
"""

import torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import warnings
warnings.filterwarnings('ignore')

# 项目路径配置
PROJECT_ROOT = Path(__file__).parent
MODELS_DIR = PROJECT_ROOT / "models"
PRETRAINED_MODEL_DIR = MODELS_DIR / "twitter-roberta-base-sentiment-latest"
TRAINED_MODEL_DIR = PROJECT_ROOT / "trained_model"

class SentimentPredictor:
    def __init__(self):
        # 设置设备
        if torch.backends.mps.is_available():
            self.device = torch.device("mps")
            print("✅ 使用 MPS 加速预测")
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
            print("✅ 使用 CUDA 加速预测")
        else:
            self.device = torch.device("cpu")
            print("⚠️  使用 CPU 预测")
        
        # 选择模型路径（优先使用训练后的模型）
        if TRAINED_MODEL_DIR.exists() and any(TRAINED_MODEL_DIR.iterdir()):
            model_path = str(TRAINED_MODEL_DIR)
            print(f"🎯 使用训练后的模型: {model_path}")
        elif PRETRAINED_MODEL_DIR.exists() and any(PRETRAINED_MODEL_DIR.iterdir()):
            model_path = str(PRETRAINED_MODEL_DIR)
            print(f"🔧 使用预训练模型: {model_path}")
        else:
            print("❌ 未找到模型文件！")
            print("请先运行 python train_model.py 训练模型")
            raise FileNotFoundError("模型不存在")
        
        # 加载模型
        print("正在加载模型...")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
            self.model.to(self.device)
            self.model.eval()
            print("✅ 模型加载成功！")
        except Exception as e:
            print(f"❌ 模型加载失败: {e}")
            raise
    
    def predict(self, text):
        """预测单条文本的情感"""
        # 编码文本
        inputs = self.tokenizer(
            text, 
            return_tensors="pt", 
            truncation=True, 
            padding=True, 
            max_length=512
        ).to(self.device)
        
        # 预测
        with torch.no_grad():
            outputs = self.model(**inputs)
            probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
            predicted_label = torch.argmax(probabilities, dim=-1).item()
            confidence = probabilities[0][predicted_label].item()
        
        # 结果映射
        label_mapping = {0: "差评 👎", 1: "好评 👍"}
        
        return {
            "text": text,
            "prediction": label_mapping[predicted_label],
            "confidence": confidence,
            "scores": {
                "差评": probabilities[0][0].item(),
                "好评": probabilities[0][1].item()
            }
        }

def main():
    """主函数"""
    print("🎬 电影评论情感分析预测器")
    print("=" * 40)
    
    # 创建预测器
    try:
        predictor = SentimentPredictor()
    except:
        return
    
    # 示例预测
    examples = [
        "This movie is absolutely amazing! Great acting and wonderful story.",
        "Terrible waste of time. Boring plot and bad acting.",
        "Not bad, but could be better. Average movie overall.",
        "Outstanding performance! Highly recommended!",
        "这部电影太棒了！演技精湛，剧情引人入胜。"
    ]
    
    print("\n📝 示例预测:")
    print("-" * 40)
    
    for i, text in enumerate(examples, 1):
        result = predictor.predict(text)
        print(f"{i}. {text}")
        print(f"   预测: {result['prediction']}")
        print(f"   置信度: {result['confidence']:.3f}")
        print()
    
    # 交互式预测
    print("🎯 交互式预测 (输入 'quit' 退出):")
    print("-" * 40)
    
    while True:
        try:
            user_input = input("\n请输入电影评论: ").strip()
            
            if user_input.lower() in ['quit', 'exit', '退出', 'q']:
                print("👋 再见！")
                break
            
            if not user_input:
                print("⚠️  请输入有效的文本")
                continue
            
            result = predictor.predict(user_input)
            
            print(f"📊 预测结果:")
            print(f"   情感: {result['prediction']}")
            print(f"   置信度: {result['confidence']:.3f}")
            
            # 置信度提示
            if result['confidence'] > 0.8:
                print("   🎯 高置信度")
            elif result['confidence'] > 0.6:
                print("   ⚖️  中等置信度")
            else:
                print("   🤔 低置信度")
                
        except KeyboardInterrupt:
            print("\n👋 再见！")
            break
        except Exception as e:
            print(f"❌ 预测出错: {e}")

if __name__ == "__main__":
    main() 