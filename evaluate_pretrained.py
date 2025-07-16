#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
预训练模型评估脚本
直接使用 twitter-roberta-base-sentiment-latest 在 Rotten Tomatoes 数据集上评估性能
无需训练，快速验证模型效果 - 基于 Google Colab 代码优化
"""

import os
import numpy as np
from pathlib import Path
from tqdm import tqdm
from datasets import load_dataset
from transformers import pipeline
from transformers.pipelines.pt_utils import KeyDataset
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import warnings
warnings.filterwarnings('ignore')

# 项目路径配置
PROJECT_ROOT = Path(__file__).parent
DATASETS_DIR = PROJECT_ROOT / "datasets"

def get_device():
    """获取最佳设备"""
    if torch.backends.mps.is_available():
        return "mps"
    elif torch.cuda.is_available():
        return "cuda:0"
    else:
        return "cpu"

def download_dataset():
    """下载数据集到本地"""
    print("📥 下载 Rotten Tomatoes 数据集...")
    
    # 设置本地缓存
    os.environ['HF_DATASETS_CACHE'] = str(DATASETS_DIR)
    DATASETS_DIR.mkdir(exist_ok=True, parents=True)
    
    try:
        dataset = load_dataset("rotten_tomatoes", cache_dir=str(DATASETS_DIR))
        print(f"✅ 数据集信息:")
        print(f"   训练集: {len(dataset['train']):,} 条")
        print(f"   验证集: {len(dataset['validation']):,} 条")
        print(f"   测试集: {len(dataset['test']):,} 条")
        return dataset
    except Exception as e:
        print(f"❌ 数据集下载失败: {e}")
        return None

def create_pipeline():
    """创建推理pipeline"""
    model_path = "cardiffnlp/twitter-roberta-base-sentiment-latest"
    device = get_device()
    
    print(f"🔧 创建推理Pipeline:")
    print(f"   模型: {model_path}")
    print(f"   设备: {device}")
    
    try:
        pipe = pipeline(
            "sentiment-analysis",
            model=model_path,
            tokenizer=model_path,
            return_all_scores=True,
            device=device
        )
        print("✅ Pipeline创建成功")
        return pipe
    except Exception as e:
        print(f"❌ Pipeline创建失败: {e}")
        return None

def run_inference(pipe, dataset):
    """运行推理 - 基于你的Colab代码"""
    print("\n🚀 开始推理...")
    
    # 获取真实标签
    y_true = dataset["test"]["label"]
    
    # 批量推理
    y_pred = []
    print("推理进度:")
    
    for output in tqdm(pipe(KeyDataset(dataset["test"], "text")), total=len(dataset["test"])):
        # 获取负面和正面分数
        negative_score = output[0]["score"]  # LABEL_0 (负面)
        positive_score = output[2]["score"]  # LABEL_2 (正面)
        
        # 二分类：选择分数更高的类别
        assignment = np.argmax([negative_score, positive_score])
        y_pred.append(assignment)
    
    print("✅ 推理完成")
    return y_true, y_pred

def evaluate_performance(y_true, y_pred):
    """评估模型性能 - 基于你的代码但增强了功能"""
    print("\n📊 模型性能评估:")
    print("=" * 50)
    
    # 计算准确率
    accuracy = accuracy_score(y_true, y_pred)
    print(f"🎯 整体准确率: {accuracy:.4f}")
    
    # 详细分类报告
    print(f"\n📈 详细分类报告:")
    performance = classification_report(
        y_true, y_pred,
        target_names=["差评 👎", "好评 👍"],
        digits=4
    )
    print(performance)
    
    return accuracy

def plot_confusion_matrix(y_true, y_pred):
    """绘制混淆矩阵"""
    print("\n📊 生成混淆矩阵...")
    
    cm = confusion_matrix(y_true, y_pred)
    
    # 配置中文字体支持
    plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'DejaVu Sans', 'SimHei', 'Helvetica']
    plt.rcParams['axes.unicode_minus'] = False
    
    # 设置更大的图形和字体
    plt.rcParams.update({'font.size': 14})
    plt.figure(figsize=(10, 8))
    
    # 创建热力图，使用更大的字体
    ax = sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                     xticklabels=['差评 👎', '好评 👍'], 
                     yticklabels=['差评 👎', '好评 👍'],
                     cbar_kws={'shrink': 0.8},
                     annot_kws={'size': 16})
    
    # 设置标题和标签，使用更大的字体
    plt.title('预训练模型混淆矩阵 - Rotten Tomatoes', fontsize=18, fontweight='bold', pad=20)
    plt.xlabel('预测标签', fontsize=16, labelpad=10)
    plt.ylabel('真实标签', fontsize=16, labelpad=10)
    
    # 调整刻度标签大小
    ax.tick_params(axis='both', which='major', labelsize=14)
    
    # 确保布局合适
    plt.tight_layout()
    
    # 保存图片，提高质量
    save_path = PROJECT_ROOT / 'pretrained_confusion_matrix.png'
    plt.savefig(save_path, dpi=400, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.show()
    print(f"📊 混淆矩阵已保存: {save_path}")

def sample_predictions(pipe, dataset, num_samples=5):
    """显示一些预测样例"""
    print(f"\n📝 预测样例 (随机{num_samples}条):")
    print("-" * 60)
    
    # 随机选择一些样例
    import random
    test_data = dataset["test"]
    indices = random.sample(range(len(test_data)), num_samples)
    
    for i, idx in enumerate(indices, 1):
        text = test_data[idx]["text"]
        true_label = "好评 👍" if test_data[idx]["label"] == 1 else "差评 👎"
        
        # 预测
        result = pipe(text)[0]  # 取第一个结果
        
        # 解析预测结果
        negative_score = result[0]["score"]
        positive_score = result[2]["score"]
        pred_label = "好评 👍" if positive_score > negative_score else "差评 👎"
        confidence = max(negative_score, positive_score)
        
        print(f"{i}. 评论: {text[:80]}...")
        print(f"   真实: {true_label} | 预测: {pred_label} | 置信度: {confidence:.3f}")
        print()

def main():
    """主函数"""
    print("🎬 预训练模型快速评估")
    print("🚀 基于 twitter-roberta-base-sentiment-latest")
    print("=" * 60)
    
    # 1. 下载数据集
    dataset = download_dataset()
    if dataset is None:
        return
    
    # 2. 创建pipeline
    pipe = create_pipeline()
    if pipe is None:
        return
    
    # 3. 显示预测样例
    sample_predictions(pipe, dataset)
    
    # 4. 批量推理
    y_true, y_pred = run_inference(pipe, dataset)
    
    # 5. 评估性能
    accuracy = evaluate_performance(y_true, y_pred)
    
    # 6. 生成混淆矩阵
    plot_confusion_matrix(y_true, y_pred)
    
    print("\n" + "=" * 60)
    print("🎉 评估完成！")
    print(f"📈 预训练模型在Rotten Tomatoes测试集上的准确率: {accuracy:.1%}")
    print("💡 如需更好的性能，可以运行 python train_model.py 进行微调")

if __name__ == "__main__":
    main() 