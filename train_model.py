#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
电影评论情感分析模型训练脚本
使用 twitter-roberta-base-sentiment-latest 模型和 Rotten Tomatoes 数据集
所有文件本地管理，针对 Mac M2 优化
"""

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from datasets import load_dataset
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification,
    TrainingArguments, 
    Trainer,
    DataCollatorWithPadding
)
import warnings
warnings.filterwarnings('ignore')

# 项目本地路径配置
PROJECT_ROOT = Path(__file__).parent
MODELS_DIR = PROJECT_ROOT / "models"
DATASETS_DIR = PROJECT_ROOT / "datasets"
PRETRAINED_MODEL_DIR = MODELS_DIR / "twitter-roberta-base-sentiment-latest"
TRAINED_MODEL_DIR = PROJECT_ROOT / "trained_model"

def setup_directories():
    """创建必要的目录"""
    directories = [MODELS_DIR, DATASETS_DIR, TRAINED_MODEL_DIR]
    for directory in directories:
        directory.mkdir(exist_ok=True, parents=True)

def get_device():
    """获取最佳设备配置（Mac M2 MPS优先）"""
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("✅ 使用 Apple MPS 加速")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print("✅ 使用 CUDA 加速")
    else:
        device = torch.device("cpu")
        print("⚠️  使用 CPU")
    return device

def download_and_save_model():
    """下载并保存模型到本地"""
    model_name = "cardiffnlp/twitter-roberta-base-sentiment-latest"
    
    if PRETRAINED_MODEL_DIR.exists() and any(PRETRAINED_MODEL_DIR.iterdir()):
        print(f"✅ 模型已存在: {PRETRAINED_MODEL_DIR}")
        return str(PRETRAINED_MODEL_DIR)
    
    print(f"📥 下载模型: {model_name} (~500MB)")
    print("首次下载需要几分钟，请耐心等待...")
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name)
        
        PRETRAINED_MODEL_DIR.mkdir(exist_ok=True, parents=True)
        tokenizer.save_pretrained(str(PRETRAINED_MODEL_DIR))
        model.save_pretrained(str(PRETRAINED_MODEL_DIR))
        
        print(f"✅ 模型已下载到: {PRETRAINED_MODEL_DIR}")
        return str(PRETRAINED_MODEL_DIR)
        
    except Exception as e:
        print(f"❌ 下载失败: {e}")
        return model_name

def download_dataset():
    """下载数据集到本地"""
    print("📥 下载 Rotten Tomatoes 数据集 (~1MB)")
    
    os.environ['HF_DATASETS_CACHE'] = str(DATASETS_DIR)
    
    try:
        dataset = load_dataset("rotten_tomatoes", cache_dir=str(DATASETS_DIR))
        print(f"✅ 数据集已下载到: {DATASETS_DIR}")
        print(f"   训练集: {len(dataset['train']):,} 条")
        print(f"   验证集: {len(dataset['validation']):,} 条")
        print(f"   测试集: {len(dataset['test']):,} 条")
        return dataset
    except Exception as e:
        print(f"❌ 数据集下载失败: {e}")
        return None

class SentimentTrainer:
    def __init__(self, model_path):
        self.device = get_device()
        
        print(f"🔧 加载模型: {model_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_path,
            num_labels=2,  # 好评/差评
            ignore_mismatched_sizes=True
        )
        self.model.to(self.device)
        self.data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)
        
    def preprocess_data(self, dataset):
        """数据预处理"""
        print("🔄 预处理数据...")
        
        def tokenize_function(examples):
            return self.tokenizer(
                examples['text'], 
                truncation=True, 
                padding=False,
                max_length=512
            )
        
        tokenized_dataset = dataset.map(
            tokenize_function, 
            batched=True,
            remove_columns=['text']
        )
        return tokenized_dataset
    
    def compute_metrics(self, eval_pred):
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        return {'accuracy': accuracy_score(labels, predictions)}
    
    def train(self, tokenized_dataset):
        """训练模型"""
        print("🚀 开始训练...")
        
        # Mac M2 优化参数
        training_args = TrainingArguments(
            output_dir=str(TRAINED_MODEL_DIR),
            num_train_epochs=2,
            per_device_train_batch_size=8,
            per_device_eval_batch_size=8,
            warmup_steps=300,
            weight_decay=0.01,
            logging_steps=50,
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="accuracy",
            report_to=None,
            dataloader_pin_memory=False,  # Mac优化
            fp16=False,  # MPS不支持fp16
            dataloader_num_workers=0,  # Mac优化
        )
        
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=tokenized_dataset['train'],
            eval_dataset=tokenized_dataset['validation'],
            tokenizer=self.tokenizer,
            data_collator=self.data_collator,
            compute_metrics=self.compute_metrics,
        )
        
        print("⏱️  训练中...（Mac M2 预计 10-20 分钟）")
        trainer.train()
        
        trainer.save_model(str(TRAINED_MODEL_DIR))
        self.tokenizer.save_pretrained(str(TRAINED_MODEL_DIR))
        
        print(f"✅ 训练完成！模型已保存到: {TRAINED_MODEL_DIR}")
        return trainer
    
    def evaluate(self, trainer, tokenized_dataset):
        """评估模型"""
        print("📊 评估模型...")
        
        test_results = trainer.evaluate(eval_dataset=tokenized_dataset['test'])
        print(f"✅ 测试准确率: {test_results['eval_accuracy']:.4f}")
        
        predictions = trainer.predict(tokenized_dataset['test'])
        y_pred = np.argmax(predictions.predictions, axis=1)
        y_true = predictions.label_ids
        
        print("\n📈 分类报告:")
        target_names = ['差评 👎', '好评 👍']
        print(classification_report(y_true, y_pred, target_names=target_names))
        
        # 保存混淆矩阵
        cm = confusion_matrix(y_true, y_pred)
        
        # 配置中文字体支持
        plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'DejaVu Sans', 'SimHei', 'Helvetica']
        plt.rcParams['axes.unicode_minus'] = False
        
        plt.figure(figsize=(10, 8))
        ax = sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                        xticklabels=target_names, yticklabels=target_names,
                        cbar_kws={'shrink': 0.8}, annot_kws={'size': 16})
        plt.title('混淆矩阵 - 电影评论情感分析', fontsize=18, fontweight='bold', pad=20)
        plt.xlabel('预测标签', fontsize=16, labelpad=10)
        plt.ylabel('真实标签', fontsize=16, labelpad=10)
        ax.tick_params(axis='both', which='major', labelsize=14)
        plt.tight_layout()
        plt.savefig('confusion_matrix.png', dpi=400, bbox_inches='tight', facecolor='white', edgecolor='none')
        plt.show()
        print("📊 混淆矩阵已保存为 confusion_matrix.png")

def main():
    """主函数"""
    print("🎬 电影评论情感分析模型训练")
    print("=" * 50)
    
    # 1. 创建目录
    setup_directories()
    
    # 2. 下载模型
    model_path = download_and_save_model()
    
    # 3. 下载数据集
    dataset = download_dataset()
    if dataset is None:
        return
    
    # 4. 显示数据示例
    print("\n📝 数据示例:")
    for i in range(3):
        example = dataset['train'][i]
        label = "好评 👍" if example['label'] == 1 else "差评 👎"
        print(f"  {i+1}. {example['text'][:80]}...")
        print(f"     标签: {label}")
    
    # 5. 训练模型
    trainer = SentimentTrainer(model_path)
    tokenized_dataset = trainer.preprocess_data(dataset)
    model_trainer = trainer.train(tokenized_dataset)
    
    # 6. 评估模型
    trainer.evaluate(model_trainer, tokenized_dataset)
    
    print("\n🎉 完成！")
    print(f"📁 模型文件: {TRAINED_MODEL_DIR}")
    print("💡 运行 python predict.py 进行预测")

if __name__ == "__main__":
    main() 