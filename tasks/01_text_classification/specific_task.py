#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
配置化预训练模型评估脚本
支持通过配置文件指定数据集和模型参数
基于统一配置管理系统，提供灵活的实验配置
"""

import sys
from pathlib import Path
import argparse

# 添加项目根目录到路径
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.append(str(PROJECT_ROOT))

# 导入配置管理和数据加载
from utils import config, get_config, load_task_config
from core.data.loaders import HuggingFaceLoader
from transformers import pipeline
import numpy as np
from tqdm import tqdm
from transformers.pipelines.pt_utils import KeyDataset
from sklearn.metrics import classification_report


def load_data_with_config(data_config):
    """根据配置加载数据集"""
    print(f"📊 加载数据集配置:")
    print(f"   数据集: {data_config.get('dataset_name', 'rotten_tomatoes')}")
    print(f"   加载器类型: {data_config.get('loader_type', 'HuggingFaceLoader')}")
    
    loader_type = data_config.get('loader_type', 'HuggingFaceLoader')
    
    if loader_type == 'HuggingFaceLoader':
        # 使用结构化的数据加载器
        loader = HuggingFaceLoader(data_config)
        dataset = loader.load()
        return dataset
    else:
        # 回退到原有方式
        from utils import get_dataset
        return get_dataset()


def load_model_with_config(model_config):
    """根据配置加载模型"""
    model_path = model_config.get('sentiment_model', 'cardiffnlp/twitter-roberta-base-sentiment-latest')
    device = model_config.get('device', 'auto')
    
    print(f"🤖 加载模型配置:")
    print(f"   模型路径: {model_path}")
    print(f"   设备: {device}")
    
    # 自动选择设备
    if device == 'auto':
        from utils import get_device
        device = get_device()
        print(f"   自动选择设备: {device}")
    
    # 加载模型管道
    pipe = pipeline(
        model=model_path,
        tokenizer=model_path,
        return_all_scores=True,
        device=device if device != 'cpu' else -1  # transformers期望-1表示CPU
    )
    
    return pipe


def run_evaluation(dataset, model_pipe, task_config):
    """运行模型评估"""
    print(f"🔄 开始评估...")
    
    # 获取测试数据
    test_data = dataset["test"]
    print(f"   测试样本数: {len(test_data):,}")
    
    # 批量预测
    y_pred = []
    for output in tqdm(model_pipe(KeyDataset(test_data, "text")), 
                      total=len(test_data), 
                      desc="预测进度"):
        # 解析情感分析结果 (negative, neutral, positive)
        negative_score = output[0]["score"]  # NEGATIVE
        positive_score = output[2]["score"]   # POSITIVE
        # 二分类：选择negative和positive中分数更高的
        assignment = np.argmax([negative_score, positive_score])
        y_pred.append(assignment)
    
    # 获取真实标签
    y_true = test_data["label"]
    
    # 评估性能
    evaluate_performance(y_true, y_pred, task_config)


def evaluate_performance(y_true, y_pred, task_config):
    """评估模型性能"""
    print(f"\n📈 评估结果:")
    
    # 获取标签名称
    label_names = ["Negative Review", "Positive Review"]
    
    # 生成分类报告
    performance = classification_report(
        y_true, y_pred,
        target_names=label_names,
        digits=4
    )
    
    print(performance)
    
    # 保存结果（如果配置要求）
    if task_config.get('pipeline', {}).get('save_results', False):
        output_dir = config.get_path('RESULTS_DIR')
        output_dir.mkdir(exist_ok=True)
        
        result_file = output_dir / f"{task_config['task']['name']}_results.txt"
        with open(result_file, 'w', encoding='utf-8') as f:
            f.write(f"任务: {task_config['task']['name']}\n")
            f.write(f"描述: {task_config['task']['description']}\n\n")
            f.write(performance)
        
        print(f"📁 结果已保存到: {result_file}")


def main(config_file: str = "config.yaml"):
    """主函数"""
    
    print(f"🚀 启动配置化文本分类评估")
    print(f"📁 项目根目录: {PROJECT_ROOT}")
    print(f"⚙️  配置文件: {config_file}")
    
    try:
        # 1. 加载配置
        print(f"\n⚙️  加载配置...")
        task_config = load_task_config(config_file)
        
        data_config = task_config['data']
        model_config = task_config['models']
        
        # 2. 加载数据集
        print(f"\n📊 加载数据集...")
        dataset = load_data_with_config(data_config)
        if dataset is None:
            raise Exception("数据集加载失败")
        
        # 3. 加载模型
        print(f"\n🤖 加载模型...")
        model_pipe = load_model_with_config(model_config)
        
        # 4. 运行评估
        print(f"\n🔬 运行评估...")
        run_evaluation(dataset, model_pipe, task_config)
        
        print(f"\n✅ 评估完成!")
        
    except Exception as e:
        print(f"❌ 执行失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # 支持命令行参数指定配置文件
    parser = argparse.ArgumentParser(description='配置化文本分类评估')
    parser.add_argument('--config', '-c', 
                       default='config.yaml',
                       help='配置文件名称 (默认: config.yaml)')
    
    args = parser.parse_args()
    main(args.config) 