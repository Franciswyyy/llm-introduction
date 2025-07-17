#!/usr/bin/env python3
"""
HuggingFace Datasets 库学习示例
演示datasets库的常用操作和语法
"""

from datasets import Dataset, DatasetDict, load_dataset
import pandas as pd
import numpy as np

def basic_dataset_operations():
    """基础数据集操作示例"""
    print("=== 基础数据集操作 ===")
    
    # 1. 从字典创建数据集
    data = {
        'text': ['这是正面评论', '这是负面评论', '中性评论'],
        'label': [1, 0, 0.5],
        'id': [1, 2, 3]
    }
    dataset = Dataset.from_dict(data)
    print(f"从字典创建数据集: {dataset}")
    print(f"数据集大小: {len(dataset)}")
    print(f"特征: {dataset.features}")
    print(f"第一条数据: {dataset[0]}")
    
    # 2. 从pandas DataFrame创建
    df = pd.DataFrame(data)
    dataset_from_df = Dataset.from_pandas(df)
    print(f"\n从DataFrame创建: {dataset_from_df}")
    
    # 3. 数据集切片
    subset = dataset[:2]  # 前两条
    print(f"\n数据集切片: {subset}")

def load_online_datasets():
    """加载在线数据集示例"""
    print("\n=== 加载在线数据集 ===")
    
    try:
        # 加载小型数据集进行演示
        dataset = load_dataset("imdb", split="test[:100]")  # 只加载100条测试数据
        print(f"IMDB数据集: {dataset}")
        print(f"第一条数据: {dataset[0]}")
        
        # 指定缓存目录
        dataset_cached = load_dataset("imdb", split="test[:50]", 
                                    cache_dir="./resources/datasets")
        print(f"带缓存的数据集: {dataset_cached}")
        
    except Exception as e:
        print(f"网络加载失败: {e}")
        print("使用本地示例数据代替...")
        
        # 创建本地示例数据
        local_data = {
            'text': [
                'This movie is fantastic!',
                'Terrible movie, waste of time.',
                'An okay film, nothing special.',
                'Absolutely loved it!',
                'Not my cup of tea.'
            ],
            'label': [1, 0, 0, 1, 0]  # 1=positive, 0=negative
        }
        dataset = Dataset.from_dict(local_data)
        print(f"本地示例数据集: {dataset}")

def dataset_transformations():
    """数据集变换操作"""
    print("\n=== 数据集变换操作 ===")
    
    # 创建示例数据
    data = {
        'text': ['Good movie!', 'Bad film.', 'Okay story.', 'Great acting!'],
        'label': [1, 0, 0, 1]
    }
    dataset = Dataset.from_dict(data)
    
    # 1. map操作 - 添加文本长度
    def add_text_length(example):
        example['text_length'] = len(example['text'])
        return example
    
    dataset_with_length = dataset.map(add_text_length)
    print(f"添加文本长度后: {dataset_with_length[0]}")
    
    # 2. filter操作 - 过滤长文本
    def is_short_text(example):
        return len(example['text']) < 12
    
    short_texts = dataset.filter(is_short_text)
    print(f"过滤后的短文本数量: {len(short_texts)}")
    
    # 3. select操作 - 选择特定索引
    selected = dataset.select([0, 2])
    print(f"选择的数据: {selected['text']}")
    
    # 4. shuffle操作 - 打乱数据
    shuffled = dataset.shuffle(seed=42)
    print(f"打乱后的文本: {shuffled['text']}")
    
    # 5. train_test_split - 分割数据
    split_dataset = dataset.train_test_split(test_size=0.5, seed=42)
    print(f"训练集大小: {len(split_dataset['train'])}")
    print(f"测试集大小: {len(split_dataset['test'])}")

def batch_processing():
    """批处理操作示例"""
    print("\n=== 批处理操作 ===")
    
    data = {
        'numbers': list(range(10)),
        'texts': [f'text_{i}' for i in range(10)]
    }
    dataset = Dataset.from_dict(data)
    
    # 批处理函数
    def batch_square(batch):
        batch['squared'] = [x**2 for x in batch['numbers']]
        return batch
    
    # 应用批处理
    dataset_squared = dataset.map(batch_square, batched=True, batch_size=3)
    print(f"批处理结果: {dataset_squared[:5]}")

def dataset_formats():
    """数据集格式转换"""
    print("\n=== 数据集格式转换 ===")
    
    data = {
        'feature1': [1.0, 2.0, 3.0, 4.0],
        'feature2': [0.1, 0.2, 0.3, 0.4],
        'label': [0, 1, 0, 1]
    }
    dataset = Dataset.from_dict(data)
    
    # 设置格式为torch
    try:
        dataset.set_format(type='torch', columns=['feature1', 'feature2', 'label'])
        print(f"Torch格式: {type(dataset[0]['feature1'])}")
    except ImportError:
        print("PyTorch未安装，跳过torch格式转换")
    
    # 重置格式
    dataset.reset_format()
    
    # 转换为pandas
    df = dataset.to_pandas()
    print(f"转换为pandas: \n{df}")

def advanced_operations():
    """高级操作示例"""
    print("\n=== 高级操作 ===")
    
    # 1. 创建DatasetDict
    train_data = {'text': ['train1', 'train2'], 'label': [1, 0]}
    test_data = {'text': ['test1', 'test2'], 'label': [0, 1]}
    
    dataset_dict = DatasetDict({
        'train': Dataset.from_dict(train_data),
        'test': Dataset.from_dict(test_data)
    })
    print(f"DatasetDict: {dataset_dict}")
    
    # 2. 对所有分割应用相同操作
    def add_prefix(example):
        example['text'] = f"processed: {example['text']}"
        return example
    
    processed_dict = dataset_dict.map(add_prefix)
    print(f"处理后的数据: {processed_dict['train']['text']}")
    
    # 3. 合并数据集
    combined = dataset_dict['train'].concatenate(dataset_dict['test'])
    print(f"合并后的数据集: {combined}")

def save_and_load():
    """保存和加载数据集"""
    print("\n=== 保存和加载数据集 ===")
    
    data = {
        'text': ['sample1', 'sample2', 'sample3'],
        'label': [1, 0, 1]
    }
    dataset = Dataset.from_dict(data)
    
    # 保存为Arrow格式
    save_path = "./resources/datasets/sample_dataset"
    dataset.save_to_disk(save_path)
    print(f"数据集已保存到: {save_path}")
    
    # 加载数据集
    try:
        loaded_dataset = Dataset.load_from_disk(save_path)
        print(f"加载的数据集: {loaded_dataset}")
    except Exception as e:
        print(f"加载失败: {e}")

def main():
    """主函数 - 运行所有示例"""
    print("HuggingFace Datasets 库学习示例\n")
    
    basic_dataset_operations()
    load_online_datasets()
    dataset_transformations()
    batch_processing()
    dataset_formats()
    advanced_operations()
    save_and_load()
    
    print("\n=== 常用方法总结 ===")
    print("1. Dataset.from_dict() - 从字典创建")
    print("2. load_dataset() - 加载在线数据集")
    print("3. .map() - 元素级变换")
    print("4. .filter() - 过滤数据")
    print("5. .select() - 选择索引")
    print("6. .shuffle() - 打乱数据")
    print("7. .train_test_split() - 分割数据")
    print("8. .save_to_disk() / .load_from_disk() - 保存/加载")
    print("9. .set_format() - 设置输出格式")
    print("10. .concatenate() - 合并数据集")

if __name__ == "__main__":
    main() 