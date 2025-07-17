#!/usr/bin/env python3
"""
HuggingFace Transformers 库学习示例
演示transformers库的常用操作和语法
"""

from transformers import (
    AutoTokenizer, AutoModel, AutoModelForSequenceClassification,
    pipeline, TrainingArguments, Trainer
)
import torch
import numpy as np

def basic_tokenizer_usage():
    """基础分词器使用"""
    print("=== 基础分词器使用 ===")
    
    # 加载预训练分词器
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    
    text = "Hello, how are you today?"
    
    # 1. 基本编码
    tokens = tokenizer(text)
    print(f"原文: {text}")
    print(f"编码结果: {tokens}")
    
    # 2. 解码
    decoded = tokenizer.decode(tokens['input_ids'])
    print(f"解码结果: {decoded}")
    
    # 3. 批量处理
    texts = ["First sentence.", "Second sentence.", "Third sentence."]
    batch_tokens = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
    print(f"批量编码: {batch_tokens}")
    
    # 4. 获取词汇表信息
    print(f"词汇表大小: {tokenizer.vocab_size}")
    print(f"特殊令牌: {tokenizer.special_tokens_map}")

def model_loading_and_inference():
    """模型加载和推理"""
    print("\n=== 模型加载和推理 ===")
    
    # 1. 加载预训练模型
    model_name = "distilbert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    
    text = "This is a sample text for encoding."
    
    # 2. 编码文本
    inputs = tokenizer(text, return_tensors="pt")
    print(f"输入张量形状: {inputs['input_ids'].shape}")
    
    # 3. 模型推理
    with torch.no_grad():
        outputs = model(**inputs)
    
    # 4. 获取输出
    last_hidden_states = outputs.last_hidden_state
    print(f"最后隐藏状态形状: {last_hidden_states.shape}")
    
    # 5. 池化操作 (平均池化)
    sentence_embedding = last_hidden_states.mean(dim=1)
    print(f"句子嵌入形状: {sentence_embedding.shape}")

def classification_model():
    """分类模型示例"""
    print("\n=== 分类模型示例 ===")
    
    # 加载分类模型
    model_name = "distilbert-base-uncased-finetuned-sst-2-english"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    
    texts = [
        "I love this movie!",
        "This film is terrible.",
        "It's an okay movie."
    ]
    
    for text in texts:
        # 编码
        inputs = tokenizer(text, return_tensors="pt")
        
        # 推理
        with torch.no_grad():
            outputs = model(**inputs)
        
        # 获取预测结果
        logits = outputs.logits
        predictions = torch.nn.functional.softmax(logits, dim=-1)
        predicted_class = torch.argmax(predictions, dim=-1)
        
        print(f"文本: {text}")
        print(f"预测概率: {predictions[0].tolist()}")
        print(f"预测类别: {predicted_class.item()}")
        print()

def pipeline_usage():
    """Pipeline 使用示例"""
    print("\n=== Pipeline 使用示例 ===")
    
    # 1. 情感分析pipeline
    print("1. 情感分析:")
    sentiment_pipeline = pipeline("sentiment-analysis")
    texts = ["I love this!", "I hate this!", "It's okay."]
    
    for text in texts:
        result = sentiment_pipeline(text)
        print(f"  {text} -> {result}")
    
    # 2. 文本生成pipeline
    print("\n2. 文本生成:")
    try:
        generator = pipeline("text-generation", model="gpt2")
        result = generator("The future of AI is", max_length=30, num_return_sequences=1)
        print(f"  生成文本: {result[0]['generated_text']}")
    except Exception as e:
        print(f"  文本生成失败: {e}")
    
    # 3. 问答pipeline
    print("\n3. 问答系统:")
    try:
        qa_pipeline = pipeline("question-answering")
        context = "Paris is the capital of France. It is known for the Eiffel Tower."
        question = "What is Paris known for?"
        
        result = qa_pipeline(question=question, context=context)
        print(f"  问题: {question}")
        print(f"  答案: {result['answer']}")
        print(f"  置信度: {result['score']:.4f}")
    except Exception as e:
        print(f"  问答失败: {e}")

def custom_training_example():
    """自定义训练示例（简化版）"""
    print("\n=== 自定义训练示例 ===")
    
    # 模拟训练数据
    class SimpleDataset(torch.utils.data.Dataset):
        def __init__(self, texts, labels, tokenizer, max_length=128):
            self.texts = texts
            self.labels = labels
            self.tokenizer = tokenizer
            self.max_length = max_length
        
        def __len__(self):
            return len(self.texts)
        
        def __getitem__(self, idx):
            text = str(self.texts[idx])
            label = self.labels[idx]
            
            encoding = self.tokenizer(
                text,
                truncation=True,
                padding='max_length',
                max_length=self.max_length,
                return_tensors='pt'
            )
            
            return {
                'input_ids': encoding['input_ids'].flatten(),
                'attention_mask': encoding['attention_mask'].flatten(),
                'labels': torch.tensor(label, dtype=torch.long)
            }
    
    # 示例数据
    train_texts = ["I love this", "I hate this", "This is great", "This is bad"]
    train_labels = [1, 0, 1, 0]  # 1=positive, 0=negative
    
    # 加载模型和分词器
    model_name = "distilbert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, 
        num_labels=2
    )
    
    # 创建数据集
    train_dataset = SimpleDataset(train_texts, train_labels, tokenizer)
    
    # 训练参数
    training_args = TrainingArguments(
        output_dir='./resources/trained_models/example_training',
        num_train_epochs=1,
        per_device_train_batch_size=2,
        warmup_steps=10,
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=1,
        save_steps=100,
        eval_steps=100,
        save_total_limit=2,
    )
    
    print(f"训练参数: {training_args}")
    print("注意: 这只是训练配置示例，实际训练需要更多数据和计算资源")

def model_saving_loading():
    """模型保存和加载"""
    print("\n=== 模型保存和加载 ===")
    
    model_name = "distilbert-base-uncased"
    save_path = "./resources/pretrained_models/example_model"
    
    # 加载模型
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    
    # 保存模型
    try:
        tokenizer.save_pretrained(save_path)
        model.save_pretrained(save_path)
        print(f"模型已保存到: {save_path}")
        
        # 从本地加载
        loaded_tokenizer = AutoTokenizer.from_pretrained(save_path)
        loaded_model = AutoModel.from_pretrained(save_path)
        print(f"模型已从本地加载: {save_path}")
        
    except Exception as e:
        print(f"保存/加载失败: {e}")

def advanced_features():
    """高级特性示例"""
    print("\n=== 高级特性 ===")
    
    # 1. 注意力可视化准备
    print("1. 注意力机制:")
    model_name = "distilbert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name, output_attentions=True)
    
    text = "The cat sat on the mat."
    inputs = tokenizer(text, return_tensors="pt")
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    attentions = outputs.attentions
    print(f"  注意力层数: {len(attentions)}")
    print(f"  第一层注意力形状: {attentions[0].shape}")
    
    # 2. 梯度检查点（内存优化）
    print("\n2. 内存优化:")
    print("  model.gradient_checkpointing_enable() - 启用梯度检查点")
    print("  这可以减少内存使用但会增加计算时间")
    
    # 3. 混合精度训练
    print("\n3. 混合精度训练:")
    print("  使用 torch.cuda.amp.autocast() 和 GradScaler")
    print("  可以加速训练并减少内存使用")

def main():
    """主函数 - 运行所有示例"""
    print("HuggingFace Transformers 库学习示例\n")
    
    basic_tokenizer_usage()
    model_loading_and_inference()
    classification_model()
    pipeline_usage()
    custom_training_example()
    model_saving_loading()
    advanced_features()
    
    print("\n=== 常用类和方法总结 ===")
    print("1. AutoTokenizer - 自动选择分词器")
    print("2. AutoModel - 基础模型（编码器）")
    print("3. AutoModelForSequenceClassification - 序列分类模型")
    print("4. pipeline() - 预配置的推理管道")
    print("5. TrainingArguments - 训练参数配置")
    print("6. Trainer - 训练器")
    print("7. .from_pretrained() - 加载预训练模型")
    print("8. .save_pretrained() - 保存模型")
    print("9. tokenizer() - 文本编码")
    print("10. model() - 模型前向传播")

if __name__ == "__main__":
    main() 