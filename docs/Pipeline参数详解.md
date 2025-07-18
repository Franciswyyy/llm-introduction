# Pipeline 参数详解 📚

## 概述

`pipeline` 是 HuggingFace Transformers 库中最重要的函数之一，它让我们可以用几行代码就完成复杂的 NLP 任务。

## 基本语法

```python
from transformers import pipeline

pipe = pipeline(
    "任务类型",                    # 必需: 告诉AI要做什么
    model="模型名或路径",           # 可选: 指定用哪个模型
    tokenizer="分词器名或路径",     # 可选: 指定分词器
    return_all_scores=True,        # 可选: 是否返回所有分数
    device=-1                      # 可选: 运行设备
)
```

## 参数详解

### 1. 任务类型 (第一个参数) 🎯

**作用**: 告诉 pipeline 要执行什么类型的任务

```python
# 常见任务类型
"sentiment-analysis"     # 情感分析 (正面/负面)
"text-classification"    # 文本分类 (新闻/体育/科技等)
"text-generation"        # 文本生成
"question-answering"     # 问答系统
"summarization"          # 文本摘要
"translation"            # 翻译
"fill-mask"             # 填空题
"token-classification"   # 词性标注/命名实体识别
```

**❓ 可以随便写吗？**
```python
# ❌ 错误 - 不支持的任务类型
pipe = pipeline("随便写的任务")  # 会报错！

# ✅ 正确 - 必须是支持的任务类型
pipe = pipeline("sentiment-analysis")  # OK

# 💡 查看所有支持的任务
from transformers.pipelines import PIPELINE_REGISTRY
print(list(PIPELINE_REGISTRY.keys()))
```

**错误示例**:
```python
# 这会抛出异常
>>> pipeline("make-coffee")
ValueError: Unknown task make-coffee, available tasks are ['audio-classification', 'sentiment-analysis', ...]
```

### 2. model 参数 🤖

**作用**: 指定使用哪个预训练模型

```python
# HuggingFace Hub 上的模型
model="cardiffnlp/twitter-roberta-base-sentiment-latest"

# 本地模型路径
model="/path/to/local/model/"

# 如果不指定，会使用该任务的默认模型
pipe = pipeline("sentiment-analysis")  # 使用默认模型
```

### 3. tokenizer 参数 ✂️

**作用**: 将文本转换为模型能理解的数字tokens

```python
# 文本 → Tokens 的过程
"I love movies" → [101, 1045, 2293, 3152, 102]
```

**❓ 为什么模型和分词器路径一样，还要两个参数？**

```python
# 原因1: 分离关注点
model="bert-base-uncased"        # 负责预测
tokenizer="bert-base-uncased"    # 负责文本处理

# 原因2: 可以混合使用不同的分词器
model="my-custom-model"          # 我训练的模型
tokenizer="bert-base-uncased"    # 但用BERT的分词器

# 原因3: 有些情况下可能不一样
model="/local/fine-tuned-model"  # 本地微调的模型
tokenizer="bert-base-chinese"    # 中文分词器

# 原因4: 明确性和灵活性
# 虽然大多数情况下相同，但分开指定更清晰
```

**实际使用建议**:
```python
# 推荐: 明确指定 (更清晰)
pipe = pipeline(
    "sentiment-analysis",
    model=model_path,
    tokenizer=model_path
)

# 简化: 只指定model (tokenizer会自动匹配)
pipe = pipeline(
    "sentiment-analysis", 
    model=model_path
    # tokenizer 会自动使用同一路径
)
```

### 4. return_all_scores 参数 📊

**作用**: 控制返回结果的详细程度

```python
text = "This movie is okay"

# return_all_scores=False (默认)
result = [{'label': 'POSITIVE', 'score': 0.6}]
# 只返回最可能的答案

# return_all_scores=True  
result = [
    {'label': 'NEGATIVE', 'score': 0.4},
    {'label': 'NEUTRAL', 'score': 0.1}, 
    {'label': 'POSITIVE', 'score': 0.6}
]
# 返回所有可能的答案和分数
```

**为什么我们的项目使用 `True`？**

```python
# 我们需要比较 NEGATIVE 和 POSITIVE 的分数
def predict_sentiment(text):
    output = pipe(text)  # return_all_scores=True
    
    negative_score = output[0]["score"]  # NEGATIVE
    positive_score = output[2]["score"]  # POSITIVE
    
    # 手动比较，选择分数更高的
    prediction = np.argmax([negative_score, positive_score])
    return prediction

# 如果用 return_all_scores=False，我们只能得到最终答案
# 无法进行自定义的分数比较
```

### 5. device 参数 💻

**作用**: 指定模型运行的设备

```python
device=-1       # CPU
device=0        # 第一块GPU
device=1        # 第二块GPU  
device="cuda"   # GPU (自动选择)
device="mps"    # Apple Silicon GPU
```

**❓ 设备填错可以吗？**

```python
# ❌ 填写不存在的GPU
device=999      # 如果没有第999块GPU，会报错

# ❌ 填写不支持的设备
device="quantum"  # 不存在的设备类型

# ❌ 在没有GPU的机器上指定GPU
device=0        # 如果机器没有GPU，会报错

# ✅ 安全的做法
try:
    pipe = pipeline("sentiment-analysis", device=0)  # 尝试GPU
except:
    pipe = pipeline("sentiment-analysis", device=-1) # 失败则用CPU

# ✅ 自动选择 (推荐)
from utils import get_device
device = get_device()  # 自动选择最佳设备
pipe = pipeline("sentiment-analysis", device=device)
```

**设备性能对比**:

| 设备 | 速度 | 内存占用 | 适用场景 |
|------|------|----------|----------|
| CPU | 慢 (1x) | 少 | 测试、小数据 |
| CUDA GPU | 快 (10-50x) | 多 | 大规模推理 |
| MPS (Apple) | 中等 (3-8x) | 中等 | M1/M2 Mac |

## 完整示例

### 我们项目中的配置

```python
# utils/model_manager.py 中的实际配置
pipe = pipeline(
    "sentiment-analysis",                    # 情感分析任务
    model=model_path,                        # 缓存的模型路径
    tokenizer=model_path,                    # 配套的分词器
    return_all_scores=True,                  # 需要所有分数进行比较
    device=device_id                         # 自动选择的设备
)
```

### 常用配置模板

```python
# 1. 最简配置 (快速测试)
pipe = pipeline("sentiment-analysis")

# 2. 标准配置 (生产环境)
pipe = pipeline(
    "sentiment-analysis",
    model="cardiffnlp/twitter-roberta-base-sentiment-latest",
    return_all_scores=True,
    device="auto"
)

# 3. 高性能配置 (大规模处理)
pipe = pipeline(
    "sentiment-analysis",
    model="local/cached/model",
    return_all_scores=True,
    device=0,  # GPU
    batch_size=32
)

# 4. 安全配置 (处理长文本)
pipe = pipeline(
    "sentiment-analysis",
    model="model_name",
    max_length=512,
    truncation=True,
    padding=True
)
```

## 常见错误和解决方案

### 错误1: 任务类型写错

```python
# ❌ 错误
pipe = pipeline("sentiment")  # 不完整的任务名
pipe = pipeline("情感分析")    # 中文任务名

# ✅ 正确
pipe = pipeline("sentiment-analysis")
```

### 错误2: 设备不匹配

```python
# ❌ 错误 - 在CPU机器上强制使用GPU
pipe = pipeline("sentiment-analysis", device=0)

# ✅ 正确 - 自动检测
def get_device():
    if torch.cuda.is_available():
        return 0
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return "mps"
    else:
        return -1

pipe = pipeline("sentiment-analysis", device=get_device())
```

### 错误3: 分词器不匹配

```python
# ❌ 可能有问题 - 分词器和模型不匹配
pipe = pipeline(
    "sentiment-analysis",
    model="bert-base-uncased",
    tokenizer="roberta-base"  # 不同的分词器
)

# ✅ 推荐 - 使用匹配的分词器
pipe = pipeline(
    "sentiment-analysis",
    model="bert-base-uncased",
    tokenizer="bert-base-uncased"  # 或者省略，自动匹配
)
```

## 高级用法

### 批量处理

```python
# 单个文本
result = pipe("I love this movie!")

# 批量处理 (更高效)
texts = ["I love it!", "It's terrible.", "Just okay."]
results = pipe(texts)
```

### 自定义参数

```python
pipe = pipeline(
    "sentiment-analysis",
    model="model_name",
    return_all_scores=True,
    device=-1,
    # 分词器参数
    max_length=512,      # 最大长度
    truncation=True,     # 截断长文本  
    padding=True,        # 填充短文本
    # 其他参数
    batch_size=8,        # 批处理大小
)
```

## 自问自答环节 🤔

### Q1: 第一个参数可以随便写吗？
**A**: ❌ 不可以！必须是HuggingFace支持的任务类型。

```python
# 查看所有支持的任务类型
from transformers.pipelines import PIPELINE_REGISTRY
print(list(PIPELINE_REGISTRY.keys()))

# 常见的任务类型：
# 'sentiment-analysis', 'text-classification', 'text-generation', 
# 'question-answering', 'summarization', 'translation' 等
```

### Q2: 分词器路径和模型既然一样，为什么要两个参数？
**A**: 虽然大多数情况下相同，但分开指定有重要意义：

1. **分离关注点**: 模型负责预测，分词器负责文本处理
2. **灵活性**: 可以混合使用（自训练模型 + 标准分词器）
3. **明确性**: 代码更清晰，意图更明确
4. **未来扩展**: 支持更复杂的模型组合

```python
# 实际上可以只指定model，tokenizer会自动匹配
pipe = pipeline("sentiment-analysis", model="model_name")

# 但明确指定更好
pipe = pipeline("sentiment-analysis", model="model_name", tokenizer="model_name")
```

### Q3: 设备填错可以吗？
**A**: ❌ 不可以！会导致程序崩溃。

```python
# 常见错误：
device=999      # GPU不存在 → RuntimeError
device="quantum" # 设备不存在 → ValueError  
device=0        # 没有GPU的机器上指定GPU → RuntimeError

# 安全做法：
device = get_device()  # 自动检测
# 或者用try-catch处理异常
```

## 总结

### 必须记住的要点

1. **任务类型**: 必须是支持的类型，不能随便写
2. **model/tokenizer**: 通常相同，分开指定更灵活
3. **return_all_scores**: True获取详细分数，False只要最终答案
4. **device**: 填错会报错，建议自动检测

### 推荐的配置策略

```python
# 开发测试阶段
pipe = pipeline("sentiment-analysis", device=-1)  # CPU，稳定

# 生产环境
pipe = pipeline(
    "sentiment-analysis", 
    model="cached/model/path",  # 使用缓存模型
    return_all_scores=True,     # 获取详细信息
    device="auto"               # 自动选择最佳设备
)
```

### 性能优化提示

1. **使用本地缓存模型** - 避免重复下载
2. **批量处理** - 一次处理多个文本更高效
3. **合适的设备** - GPU > MPS > CPU
4. **合理的参数** - 根据需求设置max_length等

---

*本文档基于项目实际使用经验总结，涵盖了 pipeline 参数的核心概念和常见问题。* 