文本分类

对电影评论的情感分析

1.表示模型进行文本分类

预训练表示模型进行分类，通常有两种方式：
- 使用特定任务模型：specific_task
- 使用嵌入模型


    # 加载预训练的多语言句子嵌入模型
    # all-mpnet-base-v2 是一个高质量的句子嵌入模型
    model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')







项目架构:
├── utils/
│   ├── config.py              # 统一配置管理
│   ├── model_manager.py       # 模型缓存和加载
│   └── data_builder.py        # 数据缓存和加载
├── core/
│   └── data/loaders.py        # 结构化数据加载器
└── tasks/
    ├── 1. text_classification/
    │   └── specific_task_v2.py     # Pipeline模型
    └── 01_text_classification/
        └── 02_embedding_classific.py  # 嵌入模型

统一的加载接口:
- load_model_pipeline(model_name)     # 用于transformers pipeline
- load_embedding_model(model_name)    # 用于sentence-transformers
- HuggingFaceLoader.load_dataset(name) # 用于数据集