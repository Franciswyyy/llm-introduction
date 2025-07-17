# 示例和学习材料

本目录包含了项目的示例代码、学习材料和测试脚本。

## 目录结构

### 架构演示
- `architecture_overview.py` - 新架构概览和说明
- `demo_new_architecture.py` - 新架构演示脚本

### 学习材料
- `learn_datasets.py` - HuggingFace Datasets 库学习示例
- `learn_transformers.py` - HuggingFace Transformers 库学习示例

## 使用说明

### 1. 架构概览
```bash
python examples/architecture_overview.py
```
查看项目的模块化架构设计和各组件功能。

### 2. 新架构演示
```bash
python examples/demo_new_architecture.py
```
运行基于新架构的情感分析任务演示。

### 3. Datasets 库学习
```bash
python examples/learn_datasets.py
```
学习HuggingFace Datasets库的常用操作：
- 数据集创建和加载
- 数据变换和处理
- 批处理操作
- 格式转换
- 保存和加载

### 4. Transformers 库学习
```bash
python examples/learn_transformers.py
```
学习HuggingFace Transformers库的使用：
- 分词器使用
- 模型加载和推理
- Pipeline 使用
- 自定义训练
- 模型保存和加载

## 学习路径建议

1. **初学者**：
   - 先运行 `learn_datasets.py` 了解数据处理
   - 再运行 `learn_transformers.py` 了解模型使用
   - 最后查看 `architecture_overview.py` 理解项目架构

2. **有经验的开发者**：
   - 直接查看 `architecture_overview.py` 了解设计思路
   - 运行 `demo_new_architecture.py` 体验完整流程
   - 根据需要查看具体的学习示例

## 扩展学习

- 添加新的学习示例时，请遵循现有的代码风格
- 每个示例都应该是独立可运行的
- 建议添加详细的注释和说明
- 可以创建更多针对特定库或技术的学习示例

## 注意事项

- 某些示例需要网络连接来下载模型
- 运行前请确保已安装相关依赖
- 大型模型可能需要较多内存和时间 