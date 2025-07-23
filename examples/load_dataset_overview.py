# 确保你已经安装了相关库: pip install datasets pandas
from datasets import load_dataset

# 1. 加载数据集 (这一步背后可能在下载和处理.arrow文件)
dataset = load_dataset("rotten_tomatoes")

# --- 现在，我们开始“勘探”这个“看不见”的数据 ---

# 2. 【宏观勘探】首先，查看数据集的整体结构和元信息
print("--- 1. 数据集宏观结构 ---")
print(dataset)
# 输出会告诉你，它有train/validation/test三部分，每部分有多少行

print("\n--- 2. 查看数据列的“表结构”(Schema) ---")
print(dataset['train'].features)
# 输出会告诉你，它有'text'和'label'两列，以及它们的数据类型

# 3. 【抽样勘探】从训练集中随机抽取10个样本来“看一看”
# .shuffle() 会打乱数据顺序，.select() 选择指定的行数
print("\n--- 3. 随机抽取10个样本进行观察 ---")
sample_dataset = dataset['train'].shuffle(seed=42).select(range(10))

# 为了方便查看，我们可以把它转换成pandas DataFrame (非常流行的表格数据处理库)
sample_df = sample_dataset.to_pandas()
print(sample_df)

# 4. 【导出样本】如果还想在Excel里看，可以把这个小样本导出成CSV
print("\n--- 4. 将这10个样本导出到CSV文件 ---")
csv_path = "rotten_tomatoes_sample.csv"
sample_df.to_csv(csv_path, index=False, encoding='utf-8-sig')
print(f"已成功将10个样本导出到: {csv_path}")
print("您现在可以用Excel或记事本打开这个CSV文件了。")



"""
这套流程总结：

加载数据： 使用datasets库。

看结构： 用print(dataset)和print(dataset.features)来理解数据的宏观“表结构”。

看内容： 用.shuffle().select()随机抽取一小部分样本。

转成熟悉格式： 把这个小样本转换成Pandas DataFrame，这是Python数据分析的“标准表格”，您可以非常方便地对它进行各种操作。

（可选）导出： 如果需要，把这个小DataFrame导出成CSV，满足您在外部工具中“亲眼看看”的需求。


"""
