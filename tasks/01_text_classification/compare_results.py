#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
文本分类任务结果对比分析脚本
运行三个不同的分类方法并生成对比报告
"""

import sys
import subprocess
import os
import re
from pathlib import Path
from datetime import datetime

# 添加项目根目录到Python路径
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.append(str(PROJECT_ROOT))

def run_script_and_capture_output(script_path):
    """
    运行Python脚本并捕获输出
    
    Args:
        script_path: Python脚本路径
        
    Returns:
        tuple: (return_code, stdout, stderr)
    """
    try:
        print(f"🚀 正在运行: {script_path}")
        result = subprocess.run(
            [sys.executable, script_path],
            capture_output=True,
            text=True,
            cwd=PROJECT_ROOT
        )
        return result.returncode, result.stdout, result.stderr
    except Exception as e:
        return -1, "", str(e)

def extract_classification_report(output):
    """
    从输出中提取分类报告
    
    Args:
        output: 脚本输出文本
        
    Returns:
        str: 提取的分类报告
    """
    lines = output.split('\n')
    report_lines = []
    in_report = False
    
    for line in lines:
        # 寻找分类报告的开始
        if 'precision' in line and 'recall' in line and 'f1-score' in line:
            in_report = True
            report_lines.append(line)
            continue
            
        if in_report:
            # 如果遇到空行且已经收集了足够的内容，结束收集
            if line.strip() == '' and len(report_lines) > 5:
                break
            report_lines.append(line)
            
    return '\n'.join(report_lines) if report_lines else "未找到分类报告"

def extract_performance_metrics(report):
    """
    从分类报告中提取关键性能指标
    
    Args:
        report: 分类报告文本
        
    Returns:
        dict: 包含精确率、召回率、F1分数的字典
    """
    metrics = {}
    
    # 查找宏平均或加权平均行
    lines = report.split('\n')
    for line in lines:
        if 'macro avg' in line or 'weighted avg' in line:
            parts = line.split()
            if len(parts) >= 4:
                try:
                    metrics['precision'] = float(parts[-4])
                    metrics['recall'] = float(parts[-3])
                    metrics['f1_score'] = float(parts[-2])
                    break
                except (ValueError, IndexError):
                    continue
                    
    return metrics

def generate_markdown_report(results):
    """
    生成Markdown格式的对比报告
    
    Args:
        results: 包含三个任务结果的字典
    """
    report_content = f"""# 文本分类任务性能对比分析

> 生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## 📋 任务概述

本报告对比了四种不同的文本分类方法在Rotten Tomatoes数据集上的性能表现：

1. **预训练模型直接分类** (`01_specific_task_v2.py`)
   - 使用twitter-roberta-base-sentiment-latest预训练模型
   - 无需训练，直接进行情感分析
   
2. **嵌入模型+分类器** (`02_embedding_classific.py`)
   - 使用Sentence Transformer生成文本嵌入
   - 训练逻辑回归分类器进行分类
   
3. **零样本分类** (`03_zero_shot_classification.py`)
   - 使用预训练嵌入模型计算文本与标签的相似度
   - 无需训练数据，通过相似度进行分类
   
4. **生成模型分类** (`04_text_generation_classification.py`)
   - 使用FLAN-T5文本生成模型进行prompt-based分类
   - 将分类任务转换为文本生成任务

## 📊 性能对比

### 详细分类报告

"""

    # 添加每个任务的详细结果
    task_names = {
        '01_specific_task_v2.py': '预训练模型直接分类',
        '02_embedding_classific.py': '嵌入模型+分类器',
        '03_zero_shot_classification.py': '零样本分类',
        '04_text_generation_classification.py': '生成模型分类'
    }
    
    for script, data in results.items():
        task_name = task_names.get(script, script)
        report_content += f"#### {task_name}\n\n"
        
        if data['success']:
            report_content += "```\n"
            report_content += data['report']
            report_content += "\n```\n\n"
        else:
            report_content += f"❌ **执行失败**: {data['error']}\n\n"
    
    # 添加性能指标对比表
    report_content += "### 性能指标汇总\n\n"
    report_content += "| 方法 | 精确率 | 召回率 | F1分数 | 状态 |\n"
    report_content += "|------|--------|--------|--------|------|\n"
    
    for script, data in results.items():
        task_name = task_names.get(script, script)
        if data['success'] and data['metrics']:
            metrics = data['metrics']
            report_content += f"| {task_name} | {metrics.get('precision', 'N/A'):.4f} | {metrics.get('recall', 'N/A'):.4f} | {metrics.get('f1_score', 'N/A'):.4f} | ✅ 成功 |\n"
        else:
            report_content += f"| {task_name} | N/A | N/A | N/A | ❌ 失败 |\n"
    
    # 添加结果解释
    report_content += """

## 📖 结果解释

### 性能指标说明

- **精确率 (Precision)**: 预测为正类的样本中，真正为正类的比例
  - 公式: TP / (TP + FP)
  - 值越高表示假阳性越少

- **召回率 (Recall)**: 实际为正类的样本中，被正确预测为正类的比例
  - 公式: TP / (TP + FN)
  - 值越高表示假阴性越少

- **F1分数 (F1-Score)**: 精确率和召回率的调和平均数
  - 公式: 2 × (Precision × Recall) / (Precision + Recall)
  - 综合评估指标，平衡精确率和召回率

### 方法对比分析

1. **预训练模型直接分类**
   - 优点: 实现简单，性能通常较好，无需训练
   - 缺点: 模型固定，难以针对特定任务优化

2. **嵌入模型+分类器**
   - 优点: 可以根据具体数据进行训练优化
   - 缺点: 需要训练数据，计算开销较大

3. **零样本分类**
   - 优点: 无需训练数据，灵活性高
   - 缺点: 性能可能不如专门训练的模型

4. **生成模型分类**
   - 优点: 利用生成模型的语言理解能力，可处理复杂的prompt
   - 缺点: 计算开销大，生成结果需要后处理

### 应用建议

- 当有足够标注数据时，推荐使用**嵌入模型+分类器**方法
- 当需要快速部署且无训练数据时，推荐使用**零样本分类**
- 当追求最佳性能且适合预训练模型的任务时，推荐使用**预训练模型直接分类**
- 当需要处理复杂的自然语言理解任务时，推荐使用**生成模型分类**

## 🔧 技术细节

- **数据集**: Rotten Tomatoes (影评情感分析)
- **评估指标**: 精确率、召回率、F1分数
- **标签类别**: 负面评价、正面评价 (二分类)
- **测试环境**: {os.name} 系统

---
*此报告由自动化脚本生成，用于对比不同文本分类方法的性能表现。*
"""

    # 保存报告到文件
    report_path = Path(__file__).parent / "classification_comparison_report.md"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report_content)
    
    print(f"📄 对比报告已生成: {report_path}")
    return report_path

def main():
    """主函数"""
    print("🎯 开始执行文本分类任务对比分析...")
    
    # 定义要运行的脚本
    scripts = [
        "tasks/01_text_classification/01_specific_task_v2.py",
        "tasks/01_text_classification/02_embedding_classific.py", 
        "tasks/01_text_classification/03_zero_shot_classification.py",
        "tasks/01_text_classification/04_text_generation_classification.py"
    ]
    
    results = {}
    
    # 运行每个脚本并收集结果
    for script in scripts:
        script_path = PROJECT_ROOT / script
        script_name = os.path.basename(script)
        
        return_code, stdout, stderr = run_script_and_capture_output(script_path)
        
        if return_code == 0:
            # 提取分类报告
            report = extract_classification_report(stdout)
            metrics = extract_performance_metrics(report)
            
            results[script_name] = {
                'success': True,
                'report': report,
                'metrics': metrics,
                'stdout': stdout,
                'stderr': stderr
            }
            print(f"✅ {script_name} 执行成功")
        else:
            results[script_name] = {
                'success': False,
                'report': "",
                'metrics': {},
                'error': stderr or "未知错误",
                'stdout': stdout,
                'stderr': stderr
            }
            print(f"❌ {script_name} 执行失败: {stderr}")
    
    # 生成对比报告
    report_path = generate_markdown_report(results)
    
    print("\n🎉 分析完成! 查看生成的对比报告:")
    print(f"📁 {report_path}")

if __name__ == "__main__":
    main() 