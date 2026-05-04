#!/usr/bin/env python3
"""
评测结果计算脚本
计算各个评测任务的分数，对于包含子任务的任务进行求平均
"""

import json
import sys
import argparse
from typing import Dict, List, Any


def get_main_metric(task_name: str, task_data: Dict[str, Any]) -> float:
    """
    获取任务的主要指标分数
    
    Args:
        task_name: 任务名称
        task_data: 任务数据字典
    
    Returns:
        主要指标的分数
    """
    # record任务使用f1作为主要指标
    if task_name == "record":
        if "f1,none" in task_data:
            return task_data["f1,none"]
        elif "em,none" in task_data:
            return task_data["em,none"]
    
    # 其他任务优先使用acc_norm，如果没有则使用acc
    if "acc_norm,none" in task_data:
        return task_data["acc_norm,none"]
    elif "acc,none" in task_data:
        return task_data["acc,none"]
    elif "f1,none" in task_data:
        return task_data["f1,none"]
    
    # 如果都没有，返回0并打印警告
    print(f"警告: 任务 {task_name} 没有找到主要指标", file=sys.stderr)
    return 0.0


def calculate_super_glue_score(results: Dict[str, Any]) -> Dict[str, Any]:
    """
    计算super-glue的分数（包含多个子任务）
    
    Args:
        results: 结果字典
    
    Returns:
        包含子任务分数和平均分的字典
    """
    super_glue_tasks = ["boolq", "cb", "copa", "multirc", "record", "sglue_rte", "wic", "wsc"]
    sub_scores = {}
    scores = []
    
    for task in super_glue_tasks:
        if task in results:
            score = get_main_metric(task, results[task])
            sub_scores[task] = score
            scores.append(score)
        else:
            print(f"警告: super-glue子任务 {task} 未找到", file=sys.stderr)
    
    avg_score = sum(scores) / len(scores) if scores else 0.0
    
    return {
        "subtasks": sub_scores,
        "average": avg_score
    }


def calculate_tqa_score(results: Dict[str, Any]) -> Dict[str, Any]:
    """
    计算tqa的分数（包含多个子任务）
    
    Args:
        results: 结果字典
    
    Returns:
        包含子任务分数和平均分的字典
    """
    tqa_tasks = ["truthfulqa_mc1", "truthfulqa_mc2"]
    sub_scores = {}
    scores = []
    
    for task in tqa_tasks:
        if task in results:
            score = get_main_metric(task, results[task])
            sub_scores[task] = score
            scores.append(score)
        else:
            print(f"警告: tqa子任务 {task} 未找到", file=sys.stderr)
    
    avg_score = sum(scores) / len(scores) if scores else 0.0
    
    return {
        "subtasks": sub_scores,
        "average": avg_score
    }


def calculate_mmlu_score(results: Dict[str, Any]) -> Dict[str, Any]:
    """
    计算mmlu的分数（不显示子任务）
    
    Args:
        results: 结果字典
    
    Returns:
        只包含平均分的字典
    """
    if "mmlu" in results:
        score = get_main_metric("mmlu", results["mmlu"])
        return {
            "average": score
        }
    else:
        print("警告: mmlu任务未找到", file=sys.stderr)
        return {
            "average": 0.0
        }


def calculate_single_task_score(task_name: str, results: Dict[str, Any]) -> Dict[str, Any]:
    """
    计算单个任务的分数（显示所有可用指标）
    
    Args:
        task_name: 任务名称
        results: 结果字典
    
    Returns:
        包含所有指标的字典
    """
    if task_name not in results:
        print(f"警告: 任务 {task_name} 未找到", file=sys.stderr)
        return {}
    
    task_data = results[task_name]
    metrics = {}
    
    # 提取所有指标
    for key, value in task_data.items():
        if key != "alias" and isinstance(value, (int, float)):
            metrics[key] = value
    
    # 添加主要指标
    main_score = get_main_metric(task_name, task_data)
    metrics["main_score"] = main_score
    
    return metrics

def calculate_PPL_score(task_name: str, results: Dict[str, Any]) -> Dict[str, Any]:
    if task_name not in results:
        print(f"警告: 任务 {task_name} 未找到", file=sys.stderr)
        return {}
    
    task_data = results[task_name]
    metrics = {}
    
    # 提取所有指标
    for key, value in task_data.items():
        if key != "alias" and isinstance(value, (int, float)):
            metrics[key] = value
    
    return metrics


def main():
    parser = argparse.ArgumentParser(description="计算评测任务的分数")
    parser.add_argument("json_file", type=str, help="评测结果JSON文件路径")
    parser.add_argument("--output", "-o", type=str, help="输出文件路径（可选，默认输出到stdout）")
    args = parser.parse_args()
    
    # 读取JSON文件
    try:
        with open(args.json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"错误: 文件 {args.json_file} 未找到", file=sys.stderr)
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"错误: JSON解析失败: {e}", file=sys.stderr)
        sys.exit(1)
    
    if "results" not in data:
        print("错误: JSON文件中没有找到'results'字段", file=sys.stderr)
        sys.exit(1)
    
    results = data["results"]
    output = {}
    
    # 1. super-glue-lm-eval-v1 (包含多个子任务)
    print("计算 super-glue-lm-eval-v1...", file=sys.stderr)
    output["super-glue-lm-eval-v1"] = calculate_super_glue_score(results)
    
    # 2. tqa (包含多个子任务)
    print("计算 tqa...", file=sys.stderr)
    output["tqa"] = calculate_tqa_score(results)
    
    # 3. 其他独立任务
    independent_tasks = [
        "arc_easy",
        "hellaswag",
        "openbookqa",
        "piqa",  # 注意：用户提到的是piqa_acc，但实际任务名可能是piqa
        "sciq",
        "winogrande",
        "gpqa_diamond_zeroshot"
    ]
    
    for task in independent_tasks:
        print(f"计算 {task}...", file=sys.stderr)
        output[task] = calculate_single_task_score(task, results)
        
    # 4. PPL
    PPL_tasks = [
        "lambada_openai",
        "wikitext"
    ]
    for task in PPL_tasks:
        print(f"计算 {task}...", file=sys.stderr)
        output[task] = calculate_PPL_score(task, results)
    
    # 输出结果
    output_json = json.dumps(output, indent=2, ensure_ascii=False)
    
    if args.output:
        with open(args.output, 'w', encoding='utf-8') as f:
            f.write(output_json)
        print(f"结果已保存到 {args.output}", file=sys.stderr)
    else:
        print(output_json)


if __name__ == "__main__":
    main()
