import json
import re
import os
import torch
from typing import Dict, List, Optional, Tuple
from torch.utils.data import DataLoader
from transformers import PreTrainedTokenizer
from accelerate import Accelerator
from omegaconf import DictConfig
from tqdm import tqdm

from utils.dataset_eval import get_eval_dataloader
from utils.reward import extract_answer, grade


def check_format_reward(generated_text: str) -> float:
    """
    检查生成的答案是否符合格式要求
    
    使用 reward.py 的逻辑：检查是否包含 \\boxed{...} 格式
    
    Returns:
        format_reward: 1.0 如果格式正确，0.0 否则
    """
    # 检查是否包含 \\boxed{...} 或 \boxed{...}
    if "\\boxed" in generated_text:
        # 进一步验证是否能成功提取答案
        extracted = extract_answer(generated_text)
        if extracted is not None:
            return 1.0
    
    return 0.0


def check_answer_reward(generated_text: str, ground_truth: str, is_fast: bool = True) -> float:
    """
    检查生成的答案是否正确
    
    使用 reward.py 中的 grade 函数，它集成了 math-verify 和多种验证方法
    
    Args:
        generated_text: 生成的文本
        ground_truth: 标准答案
        is_fast: 是否使用快速模式
    Returns:
        answer_reward: 1.0 如果答案正确，0.0 否则
    """
    if not ground_truth:
        return 0.0
    
    # 提取生成的答案
    extracted_answer = extract_answer(generated_text)
    if extracted_answer is None:
        return 0.0
    
    is_correct = grade(extracted_answer, ground_truth, fast=is_fast)
    return 1.0 if is_correct else 0.0


@torch.no_grad()
def evaluate_dataset(
    model,
    tokenizer: PreTrainedTokenizer,
    dataloader: DataLoader,
    accelerator: Accelerator,
    dataset_name: str,
    gen_length: int = 512,
    block_size: int = 4,
    temperature: float = 1.0,
    ratio: Optional[float] = None,
    max_samples: Optional[int] = None,
    save_results: bool = True,
    output_dir: Optional[str] = None,
    global_step: Optional[int] = None,
    is_fast: bool = True,
) -> Dict[str, float]:
    """
    在单个数据集上进行评估（分布式版本，所有进程都参与）
    
    Args:
        model: 模型实例
        tokenizer: tokenizer实例
        dataloader: 评估数据加载器（已使用DistributedSampler）
        accelerator: Accelerator实例
        dataset_name: 数据集名称（用于日志）
        gen_length: 生成长度
        block_size: block大小
        temperature: 温度参数
        ratio: ratio参数
        max_samples: 最大评估样本数（None表示全部）
        
    Returns:
        metrics: 包含format_reward和answer_reward的字典（全局汇总结果）
    """
    model.eval()
    
    local_format_reward = 0.0
    local_answer_reward = 0.0
    local_count = 0
    local_results = []  # 存储每个样本的详细信息
    
    total_batches = len(dataloader)
    
    if accelerator.is_main_process:
        pbar = tqdm(
            total=total_batches,
            desc=f"Evaluating {dataset_name}",
            unit="batch",
            disable=False,
            ncols=100,
        )
    else:
        pbar = None
        
    
    for batch_idx, batch in enumerate(dataloader):
        # batch是一个list of dicts（因为batch_size=1）
        for sample in batch:
            question = sample["question"]
            ground_truth = sample["ground_truth_answer"]
            sample_index = sample.get("index", batch_idx)

            messages = [
                {"role": "user", "content": question}
            ]
            # 使用add_generation_prompt=True，提示模型开始生成assistant的回复
            user_prompt = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            # Tokenize格式化后的prompt
            prompt_tokenized = tokenizer(
                user_prompt,
                add_special_tokens=False,
                return_tensors="pt",
            )
            prompt_ids = prompt_tokenized["input_ids"]
            
            if prompt_ids.shape[1] == 0:
                continue
                
            # 移动到正确的设备
            prompt_ids = prompt_ids.to(accelerator.device)
            
            # 生成回答
            generated_ids = model.generate(
                prompt_ids=prompt_ids,
                gen_length=gen_length,
                block_size=block_size,
                temperature=temperature,
                ratio=ratio,
            )
            
            # 解码生成的文本（只取生成的部分，去掉prompt）
            prompt_length = prompt_ids.shape[1]
            generated_ids_only = generated_ids[0, prompt_length:]  # 只取生成的部分
            generated_text = tokenizer.decode(
                generated_ids_only, 
                skip_special_tokens=False
            )
            
            # 计算rewards
            format_reward = check_format_reward(generated_text)
            answer_reward = check_answer_reward(generated_text, ground_truth, is_fast=is_fast)
            
            local_format_reward += format_reward
            local_answer_reward += answer_reward
            local_count += 1
            
            # 保存详细信息
            if save_results:
                local_results.append({
                    "index": sample_index,
                    "prompt": user_prompt,
                    "ground_truth_answer": ground_truth,
                    "generated_text": generated_text,
                    "format_reward": float(format_reward),
                    "answer_reward": float(answer_reward),
                    "extracted_answer": extract_answer(generated_text),
                })
            
            # 更新进度条（只在主进程更新）
            if pbar is not None:
                pbar.update(1)
                # 更新进度条描述，显示当前统计信息
                pbar.set_postfix({
                    "samples": local_count,
                    "format": f"{local_format_reward/local_count:.3f}" if local_count > 0 else "0.000",
                    "answer": f"{local_answer_reward/local_count:.3f}" if local_count > 0 else "0.000",
                })
                

    
    # 关闭进度条
    if pbar is not None:
        pbar.close()
    
    # 将本地结果转换为tensor以便收集
    local_format_reward_tensor = torch.tensor(local_format_reward, device=accelerator.device)
    local_answer_reward_tensor = torch.tensor(local_answer_reward, device=accelerator.device)
    local_count_tensor = torch.tensor(local_count, device=accelerator.device, dtype=torch.long)
    
    # 收集所有进程的结果
    gathered_format_rewards = accelerator.gather(local_format_reward_tensor)
    gathered_answer_rewards = accelerator.gather(local_answer_reward_tensor)
    gathered_counts = accelerator.gather(local_count_tensor)
    
    # 收集所有进程的详细结果（用于保存JSON）
    all_results = []
    if save_results:
        # 使用gather_object收集所有进程的结果列表
        try:
            from accelerate.utils import gather_object
            # gather_object收集所有进程的对象列表
            gathered_results = gather_object([local_results])
            # gathered_results是形状为[num_processes, ...]的列表
            if gathered_results is not None:
                # 展平结果列表
                for rank_results in gathered_results:
                    if isinstance(rank_results, list):
                        all_results.extend(rank_results)
        except (ImportError, AttributeError):
            # 如果gather_object不可用，使用临时文件方法
            import tempfile
            import pickle
            
            # 每个进程保存自己的结果到临时文件
            temp_file = os.path.join(tempfile.gettempdir(), f"eval_results_rank_{accelerator.process_index}.pkl")
            with open(temp_file, 'wb') as f:
                pickle.dump(local_results, f)
            
            # 同步所有进程
            accelerator.wait_for_everyone()
            
            # 主进程读取所有临时文件并合并
            if accelerator.is_main_process:
                for rank in range(accelerator.num_processes):
                    rank_temp_file = os.path.join(tempfile.gettempdir(), f"eval_results_rank_{rank}.pkl")
                    if os.path.exists(rank_temp_file):
                        with open(rank_temp_file, 'rb') as f:
                            rank_results = pickle.load(f)
                            all_results.extend(rank_results)
                        # 删除临时文件
                        os.remove(rank_temp_file)
            
            # 确保所有进程等待主进程完成文件读取和删除
            accelerator.wait_for_everyone()
    
    # 只在主进程上计算全局平均值和保存结果
    if accelerator.is_main_process:
        total_format_reward = gathered_format_rewards.sum().item()
        total_answer_reward = gathered_answer_rewards.sum().item()
        total_count = gathered_counts.sum().item()
        
        if total_count > 0:
            avg_format_reward = total_format_reward / total_count
            avg_answer_reward = total_answer_reward / total_count
        else:
            avg_format_reward = 0.0
            avg_answer_reward = 0.0
        
        metrics = {
            "format_reward": avg_format_reward,
            "answer_reward": avg_answer_reward,
            "total_samples": total_count,
        }
        
        # 保存详细结果到JSON文件
        if save_results and output_dir is not None and len(all_results) > 0:
            # 按index排序，确保顺序一致
            all_results.sort(key=lambda x: x.get("index", 0))
            
            # 创建输出目录
            os.makedirs(output_dir, exist_ok=True)
            
            # 生成文件名
            if global_step is not None:
                filename = f"{dataset_name}_step_{global_step}.json"
            else:
                filename = f"{dataset_name}.json"
            
            filepath = os.path.join(output_dir, filename)
            
            # 保存为JSON文件
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(all_results, f, ensure_ascii=False, indent=2)
            
            print(f"  Saved {len(all_results)} results to {filepath}")
    else:
        # 非主进程返回空字典（不会被使用）
        metrics = {
            "format_reward": 0.0,
            "answer_reward": 0.0,
            "total_samples": 0,
        }
    
    # 确保所有进程等待评估完成
    accelerator.wait_for_everyone()
    
    return metrics


def math_eval(
    model,
    config: DictConfig,
    accelerator: Accelerator,
    global_step: int,
    tokenizer: PreTrainedTokenizer,
):
    """
    在多个数学数据集上进行评估
    
    Args:
        model: 模型实例
        config: 配置对象
        accelerator: Accelerator实例
        global_step: 当前训练步数
        tokenizer: tokenizer实例
    """
    if not hasattr(config, 'eval') or config.eval is None:
        if accelerator.is_main_process:
            print("No eval configuration found, skipping evaluation.")
        return
    
    # 评估参数
    gen_length = getattr(config.eval, 'gen_length', 512)
    block_size = getattr(config.eval, 'block_size', 4)
    temperature = getattr(config.eval, 'temperature', 1.0)
    ratio = getattr(config.eval, 'ratio', None)
    max_samples_per_dataset = getattr(config.eval, 'max_samples', None)
    save_results = getattr(config.eval, 'save_results', True)  # 是否保存详细结果
    is_fast = getattr(config.eval, 'is_fast', True)  # 是否使用快速模式（默认True，使用mathd和sympy；False会额外使用math-verify）
    
    # 创建输出目录（用于保存生成的文本）
    # 所有进程都需要知道output_dir，但只有主进程创建目录
    if save_results:
        if hasattr(config, 'experiment') and hasattr(config.experiment, 'output_dir'):
            output_dir = os.path.join(config.experiment.output_dir, "eval_results")
        else:
            output_dir = "eval_results"
        if accelerator.is_main_process:
            os.makedirs(output_dir, exist_ok=True)
    else:
        output_dir = None
    
    # 确保所有进程都知道output_dir（通过broadcast或wait_for_everyone）
    accelerator.wait_for_everyone()
    
    all_metrics = {}
    
    # 遍历所有配置的评估数据集
    for dataset_name, dataset_config in config.eval.items():
        if dataset_name in ['gen_length', 'block_size', 'temperature', 'ratio', 'max_samples']:
            continue  # 跳过全局参数
        
        if not hasattr(dataset_config, 'data_path'):
            continue
        
        data_path = dataset_config.data_path
        question_key = getattr(dataset_config, 'question_key', 'question')
        answer_key = getattr(dataset_config, 'answer_key', 'ground_truth_answer')
        
        # 检查文件是否存在（所有进程都检查，但只在主进程打印警告）
        file_exists = os.path.exists(data_path)
        if not file_exists:
            if accelerator.is_main_process:
                print(f"Warning: Evaluation dataset {dataset_name} not found at {data_path}, skipping.")
            # 确保所有进程都跳过这个数据集
            accelerator.wait_for_everyone()
            continue
        
        if accelerator.is_main_process:
            print(f"\n{'='*60}")
            print(f"Evaluating on {dataset_name}...")
            print(f"Data path: {data_path}")
            print(f"{'='*60}\n")
        
        # 所有进程都创建数据加载器（使用DistributedSampler）
        try:
            eval_dataloader = get_eval_dataloader(
                data_path=data_path,
                tokenizer=tokenizer,
                question_key=question_key,
                answer_key=answer_key,
                batch_size=1,  # 评估时通常使用batch_size=1
                num_workers=0,  # 评估时通常不需要多进程加载
                max_samples=max_samples_per_dataset,
                shuffle=False,
            )
            
            # 使用accelerator准备dataloader，这会自动添加DistributedSampler
            eval_dataloader = accelerator.prepare(eval_dataloader)
            
            # 所有进程都执行评估
            metrics = evaluate_dataset(
                model=model,
                tokenizer=tokenizer,
                dataloader=eval_dataloader,
                accelerator=accelerator,
                dataset_name=dataset_name,
                gen_length=gen_length,
                block_size=block_size,
                temperature=temperature,
                ratio=ratio,
                max_samples=max_samples_per_dataset,
                save_results=save_results,
                output_dir=output_dir,
                global_step=global_step,
                is_fast=is_fast,
            )
            
            # 只在主进程上保存和打印结果
            if accelerator.is_main_process:
                # 保存指标
                all_metrics[dataset_name] = metrics
                
                # 打印结果
                print(f"\n{dataset_name} Results:")
                print(f"  Format Reward: {metrics['format_reward']:.4f}")
                print(f"  Answer Reward: {metrics['answer_reward']:.4f}")
                print(f"  Total Samples: {metrics['total_samples']}")
                
                # 记录到tensorboard
                logs = {
                    f"eval/{dataset_name}/format_reward": metrics['format_reward'],
                    f"eval/{dataset_name}/answer_reward": metrics['answer_reward'],
                    f"eval/{dataset_name}/total_samples": metrics['total_samples'],
                }
                accelerator.log(logs, step=global_step)
        
        except Exception as e:
            if accelerator.is_main_process:
                print(f"Error evaluating {dataset_name}: {e}")
                import traceback
                traceback.print_exc()
            # 确保所有进程都跳过这个数据集
            accelerator.wait_for_everyone()
            continue
        
        # 确保所有进程等待评估完成
        accelerator.wait_for_everyone()
    
    # 计算平均指标（如果有多个数据集）
    if len(all_metrics) > 0 and accelerator.is_main_process:
        avg_format_reward = sum(m['format_reward'] for m in all_metrics.values()) / len(all_metrics)
        avg_answer_reward = sum(m['answer_reward'] for m in all_metrics.values()) / len(all_metrics)
        
        logs = {
            "eval/avg_format_reward": avg_format_reward,
            "eval/avg_answer_reward": avg_answer_reward,
        }
        accelerator.log(logs, step=global_step)
        
        print(f"\n{'='*60}")
        print(f"Average Results across all datasets:")
        print(f"  Average Format Reward: {avg_format_reward:.4f}")
        print(f"  Average Answer Reward: {avg_answer_reward:.4f}")
        print(f"{'='*60}\n")

