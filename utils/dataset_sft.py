import json
import os
import time
import multiprocessing
from typing import Dict, List, Optional
from torch.utils.data import Dataset, DataLoader
from transformers import PreTrainedTokenizer
from omegaconf import DictConfig
from datasets import load_dataset, load_from_disk, Dataset as HFDataset


class SFTDataset(Dataset):
    """
    用于SFT训练的数据集类
    从jsonl文件加载数据，分别提供question和solution（使用generation_solution字段）
    """
    
    def __init__(
        self,
        data_path: str,
        tokenizer: PreTrainedTokenizer,
        max_length: Optional[int] = None,
        question_key: str = "question",
        solution_key: str = "generation_solution",
        fallback_question_key: str = "problem",
        fallback_solution_key: str = "solution",
    ):
        """
        Args:
            data_path: jsonl文件路径
            tokenizer: tokenizer实例
            max_length: 最大序列长度，如果为None则不截断
            question_key: 问题字段的key
            solution_key: 解答字段的key（优先使用）
            fallback_question_key: 问题字段的备用key
            fallback_solution_key: 解答字段的备用key
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.question_key = question_key
        self.solution_key = solution_key
        self.fallback_question_key = fallback_question_key
        self.fallback_solution_key = fallback_solution_key
        
        # 加载数据
        self.data = self._load_data(data_path)
        
    def _load_data(self, data_path: str) -> List[Dict]:
        """从jsonl文件加载数据"""
        data = []
        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    data.append(json.loads(line))
        return data
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, str]:
        """
        返回一个样本，包含question和solution
        """
        sample = self.data[idx]
        
        # 获取question
        question = sample.get(
            self.question_key, 
            sample.get(self.fallback_question_key, "")
        )
        
        # 获取solution，优先使用generation_solution
        solution = sample.get(
            self.solution_key,
            sample.get(self.fallback_solution_key, "")
        )
        
        return {
            "question": question,
            "solution": solution,
        }


def _build_preprocessed_dataset(
    data_path: str,
    tokenizer: PreTrainedTokenizer,
    max_length: Optional[int] = None,
    question_key: str = "question",
    solution_key: str = "generation_solution",
    fallback_question_key: str = "problem",
    fallback_solution_key: str = "solution",
    truncation: bool = True,
    padding: bool = False,
    num_proc: Optional[int] = None,
    cache_dir: Optional[str] = None,
) -> HFDataset:
    """
    使用datasets库高效构建SFT数据集，预先tokenize所有数据（使用chat_template）
    这个函数只在主进程调用
    """
    if num_proc is None:
        num_proc = max(1, multiprocessing.cpu_count() - 2)
    
    # 检查tokenizer是否有chat_template
    if not hasattr(tokenizer, 'chat_template') or tokenizer.chat_template is None:
        raise ValueError("Tokenizer must have a chat_template!")
    
    # 1. 加载数据
    print(f"📂 Loading dataset from {data_path}...")
    dataset = load_dataset(
        "json",
        data_files=data_path,
        cache_dir=cache_dir,
        split="train"
    )
    
    # 2. 提取字段
    def extract_fields(examples: Dict) -> Dict:
        """提取question和solution字段"""
        batch_size = len(examples[next(iter(examples.keys()))])
        questions = []
        solutions = []
        
        for i in range(batch_size):
            sample = {key: examples[key][i] for key in examples.keys()}
            questions.append(sample.get(question_key, sample.get(fallback_question_key, "")))
            solutions.append(sample.get(solution_key, sample.get(fallback_solution_key, "")))
        
        return {"question": questions, "solution": solutions}
    
    print("📝 Extracting fields...")
    dataset = dataset.map(
        extract_fields,
        batched=True,
        batch_size=1000,
        num_proc=num_proc,
        remove_columns=[col for col in dataset.column_names if col not in ["question", "solution"]],
        desc="Extracting fields",
    )
    
    # 3. Tokenize with chat_template
    def tokenize_chat_template(examples: Dict) -> Dict:
            """使用chat_template进行tokenize，添加长度信息用于后续过滤"""
            input_ids_list = []
            attention_mask_list = []
            prompt_lengths = []
            solution_lengths = []
            total_lengths = []
            valid_flags = []  # 标记哪些样本有效（不超过max_length）
            
            for question, solution in zip(examples["question"], examples["solution"]):
                messages = [
                    {"role": "user", "content": question},
                    {"role": "assistant", "content": solution}
                ]
                
                # Tokenize完整的对话（不截断，用于检查长度）
                full_prompt = tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=False
                )
                full_tokenized = tokenizer(
                    full_prompt,
                    add_special_tokens=False,
                    truncation=False,  # 不截断，用于检查长度
                    padding=padding,
                    return_tensors=None,
                )
                
                input_ids = full_tokenized["input_ids"]
                total_length = len(input_ids)
                
                # 检查是否超过max_length
                is_valid = max_length is None or total_length <= max_length
                valid_flags.append(is_valid)
                
                if is_valid:
                    attention_mask = full_tokenized["attention_mask"]
                    
                    # 计算prompt长度
                    user_only_messages = [messages[0]]
                    user_prompt = tokenizer.apply_chat_template(
                        user_only_messages, tokenize=False, add_generation_prompt=False
                    )
                    user_tokenized = tokenizer(
                        user_prompt, add_special_tokens=False, return_tensors=None
                    )
                    prompt_length = len(user_tokenized["input_ids"])
                    solution_length = total_length - prompt_length
                    
                    input_ids_list.append(input_ids)
                    attention_mask_list.append(attention_mask)
                    prompt_lengths.append(prompt_length)
                    solution_lengths.append(solution_length)
                    total_lengths.append(total_length)
                else:
                    # 即使无效也添加占位符，保持列表长度一致
                    input_ids_list.append([])
                    attention_mask_list.append([])
                    prompt_lengths.append(0)
                    solution_lengths.append(0)
                    total_lengths.append(0)
            
            result = {
                "input_ids": input_ids_list,
                "attention_mask": attention_mask_list,
                "prompt_length": prompt_lengths,
                "solution_length": solution_lengths,
                "total_length": total_lengths,
                "_valid": valid_flags,  # 临时字段，用于过滤
            }
            return result
        
    print("🔤 Tokenizing with chat_template...")
    dataset = dataset.map(
        tokenize_chat_template,
        batched=True,
        batch_size=100,
        num_proc=num_proc,
        desc="Tokenizing with chat_template",
    )
    
    # 过滤掉超过max_length的样本
    if max_length is not None:
        print(f"🔍 Filtering samples longer than {max_length}...")
        original_size = len(dataset)
        dataset = dataset.filter(lambda x: x["_valid"])
        dataset = dataset.remove_columns(["_valid"])  # 移除临时字段
        filtered_size = len(dataset)
        print(f"✅ Filtered: {original_size} -> {filtered_size} samples (removed {original_size - filtered_size})")
    
    print(f"✅ Dataset processed. Total samples: {len(dataset)}")
    return dataset


class PreprocessedSFTDataset(Dataset):
    """
    包装预处理好的HuggingFace Dataset，使其兼容PyTorch Dataset接口
    """
    def __init__(self, hf_dataset: HFDataset):
        self.hf_dataset = hf_dataset
    
    def __len__(self) -> int:
        return len(self.hf_dataset)
    
    def __getitem__(self, idx: int) -> Dict:
        item = self.hf_dataset[idx]
        # 确保返回的是list而不是tensor（在collate_fn中会转换为tensor）
        result = {
            "input_ids": item["input_ids"].tolist() if hasattr(item["input_ids"], "tolist") else item["input_ids"],
            "attention_mask": item["attention_mask"].tolist() if hasattr(item["attention_mask"], "tolist") else item["attention_mask"],
        }
        
        if "prompt_length" in item:
            result["prompt_length"] = int(item["prompt_length"])
            result["solution_length"] = int(item["solution_length"])
            result["total_length"] = int(item["total_length"])
        
        return result


def get_sft_dataloaders(
    config: DictConfig,
    tokenizer: PreTrainedTokenizer,
    accelerator=None,
):
    """
    创建用于SFT训练的DataLoader，支持accelerate多卡训练
    在分布式训练中，只在主进程进行预处理，其他进程等待并加载缓存
    
    Args:
        config: 配置对象
        tokenizer: tokenizer实例
        accelerator: Accelerator实例（可选），如果提供则自动判断主进程
        
    Returns:
        train_dataloader: 训练DataLoader（已设置drop_last=True，适配DDP）
        val_dataloader: 验证DataLoader（drop_last=False）
    """
    import torch
    from torch.utils.data import random_split
    
    # 判断是否为主进程
    if accelerator is not None:
        is_main_process = accelerator.is_main_process
    else:
        # 通过环境变量判断
        local_rank = int(os.environ.get("LOCAL_RANK", -1))
        is_main_process = local_rank == -1 or local_rank == 0
    
    data_path = config.dataset.params.data_path
    batch_size = config.training.batch_size
    max_length = config.dataset.preprocessing.max_seq_length
    num_workers = config.dataset.params.num_workers
    val_split_ratio = config.dataset.params.val_split_ratio
    question_key = config.dataset.params.question_key
    solution_key = config.dataset.params.solution_key
    fallback_question_key = config.dataset.params.fallback_question_key
    fallback_solution_key = config.dataset.params.fallback_solution_key
    truncation = config.dataset.params.truncation
    padding = config.dataset.params.padding
    seed = config.training.seed
    num_proc = config.dataset.params.get("num_proc", None)
    cache_dir = config.dataset.params.get("cache_dir", None)
    
    # 生成缓存路径
    cache_file = os.path.join(
        cache_dir or os.path.dirname(data_path),
        f".dataset_cache_{os.path.basename(data_path)}.arrow"
    )
    cache_ready_file = f"{cache_file}.ready"
    
    # 主进程：进行预处理
    if is_main_process:
        print("🚀 Main process: Starting dataset preprocessing...")
        try:
            # 尝试加载缓存
            if os.path.exists(cache_ready_file):
                try:
                    print(f"✅ Loading cached dataset from {cache_file}...")
                    # 尝试从arrow文件加载
                    if os.path.exists(cache_file):
                        hf_dataset = load_dataset("arrow", data_files=cache_file, split="train")
                    else:
                        # 如果arrow文件不存在，尝试从disk加载
                        cache_dir_path = cache_file.replace(".arrow", "")
                        if os.path.exists(cache_dir_path):
                            hf_dataset = load_from_disk(cache_dir_path)
                        else:
                            raise FileNotFoundError("Cache file not found")
                    print(f"✅ Loaded {len(hf_dataset)} samples from cache")
                except Exception as e:
                    print(f"⚠️ Failed to load cache: {e}, will rebuild...")
                    hf_dataset = None
            else:
                hf_dataset = None
            
            if hf_dataset is None:
                # 预处理数据
                hf_dataset = _build_preprocessed_dataset(
                    data_path=data_path,
                    tokenizer=tokenizer,
                    max_length=max_length,
                    question_key=question_key,
                    solution_key=solution_key,
                    fallback_question_key=fallback_question_key,
                    fallback_solution_key=fallback_solution_key,
                    truncation=truncation,
                    padding=padding,
                    num_proc=num_proc,
                    cache_dir=cache_dir,
                )
                # 保存缓存
                cache_base_dir = cache_dir or os.path.dirname(data_path)
                os.makedirs(cache_base_dir, exist_ok=True)
                print(f"💾 Saving dataset cache...")
                # 使用save_to_disk保存为目录格式（更可靠）
                cache_dir_path = cache_file.replace(".arrow", "")
                # save_to_disk是Dataset的实例方法，不是独立函数
                hf_dataset.save_to_disk(cache_dir_path)
                # 最后创建ready文件
                with open(cache_ready_file, 'w') as f:
                    f.write("ready")
                print(f"✅ Cache saved to {cache_dir_path}")
        except Exception as e:
            print(f"⚠️ Error in main process preprocessing: {e}")
            raise
    
    # 非主进程：等待主进程完成
    else:
        print(f"⏳ Process {os.getpid()} waiting for main process to finish preprocessing...")
        max_wait_time = 3600  # 最多等待1小时
        wait_time = 0
        wait_interval = 2  # 每2秒检查一次
        
        while wait_time < max_wait_time:
            if os.path.exists(cache_ready_file):
                try:
                    # 尝试从disk加载
                    cache_dir_path = cache_file.replace(".arrow", "")
                    if os.path.exists(cache_dir_path):
                        hf_dataset = load_from_disk(cache_dir_path)
                        print(f"✅ Process {os.getpid()} loaded cached dataset (waited {wait_time}s)")
                        break
                    elif os.path.exists(cache_file):
                        hf_dataset = load_dataset("arrow", data_files=cache_file, split="train")
                        print(f"✅ Process {os.getpid()} loaded cached dataset (waited {wait_time}s)")
                        break
                except Exception as e:
                    print(f"⚠️ Process {os.getpid()} failed to load cache: {e}, retrying...")
            time.sleep(wait_interval)
            wait_time += wait_interval
        else:
            raise RuntimeError(f"Timeout waiting for dataset cache. Process {os.getpid()} waited {max_wait_time}s")
    
    # 包装为PyTorch Dataset
    full_dataset = PreprocessedSFTDataset(hf_dataset)
    
    # 划分训练集和验证集
    total_size = len(full_dataset)
    val_size = int(total_size * val_split_ratio)
    train_size = total_size - val_size
    
    generator = torch.Generator().manual_seed(seed)
    train_dataset, val_dataset = random_split(
        full_dataset, 
        [train_size, val_size],
        generator=generator
    )
    
    if is_main_process:
        print(f"📊 Total samples: {total_size}")
        print(f"🚆 Train samples: {len(train_dataset)}, 🧪 Val samples: {len(val_dataset)}")
    
    # 创建自定义collate_fn用于padding
    def collate_fn(batch):
        """自定义collate函数，进行padding，并保留长度信息"""
        input_ids = [item["input_ids"] for item in batch]
        attention_mask = [item["attention_mask"] for item in batch]
        
        # Padding到batch中的最大长度
        max_len = max(len(ids) for ids in input_ids)
        
        padded_input_ids = []
        padded_attention_mask = []
        pad_lengths = []  # 每个样本的padding长度
        
        pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id else tokenizer.eos_token_id
        
        for ids, mask in zip(input_ids, attention_mask):
            pad_length = max_len - len(ids)
            pad_lengths.append(pad_length)
            padded_input_ids.append(ids + [pad_token_id] * pad_length)
            padded_attention_mask.append(mask + [0] * pad_length)
        
        result = {
            "input_ids": torch.tensor(padded_input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(padded_attention_mask, dtype=torch.long),
            "pad_lengths": torch.tensor(pad_lengths, dtype=torch.long),  # 每个样本的padding长度
        }
        
        # 保留prompt_length和solution_length信息
        if "prompt_length" in batch[0]:
            prompt_lengths = [item["prompt_length"] for item in batch]
            solution_lengths = [item["solution_length"] for item in batch]
            total_lengths = [item["total_length"] for item in batch]
            
            result["prompt_lengths"] = torch.tensor(prompt_lengths, dtype=torch.long)
            result["solution_lengths"] = torch.tensor(solution_lengths, dtype=torch.long)
            result["total_lengths"] = torch.tensor(total_lengths, dtype=torch.long)
        
        return result
    
    # 创建训练DataLoader
    # 注意：drop_last=True 是DDP训练的关键，确保每个进程的batch数量一致
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,  # 多卡训练必须设置为True
    )
    
    # 创建验证DataLoader
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,  # 验证集可以保留最后一个不完整的batch
    )
    
    return train_dataloader, val_dataloader

