import sys
import os
import json
import multiprocessing
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
os.environ["TOKENIZERS_PARALLELISM"] = "true"
os.environ["HF_DATASETS_OFFLINE"] = "true"
os.environ["HF_HUB_OFFLINE"] = "true"
os.environ["HF_OFFLINE"] = "true"
os.environ["TRANSFORMERS_OFFLINE"] = "true"
import math
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from accelerate import Accelerator
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from omegaconf import OmegaConf
from models.modeling_model.modeling_qwen3 import Qwen3ForCausalLM
from utils.utils import get_AR_attention_mask


class BatchRepeatCollator:
    """将单个样本重复N次形成batch，用于对Diffusion Loss进行蒙特卡洛估计"""
    def __init__(self, repeat_times=128):
        self.repeat_times = repeat_times
    
    def __call__(self, batch):
        # 强制要求 dataloader 的 batch_size=1
        if len(batch) != 1:
            raise ValueError(f"Collator expects batch size of 1, but got {len(batch)}")
        
        sample = batch[0]
        # 将单个样本重复 repeat_times 次
        repeated_batch = {
            k: [v] * self.repeat_times for k, v in sample.items()
        }
        
        # 转换为 torch tensor
        result = {}
        for k, v in repeated_batch.items():
            if isinstance(v[0], list):
                result[k] = torch.tensor(v, dtype=torch.long)
            else:
                result[k] = torch.tensor(v)
        
        return result


def build_packed_dataset(
    tokenizer,
    config,
    split="train",
    num_proc=None,
    cache_dir=None,
    text_column_name="text"
):
    if num_proc is None:
        num_proc = max(1, multiprocessing.cpu_count() - 2)
        
    data_path = config.dataset.params.data_path

    print(f"Loading dataset: {data_path}...")
    if "wikitext" in data_path.lower():
        raw_datasets = load_dataset(
            "wikitext",
            name=data_path,
            split=split,
            cache_dir=cache_dir,
            trust_remote_code=True
        )
    elif "openwebtext" in data_path.lower():
        raw_datasets = load_dataset(
            data_path,
            name=config.dataset.params.get("data_name", None),
            split="train",
            cache_dir=cache_dir,
            trust_remote_code=True
        )
        raw_datasets = raw_datasets.take(10000)
    elif "lambada" in data_path.lower():
        raw_datasets = load_dataset(
            data_path,
            split="test",
            cache_dir=cache_dir,
            trust_remote_code=True
        )
    else:
        raw_datasets = load_dataset(
            data_path,
            name=config.dataset.params.get("data_name", None),
            split=split,
            cache_dir=cache_dir,
            trust_remote_code=True
        )

    def tokenize_function(examples):
        tokenized_output = tokenizer(
            examples[text_column_name], 
            truncation=True,
            max_length=2049,
            padding=False,
            return_special_tokens_mask=False
        )
                            
        labels_list = [ids[1:] for ids in tokenized_output["input_ids"]]
        
        # 更新字典
        tokenized_output["labels"] = labels_list
        
        return tokenized_output

    print("Tokenizing data...")
    tokenized_datasets = raw_datasets.map(
        tokenize_function,
        batched=True,
        batch_size=1000,
        num_proc=num_proc,
        remove_columns=raw_datasets.column_names,
        desc="Running tokenizer",
    )
    
    return tokenized_datasets


if __name__ == "__main__":
    accelerator = Accelerator()
    
    evaluation_results = {}
    
    model_path_dict = {
        "ar-80BT": "public/models/Qwen/Qwen3-0.6B-Base",
    }
    
    dataset_list = ["cimec/lambada", "wikitext-103-raw-v1", "wikitext-2-raw-v1", "openwebtext",]
        
    for model_name, model_path in model_path_dict.items():

        # Load Model & Tokenizer
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, fix_mistral_regex=True)
            model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True, dtype=torch.float16)
            model.eval()
        except Exception as e:
            print(f"Failed to load {model_path}: {e}")
            continue
        # model.config.use_flex_attention = True
            # Load Dataset
            # 注意：cache_dir 最好指定绝对路径或确保存在
        for dataset_name in dataset_list:
            if dataset_name not in evaluation_results:
                evaluation_results[dataset_name] = {}

            # 【修复1】添加 training 配置，并设置 batch_size 为 1
            config = OmegaConf.create(
                {
                    "dataset": {
                        "params": {"data_path": dataset_name},
                    },
                    "model": {"attention_task": "random"},
                    "training": {"batch_size": 1} # 必须为1，因为 Collator 会做 expansion
                }
            )
            
            dataset = build_packed_dataset(tokenizer=tokenizer, config=config, split="test", cache_dir="public/.cache/huggingface/datasets")
            
            dataloader = DataLoader(
                dataset,
                shuffle=False,
                batch_size=1,
                collate_fn=BatchRepeatCollator(repeat_times=1), # 在这里扩充 batch
                num_workers=8,
                pin_memory=True,
            )
            
            # 【修复2】Prepare dataloader
            model, dataloader = accelerator.prepare(model, dataloader)

            local_total_loss_diff = torch.tensor(0.0, device=accelerator.device)
            local_total_loss_ar = torch.tensor(0.0, device=accelerator.device)
            local_total_count = torch.tensor(0.0, device=accelerator.device)
            
            accelerator.print(f"Evaluating Dataset: {dataset_name} | Model: {model_name}")
            
            for step, batch in tqdm(enumerate(dataloader), total=len(dataloader), disable=not accelerator.is_main_process):
                with torch.no_grad():
                    text_ids = batch["input_ids"] 
                    if text_ids.shape[-1] <= 32:
                        continue
                    label_ids = batch["labels"]
                    
                    current_batch_size = text_ids.size(0)

                    # --- Diffusion Loss ---
                    loss_diff = torch.tensor(2)
                    attention_mask = get_AR_attention_mask(seq_len=label_ids.size(-1), device=accelerator.device)
        
                    # logits = model(text_ids[:, :-1], attention_mask=attention_mask).logits
                    logits = model(text_ids[:, :-1]).logits
                    loss_ar = F.cross_entropy(
                        logits.view(-1, logits.size(-1)),
                        label_ids.view(-1),
                        reduction="mean", 
                        ignore_index=-100
                    )
                    accelerator.print(f"loss_ar: {loss_ar}")
                    
                    local_total_loss_diff += loss_diff.detach() * current_batch_size
                    local_total_loss_ar += loss_ar.detach() * current_batch_size
                    local_total_count += current_batch_size

            # Gather results
            accelerator.wait_for_everyone()
            
            global_count = accelerator.reduce(local_total_count, reduction="sum")
            global_loss_diff = accelerator.reduce(local_total_loss_diff, reduction="sum")
            global_loss_ar = accelerator.reduce(local_total_loss_ar, reduction="sum")
            
            # 只在主进程计算和打印
            if accelerator.is_main_process:
                avg_loss_diff = (global_loss_diff / global_count).item()
                ppl_diff = math.exp(avg_loss_diff)
                
                avg_loss_ar = (global_loss_ar / global_count).item()
                ppl_ar = math.exp(avg_loss_ar)
                
                print(f">>> Result {dataset_name} | {model_name}: PPL_Diff={ppl_diff:.4f}, PPL_AR={ppl_ar:.4f}")
                
                evaluation_results[dataset_name] = {
                    "loss_diff": avg_loss_diff,
                    "ppl_diff": ppl_diff,
                    "loss_ar": avg_loss_ar,
                    "ppl_ar": ppl_ar
                }
        if accelerator.is_main_process:
            output_path_base = "output_eval_final/PPL"
            if not os.path.exists(output_path_base):
                os.makedirs(output_path_base)
            with open(f"{output_path_base}/{model_name}_evaluation_results.json", "w", encoding="utf-8") as f:
                json.dump(evaluation_results, f, indent=4)
            
            torch.cuda.empty_cache()