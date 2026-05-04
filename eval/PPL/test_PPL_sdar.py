import sys
import os
import json
import multiprocessing
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
os.environ["TOKENIZERS_PARALLELISM"] = "false"
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
from torch.utils.data import DataLoader
from omegaconf import OmegaConf
from models.modeling_model.modeling_sdar import SDARForCausalLM
from utils.diffusion_utils import DiffusionLanguage
from utils.utils import get_AR_attention_mask, get_diffusion_attention_mask, get_full_attention_mask


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
            "en",
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
        
        eos_id = tokenizer.eos_token_id
            
        input_ids_list = [ids + [eos_id] for ids in tokenized_output["input_ids"]]
        attention_mask_list = [mask + [1] for mask in tokenized_output["attention_mask"]]
        
        labels_list = [ids[1:] for ids in input_ids_list]
        
        # 更新字典
        tokenized_output["input_ids"] = input_ids_list
        tokenized_output["attention_mask"] = attention_mask_list
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
        "sdar-40BT": "output_final/sdar-80BT/hf_model-40000",
    }
    
    dataset_list = ["EleutherAI/lambada_openai", "wikitext-103-raw-v1", "wikitext-2-raw-v1", "openwebtext",]
        
    for model_name, model_path in model_path_dict.items():
        if "sdar" in model_name.lower():
            modelclass = SDARForCausalLM
        else:
            raise ValueError()

        # Load Model & Tokenizer
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, fix_mistral_regex=True)
            model = modelclass.from_pretrained(model_path, trust_remote_code=True, dtype=torch.bfloat16)
        except Exception as e:
            print(f"Failed to load {model_path}: {e}")
            continue
        model.config.use_flex_attention = False
        model.train()
        model.gradient_checkpointing_disable()
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
                collate_fn=BatchRepeatCollator(repeat_times=64), # 在这里扩充 batch
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
                    label_ids = text_ids.clone()
                    
                    current_batch_size, seq_len = text_ids.shape
                    position_ids = torch.arange(seq_len, device=text_ids.device, dtype=torch.long).unsqueeze(0).expand(current_batch_size, -1)
                    
                    loss_diff = model(input_ids=text_ids,
                            position_ids=position_ids,
                            labels=label_ids).loss
                    loss_ar = torch.tensor(2)
                    
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